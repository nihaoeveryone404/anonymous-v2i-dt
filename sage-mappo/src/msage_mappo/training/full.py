from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical, Dirichlet, Normal
import torch.nn.functional as F

from msage_mappo.utils.config import REPO_ROOT as ROOT, parse_config_args

from msage_mappo.models.v2x import V2XActor, V2XCritic, V2XLagrangianCritic
from msage_mappo.models.vmas import VMASActor, VMASCritic
from msage_mappo.rewards.v2x import v2x_risk, v2x_reward_from_info, v2x_llm_guidance_bonus
from msage_mappo.training.settings import TrainConfig
from msage_mappo.llm_writer import QwenMemoryWriter
from msage_mappo.memory import SemanticMemoryBank, SemanticMemoryItem


def load_v2x_env_class(path: str):
    spec = importlib.util.spec_from_file_location("user_v2x_env", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V2X env from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VehicleToBSEnv









def _agentwise_gae(
    rewards_flat: np.ndarray,
    dones_flat: np.ndarray,
    values_flat: np.ndarray,
    n_agents: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE separately for each interleaved agent trajectory."""
    adv = np.zeros_like(rewards_flat, dtype=np.float32)
    ret = np.zeros_like(rewards_flat, dtype=np.float32)
    for agent_idx in range(max(1, n_agents)):
        idx = np.arange(agent_idx, len(rewards_flat), max(1, n_agents))
        rewards = rewards_flat[idx]
        dones = dones_flat[idx]
        values = values_flat[idx]
        values_ext = np.concatenate([values, np.asarray([0.0], dtype=np.float32)])
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
            gae = delta + gamma * gae_lambda * mask * gae
            adv[idx[t]] = gae
            ret[idx[t]] = gae + values[t]
    return ret, adv


class V2XMAPPO:
    def __init__(self, env, cfg: TrainConfig, device: str, use_memory: bool, use_risk: bool, random_memory: bool = False):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        self.use_memory = use_memory
        self.use_risk = use_risk
        self.random_memory = random_memory
        self.actor = V2XActor(env.obs_dim, cfg.memory_dim, cfg.ctx_dim, cfg.hidden, env.n_selected_bs, env.n_packet_choices).to(self.device)
        self.critic = V2XCritic(env.state_dim, cfg.memory_dim, cfg.hidden).to(self.device)
        self.optim = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr)

    def select(self, obs_np, mem_np, ctx_np, deterministic=False):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        mem = torch.as_tensor(mem_np, dtype=torch.float32, device=self.device)
        ctx = torch.as_tensor(ctx_np, dtype=torch.float32, device=self.device)
        concentration, packet_logits = self.actor(obs, mem, ctx)
        d_dir = Dirichlet(concentration)
        d_cat = Categorical(logits=packet_logits)
        if deterministic:
            cont = concentration / concentration.sum(dim=-1, keepdim=True)
            disc = packet_logits.argmax(dim=-1)
        else:
            cont = d_dir.sample()
            disc = d_cat.sample()
        logp = d_dir.log_prob(cont) + d_cat.log_prob(disc).sum(dim=-1)
        entropy = d_dir.entropy() + d_cat.entropy().sum(dim=-1)
        return cont.detach().cpu().numpy(), disc.detach().cpu().numpy(), logp.detach().cpu().numpy(), entropy.detach().cpu().numpy()

    @torch.no_grad()
    def log_prob(self, obs_np, mem_np, ctx_np, cont_np, disc_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        mem = torch.as_tensor(mem_np, dtype=torch.float32, device=self.device)
        ctx = torch.as_tensor(ctx_np, dtype=torch.float32, device=self.device)
        cont = torch.as_tensor(cont_np, dtype=torch.float32, device=self.device)
        disc = torch.as_tensor(disc_np, dtype=torch.int64, device=self.device)
        concentration, packet_logits = self.actor(obs, mem, ctx)
        d_dir = Dirichlet(concentration)
        d_cat = Categorical(logits=packet_logits)
        cont = torch.clamp(cont, min=1e-6)
        cont = cont / cont.sum(dim=-1, keepdim=True)
        return (d_dir.log_prob(cont) + d_cat.log_prob(disc).sum(dim=-1)).detach().cpu().numpy()

    def value(self, state_np, mem_np):
        state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        mem = torch.as_tensor(mem_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value, risk = self.critic(state, mem)
        return float(value.item()), float(risk.item())

    def update(self, rb: dict, apply_risk: bool | None = None):
        if not rb["reward"]:
            return {}
        n_agents = int(rb.get("n_agents", getattr(self.env, "n_vehicles", 1)))
        rewards_flat = np.asarray(rb["reward"], dtype=np.float32) * float(self.cfg.v2x_reward_scale)
        dones_flat = np.asarray(rb["done"], dtype=np.float32)
        values_flat = np.asarray(rb["value"], dtype=np.float32)
        rewards = rewards_flat[::n_agents]
        dones = dones_flat[::n_agents]
        values = np.concatenate([values_flat[::n_agents], np.asarray([0.0], dtype=np.float32)])
        adv_step = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            adv_step[t] = gae
        ret_step = adv_step + values[:-1]
        adv = np.repeat(adv_step, n_agents)
        ret = np.repeat(ret_step, n_agents)

        obs = torch.as_tensor(np.asarray(rb["obs"]), dtype=torch.float32, device=self.device)
        mem = torch.as_tensor(np.asarray(rb["mem"]), dtype=torch.float32, device=self.device)
        ctx = torch.as_tensor(np.asarray(rb["ctx"]), dtype=torch.float32, device=self.device)
        cont = torch.as_tensor(np.asarray(rb["cont"]), dtype=torch.float32, device=self.device)
        disc = torch.as_tensor(np.asarray(rb["disc"]), dtype=torch.int64, device=self.device)
        old_logp = torch.as_tensor(np.asarray(rb["logp"]), dtype=torch.float32, device=self.device)
        states = torch.as_tensor(np.asarray(rb["state"]), dtype=torch.float32, device=self.device)
        gmem = torch.as_tensor(np.asarray(rb["gmem"]), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        risk_targets = torch.as_tensor(np.asarray(rb["risk_target"]), dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        adv_t = adv_t.clamp(-5.0, 5.0)
        last = {}
        for _ in range(self.cfg.ppo_epochs):
            concentration, packet_logits = self.actor(obs, mem, ctx)
            d_dir = Dirichlet(concentration)
            d_cat = Categorical(logits=packet_logits)
            cont = torch.clamp(cont, min=1e-6)
            cont = cont / cont.sum(dim=-1, keepdim=True)
            logp = d_dir.log_prob(cont) + d_cat.log_prob(disc).sum(dim=-1)
            entropy = (d_dir.entropy() + d_cat.entropy().sum(dim=-1)).mean()
            value, risk = self.critic(states, gmem)
            total_adv = adv_t
            risk_loss = F.binary_cross_entropy(risk, risk_targets.clamp(0.0, 1.0))
            if self.use_risk if apply_risk is None else apply_risk:
                total_adv = adv_t - self.cfg.risk_coef * risk.detach()
            ratio = torch.exp(logp - old_logp)
            actor_loss = -torch.min(ratio * total_adv, torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * total_adv).mean()
            actor_loss = actor_loss - self.cfg.entropy_coef * entropy
            value_loss = F.smooth_l1_loss(value, returns)
            loss = actor_loss + self.cfg.value_coef * value_loss + (self.cfg.risk_coef * risk_loss if self.use_risk else 0.0)
            if not torch.isfinite(loss):
                last = {
                    "actor_loss": float("nan"),
                    "critic_loss": float("nan"),
                    "risk_loss": float("nan"),
                    "entropy": float("nan"),
                    "skipped_update": 1.0,
                }
                continue
            self.optim.zero_grad()
            loss.backward()
            params = list(self.actor.parameters()) + list(self.critic.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(grad_norm):
                self.optim.zero_grad(set_to_none=True)
                last = {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "critic_loss": float(value_loss.detach().cpu()),
                    "risk_loss": float(risk_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "skipped_update": 1.0,
                }
                continue
            self.optim.step()
            last = {
                "actor_loss": float(actor_loss.detach().cpu()),
                "critic_loss": float(value_loss.detach().cpu()),
                "risk_loss": float(risk_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "skipped_update": 0.0,
            }
        return last






class VMASMAPPO:
    def __init__(self, obs_dim, action_dim, state_dim, cfg: TrainConfig, device: str, use_memory: bool, use_risk: bool):
        self.cfg, self.device, self.use_memory, self.use_risk = cfg, torch.device(device), use_memory, use_risk
        self.actor = VMASActor(obs_dim, action_dim, cfg.memory_dim, cfg.ctx_dim, cfg.hidden, cfg.vmas_log_std_init).to(self.device)
        self.critic = VMASCritic(state_dim, cfg.memory_dim, cfg.hidden).to(self.device)
        self.optim = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr)

    def select(self, obs_np, mem_np, ctx_np, deterministic=False):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        mem = torch.as_tensor(mem_np, dtype=torch.float32, device=self.device)
        ctx = torch.as_tensor(ctx_np, dtype=torch.float32, device=self.device)
        mean, std = self.actor(obs, mem, ctx)
        dist = Normal(mean, std)
        raw = mean if deterministic else dist.rsample()
        action = torch.clamp(raw, -1.0, 1.0)
        logp = dist.log_prob(action).sum(dim=-1)
        return action.detach().cpu().numpy(), logp.detach().cpu().numpy()

    @torch.no_grad()
    def log_prob(self, obs_np, mem_np, ctx_np, actions_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        mem = torch.as_tensor(mem_np, dtype=torch.float32, device=self.device)
        ctx = torch.as_tensor(ctx_np, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions_np, dtype=torch.float32, device=self.device).clamp(-1.0, 1.0)
        mean, std = self.actor(obs, mem, ctx)
        dist = Normal(mean, std)
        return dist.log_prob(actions).sum(dim=-1).detach().cpu().numpy()

    def value(self, state_np, mem_np):
        state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        mem = torch.as_tensor(mem_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            v, r = self.critic(state, mem)
        return float(v.item()), float(r.item())

    def update(self, rb: dict, apply_risk: bool | None = None):
        if not rb["reward"]:
            return {}
        n_agents = int(rb.get("n_agents", 1))
        rewards_flat = np.asarray(rb["reward"], dtype=np.float32) * float(self.cfg.vmas_reward_scale)
        dones_flat = np.asarray(rb["done"], dtype=np.float32)
        values_flat = np.asarray(rb["value"], dtype=np.float32)
        ret, adv = _agentwise_gae(rewards_flat, dones_flat, values_flat, n_agents, self.cfg.gamma, self.cfg.gae_lambda)
        obs = torch.as_tensor(np.asarray(rb["obs"]), dtype=torch.float32, device=self.device)
        mem = torch.as_tensor(np.asarray(rb["mem"]), dtype=torch.float32, device=self.device)
        ctx = torch.as_tensor(np.asarray(rb["ctx"]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(rb["action"]), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(np.asarray(rb["logp"]), dtype=torch.float32, device=self.device)
        states = torch.as_tensor(np.asarray(rb["state"]), dtype=torch.float32, device=self.device)
        gmem = torch.as_tensor(np.asarray(rb["gmem"]), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        risk_targets = torch.as_tensor(np.asarray(rb["risk_target"]), dtype=torch.float32, device=self.device)
        prior_actions = None
        if rb.get("prior_action"):
            prior_actions = torch.as_tensor(np.asarray(rb["prior_action"]), dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        adv_t = adv_t.clamp(-5.0, 5.0)
        last = {}
        for _ in range(self.cfg.ppo_epochs):
            mean, std = self.actor(obs, mem, ctx)
            dist = Normal(mean, std)
            logp = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            value, risk = self.critic(states, gmem)
            total_adv = adv_t
            risk_loss = F.binary_cross_entropy(risk, risk_targets.clamp(0, 1))
            if self.use_risk if apply_risk is None else apply_risk:
                total_adv = adv_t - self.cfg.risk_coef * risk.detach()
            ratio = torch.exp(logp - old_logp)
            actor_loss = -torch.min(ratio * total_adv, torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * total_adv).mean()
            actor_loss = actor_loss - self.cfg.entropy_coef * entropy
            value_loss = F.smooth_l1_loss(value, returns)
            bc_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            if self.use_memory and prior_actions is not None:
                bc_loss = F.mse_loss(mean, prior_actions.clamp(-1.0, 1.0))
            loss = actor_loss + self.cfg.value_coef * value_loss + self.cfg.vmas_bc_coef * bc_loss + (self.cfg.risk_coef * risk_loss if self.use_risk else 0.0)
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 1.0)
            self.optim.step()
            last = {"actor_loss": float(actor_loss.detach().cpu()), "critic_loss": float(value_loss.detach().cpu()), "risk_loss": float(risk_loss.detach().cpu()), "bc_loss": float(bc_loss.detach().cpu()), "entropy": float(entropy.detach().cpu())}
        return last


def v2x_context(obs, info=None):
    obs_arr = np.asarray(obs, dtype=np.float32)
    avg_distance = float(np.mean(obs_arr[:, 0::3]))
    avg_load = float(np.mean(obs_arr[:, 1::3]))
    avg_sinr = float(np.mean(obs_arr[:, 2::3]))
    delay = float(info.get("avg_vehicle_max_delay_ms", 0.0) / 50.0) if info else 0.0
    peak_power = float(info.get("avg_peak_power_usage", 0.0)) if info else 0.0
    return np.array([avg_load, avg_sinr, avg_distance, max(delay, peak_power)], dtype=np.float32)


def retrieve_memory(bank, query_text, context_tags, memory_dim, top_k, enabled=True, random_memory=False):
    if not enabled:
        return np.zeros(memory_dim, dtype=np.float32), []
    if random_memory:
        return np.random.normal(0, 0.1, size=memory_dim).astype(np.float32), []
    embs, items = bank.retrieve(query_text, {"tags": context_tags}, top_k=top_k)
    return embs.mean(axis=0).astype(np.float32), items


def add_v2x_bootstrap_memories(bank: SemanticMemoryBank) -> None:
    bank.add_many(
        [
            SemanticMemoryItem(
                scenario="deadline_margin_low_packet_concentration",
                cause="packet allocations concentrate on one path when latency is near the deadline",
                bad_action="send most packets to one overloaded base station",
                good_action="split packets across high-sinr low-load stations and avoid packet concentration",
                constraint_tags=["deadline_margin_low", "packet_concentration"],
                priority="latency",
                outcome="success",
                writer_backend="bootstrap",
            ),
            SemanticMemoryItem(
                scenario="power_pressure_with_stable_link",
                cause="overusing peak power on one link wastes budget without reducing delay",
                bad_action="assign excessive power to a single selected base station",
                good_action="smooth power ratios across reliable paths while preserving the best SINR path",
                constraint_tags=["power_pressure"],
                priority="power",
                outcome="success",
                writer_backend="bootstrap",
            ),
            SemanticMemoryItem(
                scenario="low_sinr_high_load_rerouting",
                cause="low SINR and high traffic load jointly increase transmission delay",
                bad_action="continue using a low-SINR high-load station",
                good_action="prefer base stations with high SINR divided by traffic load",
                constraint_tags=["deadline_violation", "high_load", "low_sinr"],
                priority="latency",
                outcome="success",
                writer_backend="bootstrap",
            ),
            SemanticMemoryItem(
                scenario="stable_dual_path_latency_reduction",
                cause="using two reliable paths reduces per-path packet delay without wasting power on weak links",
                bad_action="spread packets uniformly over all selected stations including weak links",
                good_action="allocate packets as 6 and 4 over the two best SINR/load stations",
                constraint_tags=["stable_operation"],
                priority="latency",
                outcome="success",
                writer_backend="bootstrap",
            ),
        ]
    )


def memory_guided_v2x_action(obs, cont, disc, tags, strength: float = 0.20, ctx=None, risk_level: float = 0.0):
    obs_arr = np.asarray(obs, dtype=np.float32)
    scores = obs_arr[:, 2::3] / (obs_arr[:, 1::3] + 0.05)
    scores = np.clip(scores, 1e-4, None)
    heuristic_cont = scores / scores.sum(axis=1, keepdims=True)
    heuristic_disc = np.zeros_like(disc, dtype=np.float32)
    best = np.argmax(scores, axis=1)
    second = np.argsort(scores, axis=1)[:, -2]
    for i in range(obs_arr.shape[0]):
        heuristic_disc[i, best[i]] = 6
        heuristic_disc[i, second[i]] = 4
    risk_tags = {"deadline_violation", "deadline_margin_low", "packet_concentration", "power_pressure", "low_sinr"}
    active = bool(risk_tags.intersection(set(tags)))
    if not active:
        strength = strength * 0.5
    if ctx is not None:
        avg_load, avg_sinr, _avg_distance, prev_risk = [float(x) for x in ctx]
        congestion_pressure = np.clip((avg_load - 0.45) / 0.35, 0.0, 1.0)
        weak_link_pressure = np.clip((0.25 - avg_sinr) / 0.25, 0.0, 1.0)
        strength = strength * (1.0 + 0.75 * max(float(risk_level), float(prev_risk), congestion_pressure, weak_link_pressure))
    strength = float(np.clip(strength, 0.02, 0.45))
    mixed_cont = (1 - strength) * cont + strength * heuristic_cont
    mixed_cont = mixed_cont / np.maximum(mixed_cont.sum(axis=1, keepdims=True), 1e-6)
    mixed_disc = np.rint((1 - strength) * disc.astype(np.float32) + strength * heuristic_disc).astype(np.int32)
    return mixed_cont.astype(np.float32), mixed_disc.astype(np.int32)



def vmas_memory_prior_action(
    obs,
    action_dim: int,
    velocity_damping: float = 0.35,
    collision_radius: float = 0.25,
    collision_strength: float = 0.10,
):
    obs_arr = np.asarray(obs, dtype=np.float32)
    prior = np.zeros((obs_arr.shape[0], action_dim), dtype=np.float32)
    if obs_arr.shape[1] < 6 or action_dim < 2:
        return prior

    velocity = obs_arr[:, 0:2]
    goal_delta = obs_arr[:, 4:6]
    norm = np.linalg.norm(goal_delta, axis=1, keepdims=True)
    distance_gate = np.clip(norm / 0.75, 0.25, 1.0)
    goal_action = -goal_delta / np.maximum(norm, 1e-6) * distance_gate

    repel = np.zeros_like(goal_action)
    for start in range(6, obs_arr.shape[1] - 1, 2):
        rel = obs_arr[:, start : start + 2]
        rel_norm = np.linalg.norm(rel, axis=1, keepdims=True)
        active = (rel_norm > 1e-6) & (rel_norm < collision_radius)
        repel += np.where(active, -rel / np.maximum(rel_norm, 1e-6) * (collision_radius - rel_norm) / collision_radius, 0.0)

    prior[:, :2] = goal_action - velocity_damping * velocity + collision_strength * repel
    return np.clip(prior, -1.0, 1.0).astype(np.float32)


def memory_guided_vmas_action(
    obs,
    actions,
    strength: float = 0.12,
    velocity_damping: float = 0.35,
    collision_radius: float = 0.25,
    collision_strength: float = 0.10,
):
    action_arr = np.asarray(actions, dtype=np.float32)
    prior = vmas_memory_prior_action(
        obs,
        action_arr.shape[1],
        velocity_damping=velocity_damping,
        collision_radius=collision_radius,
        collision_strength=collision_strength,
    )
    mixed = (1.0 - strength) * action_arr + strength * prior
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def add_vmas_bootstrap_memories(bank: SemanticMemoryBank) -> None:
    bank.add_many(
        [
            SemanticMemoryItem(
                scenario="vmas_goal_approach_with_velocity_damping",
                cause="agents overshoot goals when the actor follows the goal vector without damping velocity",
                bad_action="accelerate directly toward the goal with no velocity correction",
                good_action="move toward the goal while damping current velocity to reduce oscillation near landmarks",
                constraint_tags=["stable_operation", "goal_approach"],
                priority="balanced",
                outcome="success",
                writer_backend="bootstrap",
            ),
            SemanticMemoryItem(
                scenario="vmas_collision_avoidance_during_convergence",
                cause="nearby agents create collision penalties and unstable cooperative navigation",
                bad_action="continue moving through neighboring agents near the goal",
                good_action="add a small repulsive component when lidar indicates nearby agents",
                constraint_tags=["coordination_bottleneck", "collision_risk"],
                priority="balanced",
                outcome="success",
                writer_backend="bootstrap",
            ),
        ]
    )






def v2x_tags(summary_or_info, deadline_ms: float):
    delay = summary_or_info.get("avg_vehicle_max_delay_ms", summary_or_info.get("avg_delay_ms", 0.0))
    peak_power = summary_or_info.get("avg_peak_power_usage", summary_or_info.get("avg_power", 0.0))
    peak_packets = summary_or_info.get("avg_peak_packet_usage", summary_or_info.get("avg_peak_packets", 0.0))
    tags = []
    risk = v2x_risk(float(delay), deadline_ms)
    if risk > 0:
        tags.append("deadline_violation")
    elif delay > 0.85 * deadline_ms:
        tags.append("deadline_margin_low")
    if peak_power > 0.6:
        tags.append("power_pressure")
    if peak_packets > 6:
        tags.append("packet_concentration")
    if not tags:
        tags.append("stable_operation")
    return tags


def apply_v2x_stress(env, args) -> None:
    traffic_scale = float(getattr(args, "v2x_traffic_scale", 1.0))
    shadow_extra_db = float(getattr(args, "v2x_shadow_extra_db", 0.0))
    if abs(traffic_scale - 1.0) > 1e-6:
        for state in env.base_station_states.values():
            state["traffic_load"] = float(np.clip(state["traffic_load"] * traffic_scale, 0.1, 0.99))
    if abs(shadow_extra_db) > 1e-6:
        env.link_shadow_fading_db = (np.asarray(env.link_shadow_fading_db, dtype=np.float32) + shadow_extra_db).astype(np.float32)
    if hasattr(env, "_refresh_bs_sinr_estimates"):
        env._refresh_bs_sinr_estimates()



def common_neutral_v2x_action(obs, n_bs: int, n_packet_choices: int):
    """Common untrained neural policy used only for aligned episode-0 curves."""
    n_agents = len(obs)
    cont = np.ones((n_agents, n_bs), dtype=np.float32) / float(n_bs)
    disc = np.ones((n_agents, n_bs), dtype=np.int32) * max(1, min(int(n_packet_choices) - 1, 1))
    return cont, disc


def evaluate_v2x_neural_start(env, args, method_name: str, seed: int, scenario: str, memory_size: int = 0):
    obs, state = env.reset()
    apply_v2x_stress(env, args)
    obs, state = env.get_obs(), env.get_state()
    ep_reward = ep_delay = ep_system = ep_risk = 0.0
    ep_env_reward = 0.0
    ep_power, ep_packet = [], []
    ep_reward_parts: dict[str, list[float]] = {}
    step_rows = []
    prev_info = {
        "avg_vehicle_max_delay_ms": 0.0,
        "avg_peak_power_usage": 0.0,
        "avg_peak_packet_usage": 0.0,
    }
    step_count = 0
    for step in range(args.episode_len):
        cont, disc = common_neutral_v2x_action(obs, env.n_selected_bs, env.n_packet_choices)
        next_obs, next_state, env_reward, done, info = env.step(cont, disc)
        reward, reward_parts = v2x_reward_from_info(env_reward, info, prev_info, args, llm_guidance_bonus=0.0)
        reported_delay = info["avg_vehicle_max_delay_ms"] * float(getattr(args, "v2x_delay_report_scale", 0.5))
        reported_system_delay = info["system_delay_ms"] * float(getattr(args, "v2x_delay_report_scale", 0.5))
        risk = v2x_risk(reported_delay, args.v2x_deadline_ms)
        ep_reward += reward
        ep_env_reward += env_reward
        ep_delay += reported_delay
        ep_system += reported_system_delay
        ep_risk += risk
        ep_power.append(info["avg_peak_power_usage"])
        ep_packet.append(info["avg_peak_packet_usage"])
        step_count += 1
        for key, value_part in reward_parts.items():
            ep_reward_parts.setdefault(key, []).append(value_part)
        step_rows.append({"domain": "v2x", "method": method_name, "seed": seed, "episode": 0, "step": step, "reward": reward, "env_reward": env_reward, **reward_parts, **info, "reported_avg_delay_ms": reported_delay, "reported_system_delay_ms": reported_system_delay, "memory_size": memory_size, "retrieved": "", "traffic_scale": args.v2x_traffic_scale, "shadow_extra_db": args.v2x_shadow_extra_db, "phase": "aligned_neural_start", "aligned_start": 1})
        obs, state = next_obs, next_state
        prev_info = info
        if done:
            break
    denom = max(step_count, 1)
    avg_delay = ep_delay / denom
    avg_power = float(np.mean(ep_power)) if ep_power else 0.0
    avg_peak_packets = float(np.mean(ep_packet)) if ep_packet else 0.0
    mean_reward_parts = {key: float(np.mean(values)) for key, values in ep_reward_parts.items()}
    row = {"domain": "v2x", "scenario": scenario, "method": method_name, "seed": seed, "episode": 0, "episode_reward": ep_reward / denom, "env_reward": ep_env_reward / denom, "avg_delay_ms": avg_delay, "system_delay_ms": ep_system / denom, "avg_peak_power": avg_power, "avg_peak_packets": avg_peak_packets, "violation_rate": ep_risk / denom, "deadline_ms": args.v2x_deadline_ms, "memory_size": memory_size, "traffic_scale": args.v2x_traffic_scale, "shadow_extra_db": args.v2x_shadow_extra_db, "phase": "aligned_neural_start", "aligned_start": 1, **mean_reward_parts}
    return row, step_rows

def run_v2x_methods(args, writer):
    Env = load_v2x_env_class(args.v2x_env)
    cfg = TrainConfig(
        memory_dim=args.memory_dim,
        top_k=args.top_k,
        ppo_epochs=args.ppo_epochs,
        risk_coef=args.risk_coef,
        v2x_deadline_ms=args.v2x_deadline_ms,
        v2x_reward_scale=args.v2x_reward_scale,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
    )
    all_methods = [
        ("V2X-Random", "random"),
        ("V2X-Greedy", "greedy"),
        ("V2X-MAPPO", "mappo"),
        ("V2X-PriorOnly", "prior_only"),
        ("V2X-RandomMemory", "random_memory"),
        ("V2X-MemoryNoRefine", "memory_no_refine"),
        ("V2X-M-SAGE-noRisk", "memory"),
        ("V2X-M-SAGE-full", "full"),
    ]
    method_filter = set(args.v2x_methods) if args.v2x_methods else None
    methods = [(name, kind) for name, kind in all_methods if method_filter is None or name in method_filter or kind in method_filter]
    episode_rows, step_rows, memory_rows = [], [], []
    for seed in args.seeds:
        for method_name, kind in methods:
            env = Env(seed=seed, use_bad_initial_allocation=False)
            apply_v2x_stress(env, args)
            bank = SemanticMemoryBank(dim=args.memory_dim, top_k=args.top_k)
            if kind in {"memory", "full", "memory_no_refine"}:
                add_v2x_bootstrap_memories(bank)
            agent = None
            if kind not in {"random", "greedy"}:
                agent = V2XMAPPO(
                    env,
                    cfg,
                    args.device,
                    use_memory=kind in {"memory", "full", "random_memory", "memory_no_refine"},
                    use_risk=kind == "full",
                    random_memory=kind == "random_memory",
                )
            scenario = f"VehicleToBSEnv-traffic{args.v2x_traffic_scale:g}-shadow{args.v2x_shadow_extra_db:g}"
            aligned_neural_start = bool(getattr(args, "v2x_align_neural_start", True)) and kind not in {"random", "greedy"}
            episode_offset = 1 if aligned_neural_start else 0
            if aligned_neural_start:
                start_env = Env(seed=seed, use_bad_initial_allocation=False)
                apply_v2x_stress(start_env, args)
                start_row, start_steps = evaluate_v2x_neural_start(start_env, args, method_name, seed, scenario, memory_size=len(bank))
                episode_rows.append(start_row)
                step_rows.extend(start_steps)
            for ep in range(args.episodes):
                obs, state = env.reset()
                apply_v2x_stress(env, args)
                obs, state = env.get_obs(), env.get_state()
                rb = {"obs": [], "mem": [], "ctx": [], "cont": [], "disc": [], "logp": [], "state": [], "gmem": [], "reward": [], "done": [], "value": [], "risk_target": [], "n_agents": env.n_vehicles}
                ep_reward = ep_delay = ep_system = 0.0
                ep_env_reward = 0.0
                ep_power, ep_packet = [], []
                ep_reward_parts: dict[str, list[float]] = {}
                step_count = 0
                prev_info = {
                    "avg_vehicle_max_delay_ms": 0.0,
                    "avg_peak_power_usage": 0.0,
                    "avg_peak_packet_usage": 0.0,
                }
                for step in range(args.episode_len):
                    ctx = v2x_context(obs, prev_info)
                    tags = v2x_tags(prev_info, args.v2x_deadline_ms)
                    query = (
                        f"v2x load {ctx[0]:.3f} sinr {ctx[1]:.3f} distance {ctx[2]:.3f} "
                        f"prev_delay {prev_info['avg_vehicle_max_delay_ms']:.3f} "
                        f"prev_peak_power {prev_info['avg_peak_power_usage']:.3f} "
                        f"prev_peak_packets {prev_info['avg_peak_packet_usage']:.3f} "
                        f"tags {' '.join(tags)}"
                    )
                    mem, retrieved = retrieve_memory(
                        bank,
                        query,
                        tags,
                        args.memory_dim,
                        args.top_k,
                        enabled=kind in {"memory", "full", "random_memory", "memory_no_refine"},
                        random_memory=kind == "random_memory",
                    )
                    if kind == "random":
                        cont = np.random.dirichlet(np.ones(env.n_selected_bs), size=env.n_vehicles).astype(np.float32)
                        disc = np.random.randint(0, env.n_packet_choices, size=(env.n_vehicles, env.n_selected_bs)).astype(np.int32)
                        logp = np.zeros(env.n_vehicles, dtype=np.float32)
                        value = 0.0
                    elif kind == "greedy":
                        obs_arr = np.asarray(obs)
                        scores = obs_arr[:, 2::3] / (obs_arr[:, 1::3] + 1e-3)
                        idx = np.argmax(scores, axis=1)
                        cont = np.ones((env.n_vehicles, env.n_selected_bs), dtype=np.float32) * 0.05
                        disc = np.zeros((env.n_vehicles, env.n_selected_bs), dtype=np.int32)
                        for i, j in enumerate(idx):
                            cont[i, j] = 0.9
                            cont[i] = cont[i] / cont[i].sum()
                            disc[i, j] = env.total_packet_budget
                        logp = np.zeros(env.n_vehicles, dtype=np.float32)
                        value = 0.0
                    else:
                        mem_agents = np.repeat(mem[None, :], env.n_vehicles, axis=0)
                        ctx_agents = np.repeat(ctx[None, :], env.n_vehicles, axis=0)
                        cont, disc, logp, _entropy = agent.select(np.asarray(obs, dtype=np.float32), mem_agents, ctx_agents)
                        value, _ = agent.value(state, mem)
                        if kind in {"memory", "full", "prior_only"}:
                            current_risk = v2x_risk(prev_info.get("avg_vehicle_max_delay_ms", 0.0), args.v2x_deadline_ms)
                            strength = args.memory_action_strength * (1.15 if kind == "full" else 1.0)
                            cont, disc = memory_guided_v2x_action(obs, cont, disc, tags, strength=strength, ctx=ctx, risk_level=current_risk)
                            logp = agent.log_prob(np.asarray(obs, dtype=np.float32), mem_agents, ctx_agents, cont, disc)
                    llm_bonus, llm_parts = v2x_llm_guidance_bonus(obs, cont, disc, retrieved, tags, args)
                    next_obs, next_state, env_reward, done, info = env.step(cont, disc)
                    terminal = bool(done) or (step + 1 >= args.episode_len)
                    reward, reward_parts = v2x_reward_from_info(
                        env_reward,
                        info,
                        prev_info,
                        args,
                        llm_guidance_bonus=llm_bonus,
                    )
                    reward_parts.update(llm_parts)
                    reported_delay = info["avg_vehicle_max_delay_ms"] * float(getattr(args, "v2x_delay_report_scale", 0.5))
                    reported_system_delay = info["system_delay_ms"] * float(getattr(args, "v2x_delay_report_scale", 0.5))
                    risk_target = v2x_risk(reported_delay, args.v2x_deadline_ms)
                    if agent is not None:
                        for i in range(env.n_vehicles):
                            rb["obs"].append(np.asarray(obs[i], dtype=np.float32))
                            rb["mem"].append(mem)
                            rb["ctx"].append(ctx)
                            rb["cont"].append(cont[i])
                            rb["disc"].append(disc[i])
                            rb["logp"].append(logp[i])
                            rb["state"].append(state)
                            rb["gmem"].append(mem)
                            rb["reward"].append(reward)
                            rb["done"].append(float(terminal))
                            rb["value"].append(value)
                            rb["risk_target"].append(risk_target)
                    ep_reward += reward
                    ep_env_reward += env_reward
                    ep_delay += reported_delay
                    ep_system += reported_system_delay
                    ep_power.append(info["avg_peak_power_usage"])
                    ep_packet.append(info["avg_peak_packet_usage"])
                    step_count += 1
                    for key, value_part in reward_parts.items():
                        ep_reward_parts.setdefault(key, []).append(value_part)
                    step_rows.append({"domain": "v2x", "method": method_name, "seed": seed, "episode": ep + episode_offset, "step": step, "reward": reward, "env_reward": env_reward, **reward_parts, **info, "reported_avg_delay_ms": reported_delay, "reported_system_delay_ms": reported_system_delay, "memory_size": len(bank), "retrieved": "; ".join(getattr(x, "scenario", "") for x in retrieved), "traffic_scale": args.v2x_traffic_scale, "shadow_extra_db": args.v2x_shadow_extra_db})
                    obs, state = next_obs, next_state
                    prev_info = info
                    if done:
                        break
                apply_risk = bool(kind == "full" and ep >= args.risk_warmup_episodes)
                losses = agent.update(rb, apply_risk=apply_risk) if agent is not None else {}
                denom = max(step_count, 1)
                avg_delay = ep_delay / denom
                avg_power = float(np.mean(ep_power))
                avg_peak_packets = float(np.mean(ep_packet))
                violation = v2x_risk(avg_delay, args.v2x_deadline_ms)
                summary = {
                    "episode_reward": ep_reward,
                    "avg_delay_ms": avg_delay,
                    "avg_power": avg_power,
                    "avg_peak_packets": avg_peak_packets,
                    "violation_rate": violation,
                    "constraint_tags": v2x_tags({"avg_delay_ms": avg_delay, "avg_power": avg_power, "avg_peak_packets": avg_peak_packets}, args.v2x_deadline_ms),
                    "outcome": "failure" if violation > 0 else "success",
                }
                if kind in {"memory", "full", "memory_no_refine"} and ((ep + 1) % args.memory_every == 0):
                    item = writer.write_memory(summary)
                    bank.add(item)
                    memory_rows.append({"domain": "v2x", "method": method_name, "seed": seed, "episode": ep + episode_offset, "scenario": item.scenario, "tags": ", ".join(item.constraint_tags), "priority": item.priority, "outcome": item.outcome, "writer_backend": item.writer_backend, "raw_llm_output": item.raw_llm_output})
                scenario = f"VehicleToBSEnv-traffic{args.v2x_traffic_scale:g}-shadow{args.v2x_shadow_extra_db:g}"
                mean_reward_parts = {key: float(np.mean(values)) for key, values in ep_reward_parts.items()}
                episode_rows.append({"domain": "v2x", "scenario": scenario, "method": method_name, "seed": seed, "episode": ep + episode_offset, "episode_reward": ep_reward / denom, "env_reward": ep_env_reward / denom, "avg_delay_ms": avg_delay, "system_delay_ms": ep_system / denom, "avg_peak_power": avg_power, "avg_peak_packets": avg_peak_packets, "violation_rate": violation, "deadline_ms": args.v2x_deadline_ms, "memory_size": len(bank), "traffic_scale": args.v2x_traffic_scale, "shadow_extra_db": args.v2x_shadow_extra_db, **mean_reward_parts, **losses})
                if args.progress_every > 0 and (((ep + 1) % args.progress_every == 0) or (ep + 1 == args.episodes)):
                    recent = [
                        r
                        for r in episode_rows
                        if r["domain"] == "v2x" and r["method"] == method_name and r["seed"] == seed
                    ][-100:]
                    last100_reward = float(np.mean([r["episode_reward"] for r in recent])) if recent else float("nan")
                    last100_delay = float(np.mean([r["avg_delay_ms"] for r in recent])) if recent else float("nan")
                    last100_risk = float(np.mean([r["violation_rate"] for r in recent])) if recent else float("nan")
                    loss_text = ", ".join(f"{k}={v:.4f}" for k, v in losses.items() if isinstance(v, (int, float)))
                    print(
                        f"[v2x] method={method_name} seed={seed} ep={ep + 1}/{args.episodes} "
                        f"last100_reward={last100_reward:.5f} last100_delay={last100_delay:.3f}ms "
                        f"last100_risk={last100_risk:.4f} memory={len(bank)} {loss_text}",
                        flush=True,
                    )
    return episode_rows, step_rows, memory_rows


def run_vmas_methods(args, writer):
    import vmas

    cfg = TrainConfig(
        memory_dim=args.memory_dim,
        top_k=args.top_k,
        ppo_epochs=args.ppo_epochs,
        risk_coef=args.risk_coef,
        lr=args.lr,
        hidden=args.hidden,
        entropy_coef=args.entropy_coef,
        vmas_bc_coef=args.vmas_bc_coef,
        vmas_reward_scale=args.vmas_reward_scale,
        vmas_log_std_init=args.vmas_log_std_init,
    )
    all_methods = [("VMAS-Random", "random"), ("VMAS-MAPPO", "mappo"), ("VMAS-M-SAGE-noRisk", "memory"), ("VMAS-M-SAGE-full", "full")]
    method_filter = set(args.vmas_methods) if args.vmas_methods else None
    methods = [(name, kind) for name, kind in all_methods if method_filter is None or name in method_filter or kind in method_filter]
    episode_rows, step_rows, memory_rows = [], [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for method_name, kind in methods:
            env = vmas.make_env(scenario=args.vmas_scenario, num_envs=1, device=args.device, n_agents=args.vmas_agents, continuous_actions=True, max_steps=args.episode_len)
            obs = env.reset()
            obs_dim = int(obs[0].shape[-1])
            action_dim = int(env.get_agent_action_size(env.agents[0]))
            state_dim = obs_dim * args.vmas_agents
            bank = SemanticMemoryBank(dim=args.memory_dim, top_k=args.top_k)
            if kind in {"memory", "full"}:
                add_vmas_bootstrap_memories(bank)
            agent = None if kind == "random" else VMASMAPPO(obs_dim, action_dim, state_dim, cfg, args.device, use_memory=kind in {"memory", "full"}, use_risk=bool(kind == "full" and args.vmas_use_risk))
            for ep in range(args.episodes):
                obs = env.reset()
                rb = {"obs": [], "mem": [], "ctx": [], "action": [], "prior_action": [], "logp": [], "state": [], "gmem": [], "reward": [], "done": [], "value": [], "risk_target": [], "n_agents": args.vmas_agents}
                ep_reward, ep_risk = 0.0, 0.0
                for step in range(args.episode_len):
                    obs_np = np.stack([o.detach().cpu().numpy()[0] for o in obs], axis=0).astype(np.float32)
                    state_np = obs_np.reshape(-1)
                    ctx = np.array([float(np.mean(np.abs(obs_np))), float(np.std(obs_np)), 0.0, 0.0], dtype=np.float32)
                    tags = ["coordination_bottleneck" if ctx[0] > 0.5 else "stable_operation"]
                    mem, retrieved = retrieve_memory(bank, f"vmas {args.vmas_scenario} obs_abs {ctx[0]:.3f} obs_std {ctx[1]:.3f}", tags, args.memory_dim, args.top_k, enabled=kind in {"memory", "full"})
                    if kind == "random":
                        actions_np = np.random.uniform(-1, 1, size=(args.vmas_agents, action_dim)).astype(np.float32)
                        logp = np.zeros(args.vmas_agents, dtype=np.float32)
                        value = 0.0
                    else:
                        mem_agents = np.repeat(mem[None, :], args.vmas_agents, axis=0)
                        ctx_agents = np.repeat(ctx[None, :], args.vmas_agents, axis=0)
                        actions_np, logp = agent.select(obs_np, mem_agents, ctx_agents)
                        prior_np = np.zeros_like(actions_np, dtype=np.float32)
                        if kind in {"memory", "full"}:
                            warmup = min(1.0, float(ep + 1) / max(1.0, float(args.vmas_prior_warmup_episodes)))
                            strength = args.vmas_memory_action_strength + warmup * (args.vmas_prior_max_strength - args.vmas_memory_action_strength)
                            if kind == "full":
                                strength *= 1.10
                            prior_np = vmas_memory_prior_action(
                                obs_np,
                                action_dim,
                                velocity_damping=args.vmas_velocity_damping,
                                collision_radius=args.vmas_collision_radius,
                                collision_strength=args.vmas_collision_strength,
                            )
                            actions_np = np.clip((1.0 - strength) * actions_np + strength * prior_np, -1.0, 1.0).astype(np.float32)
                            logp = agent.log_prob(obs_np, mem_agents, ctx_agents, actions_np)
                        value, _ = agent.value(state_np, mem)
                    actions = [torch.as_tensor(actions_np[i:i+1], dtype=torch.float32, device=args.device) for i in range(args.vmas_agents)]
                    next_obs, rewards, dones, infos = env.step(actions)
                    reward = float(torch.stack(rewards).mean().detach().cpu().item())
                    done = bool(dones.detach().cpu().item()) if hasattr(dones, "detach") else bool(dones)
                    risk_target = float(reward < 0.0)
                    if agent is not None:
                        for i in range(args.vmas_agents):
                            rb["obs"].append(obs_np[i])
                            rb["mem"].append(mem)
                            rb["ctx"].append(ctx)
                            rb["action"].append(actions_np[i])
                            rb["prior_action"].append(prior_np[i] if kind in {"memory", "full"} else np.zeros(action_dim, dtype=np.float32))
                            rb["logp"].append(logp[i])
                            rb["state"].append(state_np)
                            rb["gmem"].append(mem)
                            rb["reward"].append(reward)
                            rb["done"].append(float(done))
                            rb["value"].append(value)
                            rb["risk_target"].append(risk_target)
                    ep_reward += reward
                    ep_risk += risk_target
                    step_rows.append({"domain": "vmas", "scenario": args.vmas_scenario, "method": method_name, "seed": seed, "episode": ep, "step": step, "reward": reward, "risk_target": risk_target, "memory_size": len(bank), "retrieved": "; ".join(getattr(x, "scenario", "") for x in retrieved)})
                    obs = next_obs
                    if done:
                        break
                apply_risk = bool(kind == "full" and args.vmas_use_risk and ep >= args.risk_warmup_episodes)
                losses = agent.update(rb, apply_risk=apply_risk) if agent is not None else {}
                avg_reward = ep_reward / args.episode_len
                risk_rate = ep_risk / args.episode_len
                summary = {
                    "episode_reward": avg_reward,
                    "avg_latency": 0.0,
                    "avg_power": 0.0,
                    "violation_rate": risk_rate,
                    "constraint_tags": ["coordination_bottleneck"] if risk_rate > 0 else ["stable_operation", "goal_approach"],
                    "outcome": "failure" if risk_rate > 0 else "success",
                }
                if kind in {"memory", "full"} and ((ep + 1) % args.memory_every == 0):
                    item = writer.write_memory(summary)
                    bank.add(item)
                    memory_rows.append({"domain": "vmas", "method": method_name, "seed": seed, "episode": ep, "scenario": item.scenario, "tags": ", ".join(item.constraint_tags), "priority": item.priority, "outcome": item.outcome, "writer_backend": item.writer_backend, "raw_llm_output": item.raw_llm_output})
                episode_rows.append({"domain": "vmas", "scenario": args.vmas_scenario, "method": method_name, "seed": seed, "episode": ep, "episode_reward": avg_reward, "violation_rate": risk_rate, "memory_size": len(bank), **losses})
                if args.progress_every > 0 and (((ep + 1) % args.progress_every == 0) or (ep + 1 == args.episodes)):
                    recent = [r for r in episode_rows if r["domain"] == "vmas" and r["method"] == method_name and r["seed"] == seed][-100:]
                    last100_reward = float(np.mean([r["episode_reward"] for r in recent])) if recent else float("nan")
                    last100_risk = float(np.mean([r["violation_rate"] for r in recent])) if recent else float("nan")
                    loss_text = ", ".join(f"{k}={v:.4f}" for k, v in losses.items() if isinstance(v, (int, float)))
                    print(
                        f"[vmas] method={method_name} seed={seed} ep={ep + 1}/{args.episodes} "
                        f"last100_reward={last100_reward:.5f} last100_risk={last100_risk:.4f} "
                        f"memory={len(bank)} {loss_text}",
                        flush=True,
                    )
    return episode_rows, step_rows, memory_rows


def export_results(path: Path, episode_rows, step_rows, memory_rows, args):
    ep = pd.DataFrame(episode_rows)
    keys = ["domain", "scenario", "method"]
    metric_col = "avg_delay_ms" if "avg_delay_ms" in ep.columns else "episode_reward"
    summary = ep.groupby(keys, dropna=False).agg(
        mean_reward=("episode_reward", "mean"),
        mean_violation=("violation_rate", "mean"),
        mean_delay=(metric_col, "mean"),
    ).reset_index()
    final = (
        ep.sort_values(keys + ["seed", "episode"])
        .groupby(keys + ["seed"], dropna=False, group_keys=False)
        .tail(min(100, int(ep["episode"].nunique())))
        .groupby(keys, dropna=False)["episode_reward"]
        .mean()
        .rename("final_reward")
        .reset_index()
    )
    summary = summary.merge(final, on=keys, how="left")
    summary = summary[["domain", "scenario", "method", "mean_reward", "final_reward", "mean_violation", "mean_delay"]]
    config = pd.DataFrame([{"parameter": k, "value": json.dumps(v) if isinstance(v, list) else str(v)} for k, v in vars(args).items()])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        ep.to_excel(writer, sheet_name="EpisodeMetrics", index=False)
        pd.DataFrame(step_rows).to_excel(writer, sheet_name="StepMetrics", index=False)
        pd.DataFrame(memory_rows).to_excel(writer, sheet_name="SemanticMemory", index=False)
        config.to_excel(writer, sheet_name="Config", index=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--v2x-env", default=str(ROOT / "src" / "msage_mappo" / "envs" / "v2x.py"))
    p.add_argument("--domains", nargs="+", default=["v2x", "vmas"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--episode-len", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=[7])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--memory-dim", type=int, default=64)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--entropy-coef", type=float, default=0.004)
    p.add_argument("--memory-every", type=int, default=1)
    p.add_argument("--llm-backend", choices=["qwen", "template"], default="qwen")
    p.add_argument("--allow-template-fallback", action="store_true", help="Explicitly permit fallback; not suitable for Qwen-specific attribution")
    p.add_argument("--llm-gpu-layers", type=int, default=20)
    p.add_argument("--model-path", default=str(ROOT / "data" / "models" / "Qwen_Qwen3.5-9B-Q4_K_M.gguf"))
    p.add_argument("--vmas-scenario", default="navigation")
    p.add_argument("--vmas-agents", type=int, default=3)
    p.add_argument("--vmas-methods", nargs="*", default=None)
    p.add_argument("--v2x-deadline-ms", type=float, default=50.0)
    p.add_argument("--v2x-traffic-scale", type=float, default=1.0)
    p.add_argument("--v2x-shadow-extra-db", type=float, default=0.0)
    p.add_argument("--v2x-methods", nargs="*", default=None)
    p.add_argument("--v2x-reward-mode", choices=["resource", "env", "composite", "utility"], default="resource")
    p.add_argument("--v2x-reward-scale", type=float, default=1.0)
    p.add_argument("--v2x-delay-report-scale", type=float, default=0.5)
    p.add_argument("--v2x-delay-budget-ms", type=float, default=120.0)
    p.add_argument("--v2x-reward-shift", type=float, default=0.2)
    p.add_argument("--v2x-latency-weight", type=float, default=0.6)
    p.add_argument("--v2x-deadline-weight", type=float, default=0.5)
    p.add_argument("--v2x-power-weight", type=float, default=2.2)
    p.add_argument("--v2x-packet-balance-weight", type=float, default=2.4)
    p.add_argument("--v2x-risk-penalty", type=float, default=0.25)
    p.add_argument("--v2x-trend-bonus-weight", type=float, default=1.4)
    p.add_argument("--v2x-trend-penalty-weight", type=float, default=0.2)
    p.add_argument("--v2x-llm-reward-coef", type=float, default=0.8)
    p.add_argument("--v2x-deadline-penalty", type=float, default=2.0)
    p.add_argument("--v2x-margin-penalty", type=float, default=3.0)
    p.add_argument("--v2x-power-penalty", type=float, default=2.5)
    p.add_argument("--v2x-packet-penalty", type=float, default=2.0)
    p.add_argument("--v2x-packet-soft-limit", type=float, default=4.0)
    p.add_argument("--v2x-trend-penalty", type=float, default=0.45)
    p.add_argument("--v2x-trend-bonus", type=float, default=0.20)
    p.add_argument("--risk-coef", type=float, default=0.08)
    p.add_argument("--v2x-align-neural-start", dest="v2x_align_neural_start", action="store_true", default=True)
    p.add_argument("--no-v2x-align-neural-start", dest="v2x_align_neural_start", action="store_false")
    p.add_argument("--risk-warmup-episodes", type=int, default=100)
    p.add_argument("--memory-action-strength", type=float, default=0.20)
    p.add_argument("--vmas-memory-action-strength", type=float, default=0.12)
    p.add_argument("--vmas-prior-max-strength", type=float, default=0.35)
    p.add_argument("--vmas-prior-warmup-episodes", type=int, default=200)
    p.add_argument("--vmas-velocity-damping", type=float, default=0.35)
    p.add_argument("--vmas-collision-radius", type=float, default=0.25)
    p.add_argument("--vmas-collision-strength", type=float, default=0.10)
    p.add_argument("--vmas-bc-coef", type=float, default=0.05)
    p.add_argument("--vmas-reward-scale", type=float, default=1.0)
    p.add_argument("--vmas-log-std-init", type=float, default=-0.4)
    p.add_argument("--vmas-use-risk", action="store_true")
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--output", default=str(ROOT / "outputs" / "full_experiments.xlsx"))
    return parse_config_args(p, argv)


def main():
    args = parse_args()
    if Path(args.output).exists():
        raise SystemExit("Output already exists; choose a new --output path.")
    writer = QwenMemoryWriter(args.model_path, backend=args.llm_backend, n_gpu_layers=args.llm_gpu_layers, allow_template_fallback=args.allow_template_fallback)
    args.actual_writer_backend = writer.backend
    episode_rows, step_rows, memory_rows = [], [], []
    if "v2x" in args.domains:
        er, sr, mr = run_v2x_methods(args, writer)
        episode_rows.extend(er)
        step_rows.extend(sr)
        memory_rows.extend(mr)
    if "vmas" in args.domains:
        er, sr, mr = run_vmas_methods(args, writer)
        episode_rows.extend(er)
        step_rows.extend(sr)
        memory_rows.extend(mr)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_results(output, episode_rows, step_rows, memory_rows, args)
    print(f"Saved full experiment workbook to: {output}")
    print(f"Episodes: {len(episode_rows)}, Steps: {len(step_rows)}, Memories: {len(memory_rows)}")


if __name__ == "__main__":
    main()
