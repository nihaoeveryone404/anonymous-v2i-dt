from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from msage_mappo.utils.config import REPO_ROOT as ROOT, parse_config_args

from msage_mappo.models.value_decomposition import QNet, QMixer
from msage_mappo.training.full import VMASMAPPO, TrainConfig, common_neutral_v2x_action, load_v2x_env_class, v2x_reward_from_info, v2x_risk


@dataclass
class BaselineConfig:
    gamma: float = 0.98
    lr: float = 7e-4
    batch_size: int = 256
    replay_size: int = 80000
    hidden: int = 192
    target_update: int = 25
    train_after: int = 64
    train_steps: int = 6
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 500
    double_q: bool = True
    huber_delta: float = 1.0
    grad_clip: float = 2.0
    td_target_clip: float = 20.0
    vmas_reward_scale: float = 3.0


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: list[tuple[Any, ...]] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, *transition: Any) -> None:
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.data), size=batch_size, replace=False)
        cols = list(zip(*(self.data[int(i)] for i in idx)))
        return cols






class ValueBaseline:
    def __init__(self, obs_dim: int, state_dim: int, n_agents: int, n_actions: int, method: str, cfg: BaselineConfig, device: str):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.method = method
        self.cfg = cfg
        self.device = torch.device(device)
        self.q = QNet(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.target_q = QNet(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.mixer = QMixer(n_agents, state_dim, cfg.hidden).to(self.device) if method == "qmix" else None
        self.target_mixer = QMixer(n_agents, state_dim, cfg.hidden).to(self.device) if method == "qmix" else None
        if self.target_mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        params = list(self.q.parameters()) + ([] if self.mixer is None else list(self.mixer.parameters()))
        self.optim = torch.optim.Adam(params, lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_size)
        self.updates = 0

    def act(self, obs_np: np.ndarray, epsilon: float):
        if random.random() < epsilon:
            return np.random.randint(0, self.n_actions, size=obs_np.shape[0], dtype=np.int64)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.q(obs)
        return q.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)

    def update(self):
        if len(self.replay) < max(self.cfg.train_after, self.cfg.batch_size):
            return {}
        last = {}
        for _ in range(self.cfg.train_steps):
            obs, state, actions, reward, next_obs, next_state, done = self.replay.sample(self.cfg.batch_size)
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
            next_obs_t = torch.as_tensor(np.asarray(next_obs), dtype=torch.float32, device=self.device)
            actions_t = torch.as_tensor(np.asarray(actions), dtype=torch.int64, device=self.device)
            reward_t = torch.as_tensor(np.asarray(reward), dtype=torch.float32, device=self.device)
            done_t = torch.as_tensor(np.asarray(done), dtype=torch.float32, device=self.device)

            if self.method == "iql":
                q_taken = self.q(obs_t).gather(1, actions_t[:, None]).squeeze(1)
                with torch.no_grad():
                    if self.cfg.double_q:
                        next_action = self.q(next_obs_t).argmax(dim=1, keepdim=True)
                        next_q = self.target_q(next_obs_t).gather(1, next_action).squeeze(1)
                    else:
                        next_q = self.target_q(next_obs_t).max(dim=1).values
                    target = reward_t + self.cfg.gamma * (1 - done_t) * next_q
                    target = target.clamp(-self.cfg.td_target_clip, self.cfg.td_target_clip)
                loss = F.smooth_l1_loss(q_taken, target, beta=self.cfg.huber_delta)
            else:
                bs = actions_t.shape[0]
                obs_joint = obs_t.view(bs * self.n_agents, self.obs_dim)
                next_obs_joint = next_obs_t.view(bs * self.n_agents, self.obs_dim)
                action_joint = actions_t.view(bs * self.n_agents)
                agent_q = self.q(obs_joint).gather(1, action_joint[:, None]).view(bs, self.n_agents)
                with torch.no_grad():
                    if self.cfg.double_q:
                        next_action_joint = self.q(next_obs_joint).argmax(dim=1, keepdim=True)
                        next_agent_q = self.target_q(next_obs_joint).gather(1, next_action_joint).view(bs, self.n_agents)
                    else:
                        next_agent_q = self.target_q(next_obs_joint).max(dim=1).values.view(bs, self.n_agents)
                if self.method == "vdn":
                    q_tot = agent_q.sum(dim=1)
                    next_q_tot = next_agent_q.sum(dim=1)
                elif self.method == "qmix":
                    state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device)
                    next_state_t = torch.as_tensor(np.asarray(next_state), dtype=torch.float32, device=self.device)
                    q_tot = self.mixer(agent_q, state_t)
                    with torch.no_grad():
                        next_q_tot = self.target_mixer(next_agent_q, next_state_t)
                else:
                    raise ValueError(self.method)
                target = reward_t + self.cfg.gamma * (1 - done_t) * next_q_tot
                target = target.clamp(-self.cfg.td_target_clip, self.cfg.td_target_clip)
                loss = F.smooth_l1_loss(q_tot, target, beta=self.cfg.huber_delta)
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.q.parameters()) + ([] if self.mixer is None else list(self.mixer.parameters())), self.cfg.grad_clip)
            self.optim.step()
            self.updates += 1
            if self.updates % self.cfg.target_update == 0:
                self.target_q.load_state_dict(self.q.state_dict())
                if self.target_mixer is not None:
                    self.target_mixer.load_state_dict(self.mixer.state_dict())
            last = {"q_loss": float(loss.detach().cpu())}
        return last


def epsilon_for(ep: int, cfg: BaselineConfig) -> float:
    frac = min(1.0, ep / max(1, cfg.eps_decay_episodes))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def v2x_macro_actions(obs: np.ndarray, macro: np.ndarray, n_bs: int, n_packet_choices: int):
    obs_arr = np.asarray(obs, dtype=np.float32)
    n_agents = obs_arr.shape[0]
    cont = np.ones((n_agents, n_bs), dtype=np.float32) / float(n_bs)
    disc = np.ones((n_agents, n_bs), dtype=np.int32) * max(1, min(n_packet_choices - 1, 2))
    load = obs_arr[:, 1::3]
    sinr = obs_arr[:, 2::3]
    distance = obs_arr[:, 0::3]
    scores = np.clip(sinr / (load + 0.05), 1e-4, None)
    low_load_scores = np.clip((1.0 - load) * np.sqrt(np.maximum(sinr, 1e-4)) / (distance + 0.10), 1e-4, None)
    order = np.argsort(scores, axis=1)
    max_packets = int(n_packet_choices - 1)

    def normalize(row: np.ndarray) -> np.ndarray:
        row = np.clip(row.astype(np.float32), 1e-4, None)
        return row / max(float(row.sum()), 1e-6)

    def packet_row(values: list[int]) -> np.ndarray:
        out = np.zeros(n_bs, dtype=np.int32)
        for rank, value in enumerate(values[:n_bs]):
            out[rank] = int(np.clip(value, 0, max_packets))
        return out

    for i in range(n_agents):
        ranked = order[i, ::-1]
        best = int(ranked[0])
        second = int(order[i, -2]) if n_bs > 1 else best
        third = int(order[i, -3]) if n_bs > 2 else second
        m = int(macro[i])
        if m == 0:
            cont[i] = normalize(scores[i])
            disc[i] = min(max_packets, 4)
        elif m == 1:
            cont[i] = 0.05
            cont[i, best] = 0.80
            cont[i, second] = 0.15
            cont[i] /= cont[i].sum()
            disc[i] = 0
            disc[i, best] = min(max_packets, 7)
            disc[i, second] = min(max_packets, 3)
        elif m == 2:
            cont[i] = 0.05
            cont[i, best] = 0.65
            cont[i, second] = 0.30
            cont[i] /= cont[i].sum()
            disc[i] = 1
            disc[i, best] = min(max_packets, 6)
            disc[i, second] = min(max_packets, 4)
        elif m == 3:
            cont[i] = normalize(scores[i])
            disc[i] = min(max_packets, 4)
            disc[i, third] = min(max_packets, 2)
        elif m == 4:
            cont[i] = 0.08
            cont[i, best] = 0.55
            cont[i, second] = 0.35
            cont[i] /= cont[i].sum()
            disc[i] = min(max_packets, 3)
            disc[i, best] = min(max_packets, 5)
            disc[i, second] = min(max_packets, 4)
        elif m == 5:
            low_order = np.argsort(low_load_scores[i])[::-1]
            b0 = int(low_order[0])
            b1 = int(low_order[1]) if n_bs > 1 else b0
            cont[i] = 0.10
            cont[i, b0] = 0.60
            cont[i, b1] = 0.30
            cont[i] /= cont[i].sum()
            disc[i] = 1
            disc[i, b0] = min(max_packets, 6)
            disc[i, b1] = min(max_packets, 4)
        elif m == 6:
            weights = np.sqrt(scores[i])
            cont[i] = normalize(weights)
            disc[i] = min(max_packets, 3)
            disc[i, best] = min(max_packets, 5)
        elif m == 7:
            cont[i] = 0.04
            cont[i, best] = 0.58
            cont[i, second] = 0.28
            cont[i, third] = 0.10
            cont[i] /= cont[i].sum()
            disc[i] = 0
            disc[i, best] = min(max_packets, 5)
            disc[i, second] = min(max_packets, 3)
            disc[i, third] = min(max_packets, 2)
        elif m == 8:
            cont[i] = np.ones(n_bs, dtype=np.float32) / float(n_bs)
            disc[i] = 0
            disc[i, best] = min(max_packets, 4)
            disc[i, second] = min(max_packets, 3)
            disc[i, third] = min(max_packets, 3)
        elif m == 9:
            cont[i] = normalize(0.70 * scores[i] + 0.30 * low_load_scores[i])
            disc[i] = min(max_packets, 2)
            disc[i, best] = min(max_packets, 6)
            disc[i, second] = min(max_packets, 3)
        elif m == 10:
            cont[i] = 0.02
            cont[i, best] = 0.96
            cont[i] /= cont[i].sum()
            disc[i] = 0
            disc[i, best] = max_packets
        elif m == 11:
            low_order = np.argsort(low_load_scores[i])[::-1]
            b0 = int(low_order[0])
            cont[i] = 0.02
            cont[i, b0] = 0.96
            cont[i] /= cont[i].sum()
            disc[i] = 0
            disc[i, b0] = max_packets
        elif m == 12:
            cont[i] = np.ones(n_bs, dtype=np.float32) / float(n_bs)
            disc[i] = 1
            disc[i, best] = min(max_packets, 4)
            disc[i, second] = min(max_packets, 3)
            disc[i, third] = min(max_packets, 3)
        elif m == 13:
            cont[i] = 0.03
            cont[i, best] = 0.485
            cont[i, second] = 0.485
            cont[i] /= cont[i].sum()
            disc[i] = 0
            disc[i, best] = min(max_packets, 5)
            disc[i, second] = min(max_packets, 5)
        elif m == 14:
            cont[i] = 0.02
            cont[i, best] = 0.58
            cont[i, third] = 0.40
            cont[i] /= cont[i].sum()
            disc[i] = 0
            disc[i, best] = min(max_packets, 6)
            disc[i, third] = min(max_packets, 4)
        elif m == 15:
            low_order = np.argsort(low_load_scores[i])[::-1]
            b0 = int(low_order[0])
            b1 = int(low_order[1]) if n_bs > 1 else b0
            b2 = int(low_order[2]) if n_bs > 2 else b1
            cont[i] = 0.05
            cont[i, b0] = 0.45
            cont[i, b1] = 0.40
            cont[i, b2] = 0.15
            cont[i] /= cont[i].sum()
            disc[i] = 0
            disc[i, b0] = min(max_packets, 4)
            disc[i, b1] = min(max_packets, 4)
            disc[i, b2] = min(max_packets, 2)
        elif m == 16:
            cont[i] = normalize(np.sqrt(scores[i]))
            disc[i] = min(max_packets, 3)
            disc[i, best] = min(max_packets, 4)
            disc[i, second] = min(max_packets, 4)
            disc[i, third] = min(max_packets, 2)
        elif m == 17:
            cont[i] = normalize(0.45 * scores[i] + 0.55 * low_load_scores[i])
            disc[i] = min(max_packets, 2)
            disc[i, best] = min(max_packets, 5)
            disc[i, second] = min(max_packets, 3)
            disc[i, third] = min(max_packets, 2)
        else:
            cont[i] = 1.0 / n_bs
            disc[i] = min(max_packets, 2)
    return cont.astype(np.float32), disc.astype(np.int32)


def vmas_macro_action_table(scales: tuple[float, ...] = (0.30, 0.60, 0.90)) -> np.ndarray:
    dirs = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float32,
    )
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    table = [np.zeros(2, dtype=np.float32)]
    for scale in scales:
        table.extend((dirs * float(scale)).astype(np.float32))
    return np.asarray(table, dtype=np.float32)


def vmas_macro_actions(obs: np.ndarray, macro: np.ndarray, action_dim: int, table: np.ndarray):
    actions = np.zeros((obs.shape[0], action_dim), dtype=np.float32)
    action_table = np.asarray(table, dtype=np.float32)
    cols = min(2, action_dim)
    actions[:, :cols] = action_table[np.asarray(macro, dtype=np.int64) % len(action_table), :cols]
    return actions



def action_entropy(actions: list[int], n_actions: int) -> float:
    if not actions:
        return 0.0
    counts = np.bincount(np.asarray(actions, dtype=np.int64), minlength=max(1, n_actions)).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def evaluate_v2x_baseline_start(env, args, method_name: str, seed: int):
    obs, state = env.reset()
    ep_reward = ep_delay = ep_system = ep_risk = 0.0
    ep_env_reward = 0.0
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
        reward, reward_parts = v2x_reward_from_info(env_reward, info, prev_info, args)
        reported_delay = info["avg_vehicle_max_delay_ms"] * float(getattr(args, "v2x_delay_report_scale", 0.5))
        reported_system_delay = info["system_delay_ms"] * float(getattr(args, "v2x_delay_report_scale", 0.5))
        risk = v2x_risk(reported_delay, args.v2x_deadline_ms)
        ep_reward += reward
        ep_env_reward += env_reward
        ep_delay += reported_delay
        ep_system += reported_system_delay
        ep_risk += risk
        step_count += 1
        for key, value_part in reward_parts.items():
            ep_reward_parts.setdefault(key, []).append(value_part)
        step_rows.append({"domain": "v2x", "method": method_name, "seed": seed, "episode": 0, "step": step, "reward": reward, "env_reward": env_reward, **reward_parts, "avg_vehicle_max_delay_ms": info["avg_vehicle_max_delay_ms"], "reported_avg_delay_ms": reported_delay, "violation_rate": risk, "phase": "aligned_neural_start", "aligned_start": 1, "macro_action": -1, "action_unique": 1, "action_entropy": 0.0})
        obs, state = next_obs, next_state
        prev_info = info
        if done:
            break
    mean_reward_parts = {key: float(np.mean(values)) for key, values in ep_reward_parts.items()}
    denom = max(step_count, 1)
    row = {"domain": "v2x", "scenario": "VehicleToBSEnv", "method": method_name, "seed": seed, "episode": 0, "episode_reward": ep_reward / denom, "env_reward": ep_env_reward / denom, "avg_delay_ms": ep_delay / denom, "system_delay_ms": ep_system / denom, "violation_rate": ep_risk / denom, "phase": "aligned_neural_start", "aligned_start": 1, "action_unique": 1, "action_entropy": 0.0, **mean_reward_parts}
    return row, step_rows

def run_v2x_value_baselines(args, cfg):
    Env = load_v2x_env_class(args.v2x_env)
    methods = args.methods or ["iql", "vdn", "qmix"]
    rows, steps = [], []
    for seed in args.seeds:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        for method in methods:
            env = Env(seed=seed, use_bad_initial_allocation=False)
            obs, state = env.reset()
            n_agents, obs_dim, state_dim = env.n_vehicles, env.obs_dim, env.state_dim
            n_actions = int(args.v2x_macro_actions)
            agent = ValueBaseline(obs_dim, state_dim, n_agents, n_actions, method, cfg, args.device)
            method_name = f"V2X-{method.upper()}"
            aligned_neural_start = bool(getattr(args, "v2x_align_neural_start", True))
            episode_offset = 1 if aligned_neural_start else 0
            if aligned_neural_start:
                start_env = Env(seed=seed, use_bad_initial_allocation=False)
                start_row, start_steps = evaluate_v2x_baseline_start(start_env, args, method_name, seed)
                rows.append(start_row)
                steps.extend(start_steps)
            for ep in range(args.episodes):
                obs, state = env.reset()
                ep_reward = ep_delay = ep_system = ep_risk = 0.0
                ep_env_reward = 0.0
                ep_reward_parts: dict[str, list[float]] = {}
                step_count = 0
                prev_info = {
                    "avg_vehicle_max_delay_ms": 0.0,
                    "avg_peak_power_usage": 0.0,
                    "avg_peak_packet_usage": 0.0,
                }
                losses = {}
                ep_actions: list[int] = []
                for step in range(args.episode_len):
                    eps = epsilon_for(ep, cfg)
                    macro = agent.act(np.asarray(obs, dtype=np.float32), eps)
                    ep_actions.extend(int(x) for x in np.asarray(macro).reshape(-1))
                    cont, disc = v2x_macro_actions(obs, macro, env.n_selected_bs, env.n_packet_choices)
                    next_obs, next_state, env_reward, done, info = env.step(cont, disc)
                    terminal = bool(done) or (step + 1 >= args.episode_len)
                    reward, reward_parts = v2x_reward_from_info(env_reward, info, prev_info, args)
                    reported_delay = info["avg_vehicle_max_delay_ms"] * float(getattr(args, "v2x_delay_report_scale", 0.5))
                    reported_system_delay = info["system_delay_ms"] * float(getattr(args, "v2x_delay_report_scale", 0.5))
                    risk = v2x_risk(reported_delay, args.v2x_deadline_ms)
                    if method == "iql":
                        for i in range(n_agents):
                            agent.replay.add(obs[i], state, macro[i], reward * args.v2x_reward_scale, next_obs[i], next_state, float(terminal))
                    else:
                        agent.replay.add(np.asarray(obs, dtype=np.float32), state, macro, reward * args.v2x_reward_scale, np.asarray(next_obs, dtype=np.float32), next_state, float(terminal))
                    losses = agent.update()
                    ep_reward += reward
                    ep_env_reward += env_reward
                    ep_delay += reported_delay
                    ep_system += reported_system_delay
                    ep_risk += risk
                    step_count += 1
                    for key, value_part in reward_parts.items():
                        ep_reward_parts.setdefault(key, []).append(value_part)
                    steps.append({"domain": "v2x", "method": method_name, "seed": seed, "episode": ep + episode_offset, "step": step, "reward": reward, "env_reward": env_reward, **reward_parts, "avg_vehicle_max_delay_ms": info["avg_vehicle_max_delay_ms"], "reported_avg_delay_ms": reported_delay, "violation_rate": risk, "phase": "train", "aligned_start": 0, "macro_action": ";".join(str(int(x)) for x in np.asarray(macro).reshape(-1))})
                    obs, state = next_obs, next_state
                    prev_info = info
                    if done:
                        break
                mean_reward_parts = {key: float(np.mean(values)) for key, values in ep_reward_parts.items()}
                rows.append(
                    {
                        "domain": "v2x",
                        "scenario": "VehicleToBSEnv",
                        "method": method_name,
                        "seed": seed,
                        "episode": ep + episode_offset,
                        "episode_reward": ep_reward / max(step_count, 1),
                        "env_reward": ep_env_reward / max(step_count, 1),
                        "avg_delay_ms": ep_delay / max(step_count, 1),
                        "system_delay_ms": ep_system / max(step_count, 1),
                        "violation_rate": ep_risk / max(step_count, 1),
                        "phase": "train",
                        "aligned_start": 0,
                        "action_unique": len(set(ep_actions)) if ep_actions else 0,
                        "action_entropy": action_entropy(ep_actions, n_actions),
                        **mean_reward_parts,
                        **losses,
                    }
                )
                if args.progress_every > 0 and (((ep + 1) % args.progress_every == 0) or (ep + 1 == args.episodes)):
                    recent = [r for r in rows if r["method"] == method_name and r["seed"] == seed][-100:]
                    last100_reward = float(np.mean([r["episode_reward"] for r in recent])) if recent else float("nan")
                    last100_delay = float(np.mean([r["avg_delay_ms"] for r in recent])) if recent else float("nan")
                    last100_risk = float(np.mean([r["violation_rate"] for r in recent])) if recent else float("nan")
                    loss_text = ", ".join(f"{k}={v:.4f}" for k, v in losses.items() if isinstance(v, (int, float)))
                    print(
                        f"[v2x-value] method={method_name} seed={seed} ep={ep + 1}/{args.episodes} "
                        f"last100_reward={last100_reward:.5f} last100_delay={last100_delay:.3f}ms "
                        f"last100_risk={last100_risk:.4f} eps={epsilon_for(ep, cfg):.3f} "
                        f"replay={len(agent.replay)} updates={agent.updates} {loss_text}",
                        flush=True,
                    )
    return rows, steps


def run_vmas_value_baselines(args, cfg):
    import vmas

    methods = args.methods or ["iql", "vdn", "qmix"]
    action_table = vmas_macro_action_table(tuple(args.vmas_macro_scales))
    n_actions = int(len(action_table))
    rows, steps = [], []
    for seed in args.seeds:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        for method in methods:
            env = vmas.make_env(scenario=args.vmas_scenario, num_envs=1, device=args.device, n_agents=args.vmas_agents, continuous_actions=True, max_steps=args.episode_len)
            obs = env.reset()
            obs_dim = int(obs[0].shape[-1])
            action_dim = int(env.get_agent_action_size(env.agents[0]))
            state_dim = obs_dim * args.vmas_agents
            agent = ValueBaseline(obs_dim, state_dim, args.vmas_agents, n_actions, method, cfg, args.device)
            for ep in range(args.episodes):
                obs = env.reset()
                ep_reward = ep_risk = 0.0
                losses = {}
                for step in range(args.episode_len):
                    obs_np = np.stack([o.detach().cpu().numpy()[0] for o in obs], axis=0).astype(np.float32)
                    state_np = obs_np.reshape(-1)
                    eps = epsilon_for(ep, cfg)
                    macro = agent.act(obs_np, eps)
                    actions_np = vmas_macro_actions(obs_np, macro, action_dim, action_table)
                    actions = [torch.as_tensor(actions_np[i : i + 1], dtype=torch.float32, device=args.device) for i in range(args.vmas_agents)]
                    next_obs, rewards, dones, _infos = env.step(actions)
                    reward = float(torch.stack(rewards).mean().detach().cpu().item())
                    done = bool(dones.detach().cpu().item()) if hasattr(dones, "detach") else bool(dones)
                    next_obs_np = np.stack([o.detach().cpu().numpy()[0] for o in next_obs], axis=0).astype(np.float32)
                    next_state_np = next_obs_np.reshape(-1)
                    risk = float(reward < 0.0)
                    train_reward = reward * cfg.vmas_reward_scale
                    if method == "iql":
                        for i in range(args.vmas_agents):
                            agent.replay.add(obs_np[i], state_np, macro[i], train_reward, next_obs_np[i], next_state_np, float(done))
                    else:
                        agent.replay.add(obs_np, state_np, macro, train_reward, next_obs_np, next_state_np, float(done))
                    losses = agent.update()
                    ep_reward += reward
                    ep_risk += risk
                    steps.append({"domain": "vmas", "method": f"VMAS-{method.upper()}", "seed": seed, "episode": ep, "step": step, "reward": reward, "violation_rate": risk})
                    obs = next_obs
                    if done:
                        break
                rows.append({"domain": "vmas", "scenario": args.vmas_scenario, "method": f"VMAS-{method.upper()}", "seed": seed, "episode": ep, "episode_reward": ep_reward / args.episode_len, "violation_rate": ep_risk / args.episode_len, "n_actions": n_actions, **losses})
                if args.progress_every > 0 and (((ep + 1) % args.progress_every == 0) or (ep + 1 == args.episodes)):
                    method_name = f"VMAS-{method.upper()}"
                    recent = [r for r in rows if r["method"] == method_name and r["seed"] == seed][-100:]
                    last100_reward = float(np.mean([r["episode_reward"] for r in recent])) if recent else float("nan")
                    last100_risk = float(np.mean([r["violation_rate"] for r in recent])) if recent else float("nan")
                    loss_text = ", ".join(f"{k}={v:.4f}" for k, v in losses.items() if isinstance(v, (int, float)))
                    print(
                        f"[vmas-value] method={method_name} seed={seed} ep={ep + 1}/{args.episodes} "
                        f"last100_reward={last100_reward:.5f} last100_risk={last100_risk:.4f} "
                        f"eps={epsilon_for(ep, cfg):.3f} replay={len(agent.replay)} updates={agent.updates} "
                        f"actions={n_actions} {loss_text}",
                        flush=True,
                    )
    return rows, steps


def run_ippo_baselines(args):
    from msage_mappo.training.full import V2XMAPPO

    rows, steps = [], []
    cfg = TrainConfig(
        memory_dim=64,
        top_k=3,
        ppo_epochs=args.ppo_epochs,
        lr=args.lr,
        hidden=args.hidden,
        entropy_coef=args.entropy_coef,
        risk_coef=0.0,
        v2x_reward_scale=args.v2x_reward_scale,
        vmas_reward_scale=args.vmas_ppo_reward_scale,
        vmas_log_std_init=args.vmas_log_std_init,
    )
    zero_mem = np.zeros(64, dtype=np.float32)
    zero_ctx = np.zeros(4, dtype=np.float32)
    if "vmas" in args.domains:
        import vmas
        for seed in args.seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = vmas.make_env(scenario=args.vmas_scenario, num_envs=1, device=args.device, n_agents=args.vmas_agents, continuous_actions=True, max_steps=args.episode_len)
            obs = env.reset()
            obs_dim = int(obs[0].shape[-1])
            action_dim = int(env.get_agent_action_size(env.agents[0]))
            agent = VMASMAPPO(obs_dim, action_dim, obs_dim, cfg, args.device, use_memory=False, use_risk=False)
            for ep in range(args.episodes):
                obs = env.reset()
                rb = {"obs": [], "mem": [], "ctx": [], "action": [], "prior_action": [], "logp": [], "state": [], "gmem": [], "reward": [], "done": [], "value": [], "risk_target": [], "n_agents": args.vmas_agents}
                ep_reward = ep_risk = 0.0
                for step in range(args.episode_len):
                    obs_np = np.stack([o.detach().cpu().numpy()[0] for o in obs], axis=0).astype(np.float32)
                    mem_agents = np.repeat(zero_mem[None, :], args.vmas_agents, axis=0)
                    ctx_agents = np.repeat(zero_ctx[None, :], args.vmas_agents, axis=0)
                    actions_np, logp = agent.select(obs_np, mem_agents, ctx_agents)
                    values = [agent.value(obs_np[i], zero_mem)[0] for i in range(args.vmas_agents)]
                    actions = [torch.as_tensor(actions_np[i : i + 1], dtype=torch.float32, device=args.device) for i in range(args.vmas_agents)]
                    next_obs, rewards, dones, _infos = env.step(actions)
                    reward = float(torch.stack(rewards).mean().detach().cpu().item())
                    done = bool(dones.detach().cpu().item()) if hasattr(dones, "detach") else bool(dones)
                    risk = float(reward < 0.0)
                    for i in range(args.vmas_agents):
                        rb["obs"].append(obs_np[i])
                        rb["mem"].append(zero_mem)
                        rb["ctx"].append(zero_ctx)
                        rb["action"].append(actions_np[i])
                        rb["prior_action"].append(np.zeros(action_dim, dtype=np.float32))
                        rb["logp"].append(logp[i])
                        rb["state"].append(obs_np[i])
                        rb["gmem"].append(zero_mem)
                        rb["reward"].append(reward)
                        rb["done"].append(float(done))
                        rb["value"].append(values[i])
                        rb["risk_target"].append(risk)
                    ep_reward += reward
                    ep_risk += risk
                    steps.append({"domain": "vmas", "method": "VMAS-IPPO", "seed": seed, "episode": ep, "step": step, "reward": reward, "violation_rate": risk})
                    obs = next_obs
                    if done:
                        break
                losses = agent.update(rb, apply_risk=False)
                rows.append({"domain": "vmas", "scenario": args.vmas_scenario, "method": "VMAS-IPPO", "seed": seed, "episode": ep, "episode_reward": ep_reward / args.episode_len, "violation_rate": ep_risk / args.episode_len, **losses})
                if args.progress_every > 0 and (((ep + 1) % args.progress_every == 0) or (ep + 1 == args.episodes)):
                    recent = [r for r in rows if r["method"] == "VMAS-IPPO" and r["seed"] == seed][-100:]
                    last100_reward = float(np.mean([r["episode_reward"] for r in recent])) if recent else float("nan")
                    last100_risk = float(np.mean([r["violation_rate"] for r in recent])) if recent else float("nan")
                    loss_text = ", ".join(f"{k}={v:.4f}" for k, v in losses.items() if isinstance(v, (int, float)))
                    print(
                        f"[vmas-ippo] seed={seed} ep={ep + 1}/{args.episodes} "
                        f"last100_reward={last100_reward:.5f} last100_risk={last100_risk:.4f} {loss_text}",
                        flush=True,
                    )
    if "v2x" in args.domains:
        Env = load_v2x_env_class(args.v2x_env)
        for seed in args.seeds:
            env = Env(seed=seed, use_bad_initial_allocation=False)
            agent = V2XMAPPO(env, cfg, args.device, use_memory=False, use_risk=False)
            method_name = "V2X-IPPO"
            aligned_neural_start = bool(getattr(args, "v2x_align_neural_start", True))
            episode_offset = 1 if aligned_neural_start else 0
            if aligned_neural_start:
                start_env = Env(seed=seed, use_bad_initial_allocation=False)
                start_row, start_steps = evaluate_v2x_baseline_start(start_env, args, method_name, seed)
                rows.append(start_row)
                steps.extend(start_steps)
            for ep in range(args.episodes):
                obs, state = env.reset()
                rb = {"obs": [], "mem": [], "ctx": [], "cont": [], "disc": [], "logp": [], "state": [], "gmem": [], "reward": [], "done": [], "value": [], "risk_target": [], "n_agents": env.n_vehicles}
                ep_reward = ep_delay = ep_system = ep_risk = 0.0
                ep_env_reward = 0.0
                ep_reward_parts: dict[str, list[float]] = {}
                step_count = 0
                prev_info = {
                    "avg_vehicle_max_delay_ms": 0.0,
                    "avg_peak_power_usage": 0.0,
                    "avg_peak_packet_usage": 0.0,
                }
                for step in range(args.episode_len):
                    mem_agents = np.repeat(zero_mem[None, :], env.n_vehicles, axis=0)
                    ctx_agents = np.repeat(zero_ctx[None, :], env.n_vehicles, axis=0)
                    cont, disc, logp, _ = agent.select(np.asarray(obs, dtype=np.float32), mem_agents, ctx_agents)
                    value, _ = agent.value(state, zero_mem)
                    next_obs, next_state, env_reward, done, info = env.step(cont, disc)
                    terminal = bool(done) or (step + 1 >= args.episode_len)
                    reward, reward_parts = v2x_reward_from_info(env_reward, info, prev_info, args)
                    reported_delay = info["avg_vehicle_max_delay_ms"] * float(getattr(args, "v2x_delay_report_scale", 0.5))
                    reported_system_delay = info["system_delay_ms"] * float(getattr(args, "v2x_delay_report_scale", 0.5))
                    risk = v2x_risk(reported_delay, args.v2x_deadline_ms)
                    for i in range(env.n_vehicles):
                        rb["obs"].append(obs[i])
                        rb["mem"].append(zero_mem)
                        rb["ctx"].append(zero_ctx)
                        rb["cont"].append(cont[i])
                        rb["disc"].append(disc[i])
                        rb["logp"].append(logp[i])
                        rb["state"].append(state)
                        rb["gmem"].append(zero_mem)
                        rb["reward"].append(reward)
                        rb["done"].append(float(terminal))
                        rb["value"].append(value)
                        rb["risk_target"].append(risk)
                    ep_reward += reward
                    ep_env_reward += env_reward
                    ep_delay += reported_delay
                    ep_system += reported_system_delay
                    ep_risk += risk
                    step_count += 1
                    for key, value_part in reward_parts.items():
                        ep_reward_parts.setdefault(key, []).append(value_part)
                    steps.append({"domain": "v2x", "method": method_name, "seed": seed, "episode": ep + episode_offset, "step": step, "reward": reward, "env_reward": env_reward, **reward_parts, "avg_vehicle_max_delay_ms": info["avg_vehicle_max_delay_ms"], "reported_avg_delay_ms": reported_delay, "violation_rate": risk, "phase": "train", "aligned_start": 0})
                    obs, state = next_obs, next_state
                    prev_info = info
                    if done:
                        break
                losses = agent.update(rb, apply_risk=False)
                mean_reward_parts = {key: float(np.mean(values)) for key, values in ep_reward_parts.items()}
                denom = max(step_count, 1)
                rows.append({"domain": "v2x", "scenario": "VehicleToBSEnv", "method": method_name, "seed": seed, "episode": ep + episode_offset, "episode_reward": ep_reward / denom, "env_reward": ep_env_reward / denom, "avg_delay_ms": ep_delay / denom, "system_delay_ms": ep_system / denom, "violation_rate": ep_risk / denom, "phase": "train", "aligned_start": 0, **mean_reward_parts, **losses})
                if args.progress_every > 0 and (((ep + 1) % args.progress_every == 0) or (ep + 1 == args.episodes)):
                    recent = [r for r in rows if r["method"] == method_name and r["seed"] == seed][-100:]
                    last100_reward = float(np.mean([r["episode_reward"] for r in recent])) if recent else float("nan")
                    last100_delay = float(np.mean([r["avg_delay_ms"] for r in recent])) if recent else float("nan")
                    last100_risk = float(np.mean([r["violation_rate"] for r in recent])) if recent else float("nan")
                    loss_text = ", ".join(f"{k}={v:.4f}" for k, v in losses.items() if isinstance(v, (int, float)))
                    print(
                        f"[v2x-ippo] method=V2X-IPPO seed={seed} ep={ep + 1}/{args.episodes} "
                        f"last100_reward={last100_reward:.5f} last100_delay={last100_delay:.3f}ms "
                        f"last100_risk={last100_risk:.4f} {loss_text}",
                        flush=True,
                    )
    return rows, steps


def export(path: Path, rows, steps, args):
    ep = pd.DataFrame(rows)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        ep.to_excel(writer, sheet_name="EpisodeMetrics", index=False)
        pd.DataFrame(steps).to_excel(writer, sheet_name="StepMetrics", index=False)
        config.to_excel(writer, sheet_name="Config", index=False)
    print(f"saved {path}")
    print(summary.round(5).to_string(index=False))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--domains", nargs="+", default=["v2x", "vmas"])
    p.add_argument("--methods", nargs="*", default=None, help="iql vdn qmix ippo")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--episode-len", type=int, default=10)
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--v2x-env", default=str(ROOT / "src" / "msage_mappo" / "envs" / "v2x.py"))
    p.add_argument("--v2x-deadline-ms", type=float, default=50.0)
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
    p.add_argument("--v2x-llm-reward-coef", type=float, default=0.0)
    p.add_argument("--v2x-deadline-penalty", type=float, default=2.0)
    p.add_argument("--v2x-margin-penalty", type=float, default=3.0)
    p.add_argument("--v2x-power-penalty", type=float, default=2.5)
    p.add_argument("--v2x-packet-penalty", type=float, default=2.0)
    p.add_argument("--v2x-packet-soft-limit", type=float, default=4.0)
    p.add_argument("--v2x-trend-penalty", type=float, default=0.45)
    p.add_argument("--v2x-trend-bonus", type=float, default=0.20)
    p.add_argument("--vmas-scenario", default="navigation")
    p.add_argument("--vmas-agents", type=int, default=3)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--entropy-coef", type=float, default=0.004)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--replay-size", type=int, default=80000)
    p.add_argument("--target-update", type=int, default=25)
    p.add_argument("--train-after", type=int, default=64)
    p.add_argument("--train-steps", type=int, default=6)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-episodes", type=int, default=500)
    p.add_argument("--v2x-macro-actions", type=int, default=18)
    p.add_argument("--v2x-align-neural-start", dest="v2x_align_neural_start", action="store_true", default=True)
    p.add_argument("--no-v2x-align-neural-start", dest="v2x_align_neural_start", action="store_false")
    p.add_argument("--disable-double-q", action="store_true")
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--td-target-clip", type=float, default=20.0)
    p.add_argument("--vmas-reward-scale", type=float, default=3.0)
    p.add_argument("--vmas-ppo-reward-scale", type=float, default=5.0)
    p.add_argument("--vmas-log-std-init", type=float, default=-0.7)
    p.add_argument("--vmas-macro-scales", nargs="+", type=float, default=[0.30, 0.60, 0.90])
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--output", default=str(ROOT / "outputs" / "additional_baselines.xlsx"))
    return parse_config_args(p, argv)


def main():
    args = parse_args()
    if Path(args.output).exists():
        raise SystemExit("Output already exists; choose a new --output path.")
    cfg = BaselineConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        hidden=args.hidden,
        target_update=args.target_update,
        train_after=args.train_after,
        train_steps=args.train_steps,
        eps_end=args.eps_end,
        eps_decay_episodes=args.eps_decay_episodes,
        double_q=not args.disable_double_q,
        huber_delta=args.huber_delta,
        grad_clip=args.grad_clip,
        td_target_clip=args.td_target_clip,
        vmas_reward_scale=args.vmas_reward_scale,
    )
    methods = set(args.methods or ["iql", "vdn", "qmix", "ippo"])
    rows, steps = [], []
    value_methods = sorted(methods.intersection({"iql", "vdn", "qmix"}))
    if value_methods:
        original = args.methods
        args.methods = value_methods
        if "v2x" in args.domains:
            er, sr = run_v2x_value_baselines(args, cfg)
            rows.extend(er)
            steps.extend(sr)
        if "vmas" in args.domains:
            er, sr = run_vmas_value_baselines(args, cfg)
            rows.extend(er)
            steps.extend(sr)
        args.methods = original
    if "ippo" in methods:
        er, sr = run_ippo_baselines(args)
        rows.extend(er)
        steps.extend(sr)
    export(Path(args.output), rows, steps, args)


if __name__ == "__main__":
    main()
