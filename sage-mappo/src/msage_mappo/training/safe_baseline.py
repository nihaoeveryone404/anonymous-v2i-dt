from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical, Dirichlet
import torch.nn.functional as F

from msage_mappo.utils.config import REPO_ROOT as ROOT, parse_config_args

from msage_mappo.training.full import (
    TrainConfig,
    V2XActor,
    V2XLagrangianCritic,
    apply_v2x_stress,
    load_v2x_env_class,
    v2x_context,
    v2x_reward_from_info,
    v2x_risk,
)


def gae(signal: np.ndarray, dones: np.ndarray, values: np.ndarray, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    values_ext = np.concatenate([values, np.asarray([0.0], dtype=np.float32)])
    advantages = np.zeros_like(signal, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(signal))):
        mask = 1.0 - dones[t]
        delta = signal[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
        running = delta + gamma * gae_lambda * mask * running
        advantages[t] = running
    return advantages + values, advantages


def normalize_advantage(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.clip((values - values.mean()) / (values.std() + 1e-8), -5.0, 5.0)


class V2XMAPPOLagrangian:
    def __init__(self, env, cfg: TrainConfig, device: str):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = V2XActor(
            env.obs_dim,
            cfg.memory_dim,
            cfg.ctx_dim,
            cfg.hidden,
            env.n_selected_bs,
            env.n_packet_choices,
        ).to(self.device)
        self.critic = V2XLagrangianCritic(env.state_dim, cfg.memory_dim, cfg.hidden).to(self.device)
        self.optim = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.lr,
        )
        self.lagrange_multiplier = float(cfg.lagrangian_init)

    def select(self, obs_np, mem_np, ctx_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        mem = torch.as_tensor(mem_np, dtype=torch.float32, device=self.device)
        ctx = torch.as_tensor(ctx_np, dtype=torch.float32, device=self.device)
        concentration, packet_logits = self.actor(obs, mem, ctx)
        dirichlet = Dirichlet(concentration)
        categorical = Categorical(logits=packet_logits)
        cont = dirichlet.sample()
        disc = categorical.sample()
        logp = dirichlet.log_prob(cont) + categorical.log_prob(disc).sum(dim=-1)
        return (
            cont.detach().cpu().numpy(),
            disc.detach().cpu().numpy(),
            logp.detach().cpu().numpy(),
        )

    def value(self, state_np, mem_np) -> tuple[float, float]:
        state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        mem = torch.as_tensor(mem_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            reward_value, cost_value = self.critic(state, mem)
        return float(reward_value.item()), float(cost_value.item())

    def update(self, rollout: dict[str, list[Any]], apply_constraint: bool) -> dict[str, float]:
        n_agents = int(rollout["n_agents"])
        rewards = np.asarray(rollout["reward"], dtype=np.float32)[::n_agents] * float(self.cfg.v2x_reward_scale)
        costs = np.asarray(rollout["cost"], dtype=np.float32)[::n_agents]
        dones = np.asarray(rollout["done"], dtype=np.float32)[::n_agents]
        reward_values = np.asarray(rollout["value"], dtype=np.float32)[::n_agents]
        cost_values = np.asarray(rollout["cost_value"], dtype=np.float32)[::n_agents]

        reward_returns, reward_advantages = gae(
            rewards,
            dones,
            reward_values,
            self.cfg.gamma,
            self.cfg.gae_lambda,
        )
        cost_returns, cost_advantages = gae(
            costs,
            dones,
            cost_values,
            self.cfg.gamma,
            self.cfg.gae_lambda,
        )
        reward_advantages = np.repeat(normalize_advantage(reward_advantages), n_agents)
        cost_advantages = np.repeat(normalize_advantage(cost_advantages), n_agents)
        reward_returns = np.repeat(reward_returns, n_agents)
        cost_returns = np.repeat(cost_returns, n_agents)

        obs = torch.as_tensor(np.asarray(rollout["obs"]), dtype=torch.float32, device=self.device)
        mem = torch.as_tensor(np.asarray(rollout["mem"]), dtype=torch.float32, device=self.device)
        ctx = torch.as_tensor(np.asarray(rollout["ctx"]), dtype=torch.float32, device=self.device)
        cont = torch.as_tensor(np.asarray(rollout["cont"]), dtype=torch.float32, device=self.device)
        disc = torch.as_tensor(np.asarray(rollout["disc"]), dtype=torch.int64, device=self.device)
        old_logp = torch.as_tensor(np.asarray(rollout["logp"]), dtype=torch.float32, device=self.device)
        states = torch.as_tensor(np.asarray(rollout["state"]), dtype=torch.float32, device=self.device)
        reward_targets = torch.as_tensor(reward_returns, dtype=torch.float32, device=self.device)
        cost_targets = torch.as_tensor(cost_returns, dtype=torch.float32, device=self.device)
        reward_adv = torch.as_tensor(reward_advantages, dtype=torch.float32, device=self.device)
        cost_adv = torch.as_tensor(cost_advantages, dtype=torch.float32, device=self.device)

        multiplier = self.lagrange_multiplier if apply_constraint else 0.0
        combined_adv = (reward_adv - multiplier * cost_adv) / (1.0 + multiplier)
        last: dict[str, float] = {}
        for _ in range(self.cfg.ppo_epochs):
            concentration, packet_logits = self.actor(obs, mem, ctx)
            dirichlet = Dirichlet(concentration)
            categorical = Categorical(logits=packet_logits)
            normalized_cont = torch.clamp(cont, min=1e-6)
            normalized_cont = normalized_cont / normalized_cont.sum(dim=-1, keepdim=True)
            logp = dirichlet.log_prob(normalized_cont) + categorical.log_prob(disc).sum(dim=-1)
            entropy = (dirichlet.entropy() + categorical.entropy().sum(dim=-1)).mean()
            reward_value, cost_value = self.critic(states, mem)
            ratio = torch.exp(logp - old_logp)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps)
            actor_loss = -torch.min(ratio * combined_adv, clipped_ratio * combined_adv).mean()
            actor_loss = actor_loss - self.cfg.entropy_coef * entropy
            reward_value_loss = F.smooth_l1_loss(reward_value, reward_targets)
            cost_value_loss = F.smooth_l1_loss(cost_value, cost_targets)
            loss = actor_loss + self.cfg.value_coef * (reward_value_loss + cost_value_loss)

            if not torch.isfinite(loss):
                last = {"skipped_update": 1.0}
                continue
            self.optim.zero_grad()
            loss.backward()
            params = list(self.actor.parameters()) + list(self.critic.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(grad_norm):
                self.optim.zero_grad(set_to_none=True)
                last = {"skipped_update": 1.0}
                continue
            self.optim.step()
            last = {
                "actor_loss": float(actor_loss.detach().cpu()),
                "critic_loss": float(reward_value_loss.detach().cpu()),
                "cost_critic_loss": float(cost_value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "skipped_update": 0.0,
            }

        mean_cost = float(costs.mean()) if len(costs) else 0.0
        if apply_constraint:
            self.lagrange_multiplier = float(
                np.clip(
                    self.lagrange_multiplier
                    + self.cfg.lagrangian_lr * (mean_cost - self.cfg.lagrangian_cost_limit),
                    0.0,
                    self.cfg.lagrangian_max,
                )
            )
        last.update(
            {
                "constraint_cost": mean_cost,
                "cost_limit": float(self.cfg.lagrangian_cost_limit),
                "lagrangian_lambda": float(self.lagrange_multiplier),
                "constraint_active": float(apply_constraint),
            }
        )
        return last


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_seed(args: argparse.Namespace, seed: int, episode_path: Path, step_path: Path) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Env = load_v2x_env_class(args.v2x_env)
    env = Env(seed=seed, use_bad_initial_allocation=False)
    apply_v2x_stress(env, args)
    cfg = TrainConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        hidden=args.hidden,
        memory_dim=args.memory_dim,
        v2x_deadline_ms=args.v2x_deadline_ms,
        v2x_reward_scale=args.v2x_reward_scale,
        lagrangian_cost_limit=args.lagrangian_cost_limit,
        lagrangian_lr=args.lagrangian_lr,
        lagrangian_init=args.lagrangian_init,
        lagrangian_max=args.lagrangian_max,
    )
    agent = V2XMAPPOLagrangian(env, cfg, args.device)
    memory = np.zeros(args.memory_dim, dtype=np.float32)

    for episode in range(1, args.episodes + 1):
        obs, state = env.reset()
        apply_v2x_stress(env, args)
        obs, state = env.get_obs(), env.get_state()
        rollout: dict[str, Any] = {
            "obs": [],
            "mem": [],
            "ctx": [],
            "cont": [],
            "disc": [],
            "logp": [],
            "state": [],
            "reward": [],
            "cost": [],
            "done": [],
            "value": [],
            "cost_value": [],
            "n_agents": env.n_vehicles,
        }
        previous_info = {
            "avg_vehicle_max_delay_ms": 0.0,
            "avg_peak_power_usage": 0.0,
            "avg_peak_packet_usage": 0.0,
        }
        episode_reward = 0.0
        episode_env_reward = 0.0
        delays: list[float] = []
        system_delays: list[float] = []
        powers: list[float] = []
        packets: list[float] = []
        costs: list[float] = []
        step_rows: list[dict[str, Any]] = []

        for step in range(args.episode_len):
            context = v2x_context(obs, previous_info)
            agent_memory = np.repeat(memory[None, :], env.n_vehicles, axis=0)
            agent_context = np.repeat(context[None, :], env.n_vehicles, axis=0)
            cont, disc, logp = agent.select(np.asarray(obs, dtype=np.float32), agent_memory, agent_context)
            reward_value, cost_value = agent.value(state, memory)
            next_obs, next_state, env_reward, done, info = env.step(cont, disc)
            terminal = bool(done) or step + 1 >= args.episode_len
            reward, reward_parts = v2x_reward_from_info(
                env_reward,
                info,
                previous_info,
                args,
                llm_guidance_bonus=0.0,
            )
            reported_delay = float(info["avg_vehicle_max_delay_ms"]) * args.v2x_delay_report_scale
            reported_system_delay = float(info["system_delay_ms"]) * args.v2x_delay_report_scale
            cost = v2x_risk(reported_delay, args.v2x_deadline_ms)

            for agent_idx in range(env.n_vehicles):
                rollout["obs"].append(np.asarray(obs[agent_idx], dtype=np.float32))
                rollout["mem"].append(memory)
                rollout["ctx"].append(context)
                rollout["cont"].append(cont[agent_idx])
                rollout["disc"].append(disc[agent_idx])
                rollout["logp"].append(logp[agent_idx])
                rollout["state"].append(state)
                rollout["reward"].append(reward)
                rollout["cost"].append(cost)
                rollout["done"].append(float(terminal))
                rollout["value"].append(reward_value)
                rollout["cost_value"].append(cost_value)

            episode_reward += reward
            episode_env_reward += float(env_reward)
            delays.append(reported_delay)
            system_delays.append(reported_system_delay)
            powers.append(float(info["avg_peak_power_usage"]))
            packets.append(float(info["avg_peak_packet_usage"]))
            costs.append(cost)
            step_rows.append(
                {
                    "domain": "v2x",
                    "scenario": "VehicleToBSEnv",
                    "method": "V2X-MAPPO-Lagrangian",
                    "seed": seed,
                    "episode": episode,
                    "step": step,
                    "reward": float(reward),
                    "env_reward": float(env_reward),
                    "reported_avg_delay_rdu": reported_delay,
                    "reported_system_delay_rdu": reported_system_delay,
                    "constraint_cost": cost,
                    "step_violation": int(reported_delay > args.v2x_deadline_ms),
                    "avg_peak_power": float(info["avg_peak_power_usage"]),
                    "avg_peak_packets": float(info["avg_peak_packet_usage"]),
                    **{key: float(value) for key, value in reward_parts.items()},
                }
            )
            obs, state = next_obs, next_state
            previous_info = info
            if done:
                break

        apply_constraint = episode > args.lagrangian_warmup_episodes
        losses = agent.update(rollout, apply_constraint=apply_constraint)
        steps = max(len(delays), 1)
        avg_delay = float(np.mean(delays))
        avg_system_delay = float(np.mean(system_delays))
        avg_cost = float(np.mean(costs))
        episode_row = {
            "domain": "v2x",
            "scenario": "VehicleToBSEnv",
            "method": "V2X-MAPPO-Lagrangian",
            "seed": seed,
            "episode": episode,
            "episode_reward": episode_reward / steps,
            "env_reward": episode_env_reward / steps,
            "avg_delay_rdu": avg_delay,
            "system_delay_rdu": avg_system_delay,
            "avg_peak_power": float(np.mean(powers)),
            "avg_peak_packets": float(np.mean(packets)),
            "constraint_cost": avg_cost,
            "episode_violation": int(avg_delay > args.v2x_deadline_ms),
            "deadline_rdu": args.v2x_deadline_ms,
            **losses,
        }
        append_jsonl(episode_path, [episode_row])
        append_jsonl(step_path, step_rows)

        if args.progress_every > 0 and (episode % args.progress_every == 0 or episode == args.episodes):
            recent = [
                row
                for row in load_jsonl(episode_path)
                if int(row["seed"]) == seed
            ][-100:]
            mean_delay = float(np.mean([row["avg_delay_rdu"] for row in recent]))
            event_rate = float(np.mean([row["episode_violation"] for row in recent]))
            mean_reward = float(np.mean([row["episode_reward"] for row in recent]))
            print(
                f"[safe-v2x] seed={seed} episode={episode}/{args.episodes} "
                f"last100_reward={mean_reward:.5f} last100_delay={mean_delay:.3f} "
                f"last100_event_rate={event_rate:.4f} lambda={agent.lagrange_multiplier:.4f}",
                flush=True,
            )


def summarize(path: Path, window: int = 100) -> None:
    rows = load_jsonl(path)
    print("\nMAPPO-Lagrangian final-window summary")
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed"]) == seed][-window:]
        if not seed_rows:
            continue
        print(
            f"seed={seed} reward={np.mean([r['episode_reward'] for r in seed_rows]):.5f} "
            f"delay={np.mean([r['avg_delay_rdu'] for r in seed_rows]):.3f} "
            f"cost={np.mean([r['constraint_cost'] for r in seed_rows]):.5f} "
            f"events={sum(int(r['episode_violation']) for r in seed_rows)}/{len(seed_rows)} "
            f"lambda={seed_rows[-1]['lagrangian_lambda']:.4f}"
        )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Action-space-matched MAPPO-Lagrangian baseline for the V2X experiment.")
    parser.add_argument("--v2x-env", default=str(ROOT / "src" / "msage_mappo" / "envs" / "v2x.py"))
    parser.add_argument("--domains", nargs="+", choices=["v2x"], default=["v2x"])
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "v2x_safe_baseline"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--episode-len", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123, 2026, 3407])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.004)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--lagrangian-cost-limit", type=float, default=0.005)
    parser.add_argument("--lagrangian-lr", type=float, default=0.2)
    parser.add_argument("--lagrangian-init", type=float, default=0.0)
    parser.add_argument("--lagrangian-max", type=float, default=20.0)
    parser.add_argument("--lagrangian-warmup-episodes", type=int, default=50)
    parser.add_argument("--v2x-deadline-ms", type=float, default=50.0)
    parser.add_argument("--v2x-traffic-scale", type=float, default=1.0)
    parser.add_argument("--v2x-shadow-extra-db", type=float, default=0.0)
    parser.add_argument("--v2x-reward-mode", choices=["resource", "env", "composite", "utility"], default="resource")
    parser.add_argument("--v2x-reward-scale", type=float, default=1.0)
    parser.add_argument("--v2x-delay-report-scale", type=float, default=0.5)
    parser.add_argument("--v2x-delay-budget-ms", type=float, default=120.0)
    parser.add_argument("--v2x-reward-shift", type=float, default=0.4)
    parser.add_argument("--v2x-latency-weight", type=float, default=2.4)
    parser.add_argument("--v2x-deadline-weight", type=float, default=1.4)
    parser.add_argument("--v2x-power-weight", type=float, default=0.9)
    parser.add_argument("--v2x-packet-balance-weight", type=float, default=0.8)
    parser.add_argument("--v2x-risk-penalty", type=float, default=1.0)
    parser.add_argument("--v2x-trend-bonus-weight", type=float, default=1.4)
    parser.add_argument("--v2x-trend-penalty-weight", type=float, default=0.6)
    parser.add_argument("--v2x-packet-soft-limit", type=float, default=4.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    return parse_config_args(parser, argv)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "episode_metrics.jsonl"
    step_path = output_dir / "step_metrics.jsonl"
    config_path = output_dir / "config.json"

    if not args.resume and (episode_path.exists() or step_path.exists()):
        raise SystemExit(f"Output exists in {output_dir}; use --resume or choose a new --output-dir.")
    config_path.write_text(json.dumps(vars(args), indent=2, ensure_ascii=True), encoding="utf-8")

    completed = set()
    if args.resume:
        rows = load_jsonl(episode_path)
        for seed in args.seeds:
            count = sum(1 for row in rows if int(row["seed"]) == seed)
            if count == args.episodes:
                completed.add(seed)
            elif count > 0:
                raise SystemExit(
                    f"Seed {seed} has a partial log ({count}/{args.episodes}); "
                    "resume requires a complete seed boundary."
                )
    for seed in args.seeds:
        if seed in completed:
            print(f"[safe-v2x] seed={seed} already complete; skipping", flush=True)
            continue
        run_seed(args, seed, episode_path, step_path)
    summarize(episode_path)


if __name__ == "__main__":
    main()
