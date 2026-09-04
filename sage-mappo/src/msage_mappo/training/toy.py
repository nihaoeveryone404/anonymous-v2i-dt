from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from msage_mappo.utils.config import REPO_ROOT as ROOT, parse_config_args

from msage_mappo.agent import MSAGEMAPPO, PPOConfig, compute_returns_advantages
from msage_mappo.envs import ToyV2XConfig, ToyV2XEnv
from msage_mappo.llm_writer import QwenMemoryWriter
from msage_mappo.memory import SemanticMemoryBank


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--episode-len", type=int, default=32)
    p.add_argument("--num-agents", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--memory-dim", type=int, default=64)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--llm-backend", choices=["qwen", "template"], default="qwen")
    p.add_argument("--allow-template-fallback", action="store_true")
    p.add_argument("--model-path", type=str, default=str(ROOT / "data" / "models" / "Qwen_Qwen3.5-9B-Q4_K_M.gguf"))
    p.add_argument("--llm-every", type=int, default=5)
    p.add_argument("--llm-gpu-layers", type=int, default=20)
    p.add_argument("--output-dir", type=str, default=str(ROOT / "outputs"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parse_config_args(p, argv)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"msage_toy_v2x_seed{args.seed}_ep{args.episodes}"
    excel_path = out_dir / f"{run_name}.xlsx"
    memory_path = out_dir / f"{run_name}_memory.jsonl"
    if excel_path.exists() or memory_path.exists():
        raise SystemExit("Output already exists; choose a new --output-dir.")

    env = ToyV2XEnv(ToyV2XConfig(num_agents=args.num_agents, episode_len=args.episode_len, seed=args.seed))
    memory_bank = SemanticMemoryBank(dim=args.memory_dim, top_k=args.top_k)
    writer = QwenMemoryWriter(
        model_path=args.model_path,
        backend=args.llm_backend,
        n_gpu_layers=args.llm_gpu_layers,
        allow_template_fallback=args.allow_template_fallback,
    )
    args.actual_writer_backend = writer.backend
    agent = MSAGEMAPPO(
        PPOConfig(
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            memory_dim=args.memory_dim,
            device=args.device,
        )
    )

    episode_rows = []
    step_rows = []
    update_rows = []
    memory_rows = []

    for ep in trange(args.episodes, desc="training"):
        obs, state = env.reset()
        done = False
        rollout = {
            "obs": [],
            "mem": [],
            "ctx": [],
            "path": [],
            "power": [],
            "log_prob": [],
            "state": [],
            "global_mem": [],
            "reward": [],
            "done": [],
            "value": [],
            "risk_targets": [],
        }
        ep_infos = []
        ep_reward = 0.0
        step = 0
        while not done:
            context = context_from_obs(obs)
            tags = tags_from_context(context)
            query_text = query_text_from_context(context, tags)
            mem_topk, retrieved = memory_bank.retrieve(query_text, {"tags": tags}, top_k=args.top_k)
            mem_vec = mean_memory(mem_topk, args.memory_dim)
            mem_agents = np.repeat(mem_vec[None, :], env.num_agents, axis=0)
            ctx_agents = np.repeat(context[None, :], env.num_agents, axis=0)
            global_mem = mem_vec
            value, _risk_pred = agent.value(state, global_mem)
            action = agent.act(obs, mem_agents, ctx_agents)

            next_obs, next_state, reward, done, info = env.step(action["path"], action["power"])
            ep_reward += reward
            ep_infos.append(info)

            for i in range(env.num_agents):
                rollout["obs"].append(obs[i])
                rollout["mem"].append(mem_agents[i])
                rollout["ctx"].append(ctx_agents[i])
                rollout["path"].append(action["path"][i])
                rollout["power"].append(action["power"][i])
                rollout["log_prob"].append(action["log_prob"][i])
                rollout["state"].append(state)
                rollout["global_mem"].append(global_mem)
                rollout["reward"].append(reward)
                rollout["done"].append(float(done))
                rollout["value"].append(value)
                rollout["risk_targets"].append(info["risk_target"])

            step_rows.append(
                {
                    "episode": ep,
                    "step": step,
                    "reward": reward,
                    "avg_latency": info["avg_latency"],
                    "avg_power": info["avg_power"],
                    "violation_rate": info["violation_rate"],
                    "tail_latency": info["tail_latency"],
                    "queue_pressure": info["queue_pressure"],
                    "deadline_margin": info["deadline_margin"],
                    "memory_bank_size": len(memory_bank),
                    "retrieved_count": len(retrieved),
                    "retrieved_scenarios": "; ".join(item.scenario for item in retrieved),
                }
            )

            obs, state = next_obs, next_state
            step += 1

        returns, advantages = compute_returns_advantages(
            rollout["reward"],
            rollout["value"],
            rollout["done"],
            gamma=agent.cfg.gamma,
            gae_lambda=agent.cfg.gae_lambda,
        )
        rollout["returns"] = returns
        rollout["advantages"] = advantages
        update_stats = agent.update(rollout)
        update_stats["episode"] = ep
        update_rows.append(update_stats)

        summary = summarize_episode(ep, ep_reward, ep_infos, env)
        if (ep + 1) % args.llm_every == 0 or summary["violation_rate"] > 0.0:
            item = writer.write_memory(summary)
            memory_bank.add(item)
            row = {
                "episode": ep,
                "scenario": item.scenario,
                "cause": item.cause,
                "bad_action": item.bad_action,
                "good_action": item.good_action,
                "constraint_tags": ", ".join(item.constraint_tags),
                "priority": item.priority,
                "outcome": item.outcome,
                "writer_backend": item.writer_backend,
                "raw_llm_output": item.raw_llm_output,
                **item.stats,
            }
            memory_rows.append(row)

        episode_rows.append(
            {
                "episode": ep,
                "episode_reward": ep_reward,
                "avg_latency": summary["avg_latency"],
                "avg_power": summary["avg_power"],
                "violation_rate": summary["violation_rate"],
                "tail_latency": summary["tail_latency"],
                "queue_pressure": summary["queue_pressure"],
                "deadline_margin": summary["deadline_margin"],
                "memory_bank_size": len(memory_bank),
                "llm_backend": writer.backend,
            }
        )

    memory_bank.save_jsonl(memory_path)
    export_excel(excel_path, episode_rows, step_rows, update_rows, memory_rows, args)
    print(f"Saved Excel results to: {excel_path}")
    print(f"Saved memory bank to: {memory_path}")


def context_from_obs(obs: np.ndarray) -> np.ndarray:
    avg_sinr = float(np.mean(obs[:, :2]))
    queue_pressure = float(np.mean(obs[:, 2]))
    deadline_margin = float(np.mean(obs[:, 3]))
    avg_power = float(np.mean(obs[:, 4]))
    return np.array([deadline_margin, queue_pressure, avg_power, avg_sinr], dtype=np.float32)


def tags_from_context(context: np.ndarray) -> list[str]:
    deadline_margin, queue_pressure, avg_power, _avg_sinr = context
    tags = []
    if deadline_margin < 0.25:
        tags.append("deadline_violation")
    if queue_pressure > 0.45:
        tags.append("queue_congestion")
    if avg_power > 0.7:
        tags.append("power_pressure")
    if not tags:
        tags.append("stable_operation")
    return tags


def query_text_from_context(context: np.ndarray, tags: list[str]) -> str:
    deadline_margin, queue_pressure, avg_power, avg_sinr = context
    return (
        f"deadline_margin {deadline_margin:.3f} queue_pressure {queue_pressure:.3f} "
        f"avg_power {avg_power:.3f} avg_sinr {avg_sinr:.3f} tags {' '.join(tags)}"
    )


def mean_memory(mem_topk: np.ndarray, dim: int) -> np.ndarray:
    if mem_topk.size == 0:
        return np.zeros(dim, dtype=np.float32)
    return mem_topk.mean(axis=0).astype(np.float32)


def summarize_episode(ep: int, ep_reward: float, infos: list[dict], env: ToyV2XEnv) -> dict:
    avg = lambda key: float(np.mean([info[key] for info in infos]))
    summary = {
        "episode": ep,
        "episode_reward": float(ep_reward),
        "avg_latency": avg("avg_latency"),
        "avg_power": avg("avg_power"),
        "violation_rate": avg("violation_rate"),
        "tail_latency": avg("tail_latency"),
        "queue_pressure": avg("queue_pressure"),
        "deadline_margin": avg("deadline_margin"),
    }
    summary["constraint_tags"] = ToyV2XEnv.constraint_tags_from_info(summary)
    summary["outcome"] = "failure" if summary["violation_rate"] > 0 else "success"
    return summary


def export_excel(excel_path: Path, episode_rows, step_rows, update_rows, memory_rows, args):
    summary_rows = []
    ep_df = pd.DataFrame(episode_rows)
    if not ep_df.empty:
        last = ep_df.tail(max(1, min(5, len(ep_df))))
        summary_rows.append(
            {
                "metric": "final_5_episode_reward",
                "value": float(last["episode_reward"].mean()),
                "note": "Mean reward over final episodes",
            }
        )
        for metric in ["avg_latency", "avg_power", "violation_rate", "tail_latency"]:
            summary_rows.append(
                {
                    "metric": f"final_5_{metric}",
                    "value": float(last[metric].mean()),
                    "note": "Mean over final episodes",
                }
            )
    config_rows = [{"parameter": k, "value": str(v)} for k, v in vars(args).items()]
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(episode_rows).to_excel(writer, sheet_name="EpisodeMetrics", index=False)
        pd.DataFrame(step_rows).to_excel(writer, sheet_name="StepMetrics", index=False)
        pd.DataFrame(update_rows).to_excel(writer, sheet_name="PPOUpdates", index=False)
        pd.DataFrame(memory_rows).to_excel(writer, sheet_name="SemanticMemory", index=False)
        pd.DataFrame(config_rows).to_excel(writer, sheet_name="Config", index=False)


if __name__ == "__main__":
    main()
