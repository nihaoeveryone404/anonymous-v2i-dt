from __future__ import annotations

import json
import os
import re
from pathlib import Path
import site
from typing import Any

from .memory import SemanticMemoryItem


class QwenMemoryWriter:
    """Trajectory-to-semantic-memory writer backed by a local GGUF Qwen model."""

    def __init__(
        self,
        model_path: str | Path,
        backend: str = "qwen",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        temperature: float = 0.2,
        allow_template_fallback: bool = False,
    ):
        self.model_path = Path(model_path)
        self.backend = backend
        self.temperature = temperature
        self.allow_template_fallback = allow_template_fallback
        self.llm = None
        if backend == "qwen":
            if not self.model_path.is_file():
                if not allow_template_fallback:
                    raise FileNotFoundError(f"Qwen GGUF file not found: {self.model_path}. Provide --model-path or explicitly use --llm-backend template for smoke tests.")
            try:
                add_llama_dll_dirs()
                from llama_cpp import Llama

                self.llm = Llama(
                    model_path=str(self.model_path),
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )
            except Exception:
                if not allow_template_fallback:
                    raise
                self.backend = "template"
        elif backend != "template":
            raise ValueError(f"Unsupported LLM backend: {backend}")

    def write_memory(self, trajectory_summary: dict[str, Any]) -> SemanticMemoryItem:
        if self.backend == "qwen" and self.llm is not None:
            text = self._generate_json_text(trajectory_summary)
            try:
                payload = extract_json(text)
                item = item_from_payload(payload, trajectory_summary)
                item.writer_backend = "qwen"
                item.raw_llm_output = text[:1000]
                return item
            except Exception:
                if not self.allow_template_fallback:
                    raise
        item = template_memory(trajectory_summary)
        item.writer_backend = "template_fallback" if self.backend == "qwen" else "template"
        return item

    def _generate_json_text(self, trajectory_summary: dict[str, Any]) -> str:
        system = (
            "You are a semantic memory writer for constrained multi-agent reinforcement learning. "
            "Return only one valid JSON object. No markdown. No explanation."
        )
        user = build_prompt(trajectory_summary)
        if hasattr(self.llm, "create_chat_completion"):
            raw = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=384,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            return raw["choices"][0]["message"]["content"]
        raw = self.llm(
            f"{system}\n\n{user}",
            max_tokens=384,
            temperature=self.temperature,
        )
        return raw["choices"][0]["text"]


def build_prompt(summary: dict[str, Any]) -> str:
    return f"""Convert the trajectory summary into one compact semantic memory item.
Return exactly one JSON object with these keys:
"scenario", "cause", "bad_action", "good_action", "constraint_tags", "priority", "outcome".

Rules:
- "constraint_tags" must be a JSON array of strings.
- "priority" must be one of "latency", "power", or "balanced".
- "outcome" must be one of "success", "failure", or "partial_success".
- Do not include markdown fences.

Trajectory summary:
{json.dumps(summary, ensure_ascii=False)}
"""


def add_llama_dll_dirs() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    candidates: list[Path] = []
    for sp in site.getsitepackages():
        base = Path(sp)
        candidates.extend(
            [
                base / "llama_cpp" / "lib",
                base / "torch" / "lib",
            ]
        )
    prefix = Path(os.environ.get("CONDA_PREFIX", ""))
    if prefix:
        candidates.extend([prefix / "Library" / "bin", prefix / "DLLs"])
    for path in candidates:
        if path.exists():
            try:
                os.add_dll_directory(str(path))
            except OSError:
                pass


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in LLM output: {text}")
    return json.loads(match.group(0))


def item_from_payload(payload: dict[str, Any], summary: dict[str, Any]) -> SemanticMemoryItem:
    tags = payload.get("constraint_tags") or summary.get("constraint_tags") or ["stable_operation"]
    if isinstance(tags, str):
        tags = [tags]
    stats = {
        "episode_reward": float(summary.get("episode_reward", 0.0)),
        "avg_latency": float(summary.get("avg_latency", 0.0)),
        "avg_power": float(summary.get("avg_power", 0.0)),
        "violation_rate": float(summary.get("violation_rate", 0.0)),
    }
    return SemanticMemoryItem(
        scenario=str(payload.get("scenario", "unknown_scenario")),
        cause=str(payload.get("cause", "unknown_cause")),
        bad_action=str(payload.get("bad_action", "unknown_bad_action")),
        good_action=str(payload.get("good_action", "unknown_good_action")),
        constraint_tags=list(tags),
        priority=str(payload.get("priority", "latency")),
        outcome=str(payload.get("outcome", summary.get("outcome", "unknown"))),
        stats=stats,
        writer_backend="qwen",
    )


def template_memory(summary: dict[str, Any]) -> SemanticMemoryItem:
    tags = summary.get("constraint_tags", ["stable_operation"])
    violation = summary.get("violation_rate", 0.0)
    queue = summary.get("queue_pressure", 0.0)
    power = summary.get("avg_power", 0.0)
    if violation > 0:
        scenario = "deadline_risk_under_congestion"
        cause = "queues and weak links increase latency beyond the service deadline"
        bad_action = "continue using overloaded or degraded paths"
        good_action = "redistribute traffic to stronger paths before the deadline margin collapses"
        priority = "latency"
        outcome = "failure"
    elif power > 0.65:
        scenario = "power_pressure_with_stable_latency"
        cause = "latency is acceptable but transmit power is high"
        bad_action = "increase power on already stable links"
        good_action = "reduce power while keeping traffic on reliable paths"
        priority = "power"
        outcome = "partial_success"
    elif queue > 0.45:
        scenario = "queue_congestion_without_violation"
        cause = "queue pressure is rising but deadlines are still feasible"
        bad_action = "wait until deadline violation appears"
        good_action = "shift some traffic early to avoid future congestion"
        priority = "balanced"
        outcome = "success"
    else:
        scenario = "stable_low_risk_operation"
        cause = "links and queues remain within safe operating range"
        bad_action = "overreact with unnecessary power or path switching"
        good_action = "maintain balanced traffic split and conservative power"
        priority = "balanced"
        outcome = "success"
    return SemanticMemoryItem(
        scenario=scenario,
        cause=cause,
        bad_action=bad_action,
        good_action=good_action,
        constraint_tags=list(tags),
        priority=priority,
        outcome=outcome,
        stats={
            "episode_reward": float(summary.get("episode_reward", 0.0)),
            "avg_latency": float(summary.get("avg_latency", 0.0)),
            "avg_power": float(summary.get("avg_power", 0.0)),
            "violation_rate": float(summary.get("violation_rate", 0.0)),
        },
        writer_backend="template",
    )
