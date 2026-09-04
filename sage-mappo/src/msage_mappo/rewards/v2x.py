from __future__ import annotations

import numpy as np


def v2x_llm_guidance_bonus(obs, cont, disc, retrieved_items, tags, args) -> tuple[float, dict[str, float]]:
    """Reward alignment with retrieved semantic-memory guidance.

    The bonus is bounded and auditable: it does not call an LLM online. It uses
    retrieved LLM/template memories as the semantic signal and rewards actions
    that follow their shared recommendation: prefer high-SINR, low-load links
    and avoid concentrating all packets on one route under risk.
    """
    coef = float(getattr(args, "v2x_llm_reward_coef", 0.0))
    if coef <= 0.0 or not retrieved_items:
        return 0.0, {
            "reward_llm_bonus": 0.0,
            "reward_llm_action_alignment": 0.0,
            "reward_llm_tag_overlap": 0.0,
            "reward_llm_memory_count": float(len(retrieved_items) if retrieved_items else 0),
        }

    obs_arr = np.asarray(obs, dtype=np.float32)
    cont_arr = np.asarray(cont, dtype=np.float32)
    disc_arr = np.asarray(disc, dtype=np.float32)
    scores = obs_arr[:, 2::3] / (obs_arr[:, 1::3] + 0.05)
    scores = np.clip(scores, 1e-4, None)
    target_cont = scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-6)

    target_packets = np.zeros_like(disc_arr, dtype=np.float32)
    order = np.argsort(scores, axis=1)
    best = order[:, -1]
    second = order[:, -2] if scores.shape[1] > 1 else best
    for i in range(obs_arr.shape[0]):
        target_packets[i, best[i]] = 0.60
        target_packets[i, second[i]] = 0.40

    cont_norm = cont_arr / np.maximum(cont_arr.sum(axis=1, keepdims=True), 1e-6)
    packet_norm = np.clip(disc_arr, 0.0, None)
    packet_norm = packet_norm / np.maximum(packet_norm.sum(axis=1, keepdims=True), 1e-6)
    cont_alignment = 1.0 - 0.5 * np.abs(cont_norm - target_cont).sum(axis=1)
    packet_alignment = 1.0 - 0.5 * np.abs(packet_norm - target_packets).sum(axis=1)
    action_alignment = float(np.clip(0.60 * np.mean(cont_alignment) + 0.40 * np.mean(packet_alignment), 0.0, 1.0))

    current_tags = set(tags or [])
    overlaps = []
    qwen_hits = 0
    latency_priority = 0
    for item in retrieved_items:
        item_tags = set(getattr(item, "constraint_tags", []) or [])
        overlaps.append(len(current_tags.intersection(item_tags)) / max(1, len(current_tags.union(item_tags))))
        backend = str(getattr(item, "writer_backend", ""))
        qwen_hits += int("qwen" in backend)
        latency_priority += int(str(getattr(item, "priority", "")) == "latency")
    tag_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    qwen_ratio = qwen_hits / max(1, len(retrieved_items))
    latency_ratio = latency_priority / max(1, len(retrieved_items))
    semantic_weight = 1.0 + 0.25 * tag_overlap + 0.10 * qwen_ratio + 0.10 * latency_ratio
    bonus = float(np.clip(coef * semantic_weight * action_alignment, 0.0, 1.5 * coef))
    return bonus, {
        "reward_llm_bonus": bonus,
        "reward_llm_action_alignment": action_alignment,
        "reward_llm_tag_overlap": tag_overlap,
        "reward_llm_memory_count": float(len(retrieved_items)),
    }


def v2x_risk(delay_ms: float, deadline_ms: float) -> float:
    return float(np.clip((delay_ms - deadline_ms) / max(deadline_ms, 1e-6), 0.0, 1.0))


def v2x_reward_from_info(
    env_reward: float,
    info: dict,
    prev_info: dict | None,
    args,
    llm_guidance_bonus: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """V2X reward with the original plotted delay scale.

    The default ``resource`` mode is a positive mixed utility. Delay is not
    negated directly; it is converted into bounded quality and deadline-margin
    utilities, then combined with power efficiency, packet balance, temporal
    improvement, risk, and the LLM-memory guidance bonus.
    """
    mode = getattr(args, "v2x_reward_mode", "resource")
    delay_scale = float(getattr(args, "v2x_delay_report_scale", 0.5))
    raw_delay = float(info.get("avg_vehicle_max_delay_ms", -float(env_reward)))
    delay = raw_delay * delay_scale
    deadline = float(getattr(args, "v2x_deadline_ms", 50.0))
    llm_bonus = float(llm_guidance_bonus)
    if mode == "env":
        return float(env_reward), {
            "reward_env": float(env_reward),
            "reward_raw_delay_ms": raw_delay,
            "reward_delay_cost": delay,
            "reward_latency_utility": 0.0,
            "reward_deadline_utility": 0.0,
            "reward_power_utility": 0.0,
            "reward_packet_balance": 0.0,
            "reward_deadline_cost": 0.0,
            "reward_margin_cost": 0.0,
            "reward_power_cost": 0.0,
            "reward_packet_cost": 0.0,
            "reward_trend_cost": 0.0,
            "reward_improvement_bonus": 0.0,
            "reward_risk_penalty": 0.0,
            "reward_llm_bonus": 0.0,
        }

    peak_power = float(info.get("avg_peak_power_usage", info.get("avg_power", 0.0)))
    power_pressure = float(np.clip(peak_power, 0.0, 1.0))
    peak_packets = float(info.get("avg_peak_packet_usage", info.get("avg_peak_packets", 0.0)))
    packet_soft_limit = float(getattr(args, "v2x_packet_soft_limit", 4.0))
    prev_delay = 0.0 if prev_info is None else float(prev_info.get("avg_vehicle_max_delay_ms", 0.0)) * delay_scale

    if mode == "resource":
        packet_pressure = float(np.clip((peak_packets - packet_soft_limit) / max(10.0 - packet_soft_limit, 1.0), 0.0, 1.0))
        delay_delta = delay - prev_delay if prev_delay > 0.0 else 0.0
        trend_cost = float(np.clip(delay_delta / max(deadline, 1e-6), 0.0, 1.0))
        improvement_bonus = float(np.clip(-delay_delta / max(deadline, 1e-6), 0.0, 1.0))
        delay_budget = max(float(getattr(args, "v2x_delay_budget_ms", 120.0)), deadline + 1e-6)
        latency_utility = float(np.clip((delay_budget - delay) / max(delay_budget - 0.70 * deadline, 1e-6), 0.0, 1.0))
        deadline_margin = float(np.clip((deadline - delay) / max(0.35 * deadline, 1e-6), -1.0, 1.0))
        deadline_utility = float(0.5 + 0.5 * deadline_margin)
        power_utility = float(1.0 - power_pressure)
        packet_balance = float(1.0 - packet_pressure)
        risk = v2x_risk(delay, deadline)
        parts = {
            "reward_env": float(env_reward),
            "reward_raw_delay_ms": raw_delay,
            "reward_delay_cost": delay,
            "reward_latency_utility": latency_utility,
            "reward_deadline_utility": deadline_utility,
            "reward_power_utility": power_utility,
            "reward_packet_balance": packet_balance,
            "reward_deadline_cost": 0.0,
            "reward_margin_cost": 0.0,
            "reward_power_cost": power_pressure,
            "reward_packet_cost": packet_pressure,
            "reward_trend_cost": trend_cost,
            "reward_improvement_bonus": improvement_bonus,
            "reward_risk_penalty": risk,
            "reward_llm_bonus": llm_bonus,
        }
        reward = (
            float(getattr(args, "v2x_reward_shift", 0.4))
            + float(getattr(args, "v2x_latency_weight", 2.4)) * latency_utility
            + float(getattr(args, "v2x_deadline_weight", 1.4)) * deadline_utility
            + float(getattr(args, "v2x_power_weight", 0.9)) * power_utility
            + float(getattr(args, "v2x_packet_balance_weight", 0.8)) * packet_balance
            + float(getattr(args, "v2x_trend_bonus_weight", 1.4)) * improvement_bonus
            - float(getattr(args, "v2x_trend_penalty_weight", 0.6)) * trend_cost
            - float(getattr(args, "v2x_risk_penalty", 1.0)) * risk
            + llm_bonus
        )
        return float(reward), parts
    if mode == "utility":
        delay_budget = float(getattr(args, "v2x_delay_budget_ms", 120.0))
        latency_utility = float(np.clip((delay_budget - delay) / max(delay_budget - deadline, 1e-6), 0.0, 1.0))
        deadline_utility = float(np.exp(-max(delay - deadline, 0.0) / max(deadline, 1e-6)))
        power_utility = float(1.0 - power_pressure)
        packet_balance = float(1.0 - np.clip((peak_packets - packet_soft_limit) / max(10.0 - packet_soft_limit, 1.0), 0.0, 1.0))
        delay_delta = delay - prev_delay if prev_delay > 0.0 else 0.0
        trend_cost = float(np.clip(delay_delta / max(deadline, 1e-6), 0.0, 0.5))
        improvement_bonus = float(np.clip(-delay_delta / max(deadline, 1e-6), 0.0, 0.5))
        risk = v2x_risk(delay, deadline)
        parts = {
            "reward_env": float(env_reward),
            "reward_raw_delay_ms": raw_delay,
            "reward_delay_cost": delay,
            "reward_latency_utility": latency_utility,
            "reward_deadline_utility": deadline_utility,
            "reward_power_utility": power_utility,
            "reward_packet_balance": packet_balance,
            "reward_deadline_cost": 0.0,
            "reward_margin_cost": 0.0,
            "reward_power_cost": power_pressure,
            "reward_packet_cost": max(1.0 - packet_balance, 0.0),
            "reward_trend_cost": trend_cost,
            "reward_improvement_bonus": improvement_bonus,
            "reward_risk_penalty": risk,
            "reward_llm_bonus": llm_bonus,
        }
        reward = (
            float(getattr(args, "v2x_reward_shift", 1.0))
            + float(getattr(args, "v2x_latency_weight", 4.0)) * latency_utility
            + float(getattr(args, "v2x_deadline_weight", 1.2)) * deadline_utility
            + float(getattr(args, "v2x_power_weight", 0.8)) * power_utility
            + float(getattr(args, "v2x_packet_balance_weight", 0.8)) * packet_balance
            + float(getattr(args, "v2x_trend_bonus_weight", 1.4)) * improvement_bonus
            - float(getattr(args, "v2x_trend_penalty_weight", 0.4)) * trend_cost
            - float(getattr(args, "v2x_risk_penalty", 1.0)) * risk
            + llm_bonus
        )
        return float(reward), parts

    deadline_excess_ms = max(delay - deadline, 0.0)
    soft_margin_ms = max(delay - 0.85 * deadline, 0.0)
    margin_pressure = float(np.clip(soft_margin_ms / max(0.15 * deadline, 1e-6), 0.0, 1.0))

    packet_pressure = float(np.clip((peak_packets - packet_soft_limit) / max(packet_soft_limit, 1.0), 0.0, 1.0))

    delay_delta = delay - prev_delay if prev_delay > 0.0 else 0.0
    trend_cost = max(delay_delta, 0.0)
    improvement_bonus = max(-delay_delta, 0.0)

    parts = {
        "reward_env": float(env_reward),
        "reward_raw_delay_ms": raw_delay,
        "reward_delay_cost": delay,
        "reward_latency_utility": 0.0,
        "reward_deadline_utility": 0.0,
        "reward_power_utility": 0.0,
        "reward_packet_balance": 0.0,
        "reward_deadline_cost": float(getattr(args, "v2x_deadline_penalty", 2.0)) * deadline_excess_ms,
        "reward_margin_cost": float(getattr(args, "v2x_margin_penalty", 3.0)) * margin_pressure,
        "reward_power_cost": float(getattr(args, "v2x_power_penalty", 2.5)) * power_pressure,
        "reward_packet_cost": float(getattr(args, "v2x_packet_penalty", 2.0)) * packet_pressure,
        "reward_trend_cost": float(getattr(args, "v2x_trend_penalty", 0.45)) * trend_cost,
        "reward_improvement_bonus": float(getattr(args, "v2x_trend_bonus", 0.20)) * improvement_bonus,
        "reward_risk_penalty": v2x_risk(delay, deadline),
        "reward_llm_bonus": llm_bonus,
    }
    composite_cost = (
        parts["reward_delay_cost"]
        + parts["reward_deadline_cost"]
        + parts["reward_margin_cost"]
        + parts["reward_power_cost"]
        + parts["reward_packet_cost"]
        + parts["reward_trend_cost"]
        - parts["reward_improvement_bonus"]
    )
    return -float(composite_cost) + llm_bonus, parts
