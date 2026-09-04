from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToyV2XConfig:
    num_agents: int = 4
    episode_len: int = 32
    seed: int = 7
    queue_arrival_low: float = 0.08
    queue_arrival_high: float = 0.42
    deadline: float = 1.0
    max_power: float = 1.0
    violation_penalty: float = 2.0
    power_weight: float = 0.18


class ToyV2XEnv:
    """A lightweight hybrid-action V2X scheduling environment.

    Each agent chooses a discrete path and a continuous transmit-power level.
    The environment is intentionally small so algorithm smoke tests can run on
    a local machine before replacing it with the full V2X simulator or VMAS.
    """

    obs_dim = 5
    global_extra_dim = 3
    num_paths = 2

    def __init__(self, config: ToyV2XConfig | None = None):
        self.cfg = config or ToyV2XConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.t = 0
        self.sinr = np.zeros((self.cfg.num_agents, self.num_paths), dtype=np.float32)
        self.queue = np.zeros(self.cfg.num_agents, dtype=np.float32)
        self.last_power = np.zeros(self.cfg.num_agents, dtype=np.float32)
        self.deadline_margin = np.ones(self.cfg.num_agents, dtype=np.float32)

    @property
    def num_agents(self) -> int:
        return self.cfg.num_agents

    @property
    def state_dim(self) -> int:
        return self.num_agents * self.obs_dim + self.global_extra_dim

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.t = 0
        self.sinr = self.rng.uniform(0.45, 1.25, size=(self.num_agents, self.num_paths)).astype(np.float32)
        self.queue = self.rng.uniform(0.08, 0.25, size=self.num_agents).astype(np.float32)
        self.last_power.fill(0.0)
        self.deadline_margin.fill(1.0)
        return self._obs(), self._state()

    def step(self, path_action: np.ndarray, power_action: np.ndarray):
        self.t += 1
        path_action = path_action.astype(np.int64)
        power = np.clip(power_action.astype(np.float32), 0.03, self.cfg.max_power)

        arrivals = self.rng.uniform(
            self.cfg.queue_arrival_low,
            self.cfg.queue_arrival_high,
            size=self.num_agents,
        ).astype(np.float32)
        self.queue = np.clip(self.queue + arrivals, 0.0, 2.5)

        chosen_sinr = self.sinr[np.arange(self.num_agents), path_action]
        rate = np.log2(1.0 + 6.0 * chosen_sinr * (0.25 + power))
        served = np.minimum(self.queue, 0.22 * rate).astype(np.float32)
        remaining = np.maximum(self.queue - served, 0.0)

        latency = remaining / (0.05 + rate)
        violation = latency > self.cfg.deadline
        self.deadline_margin = np.clip((self.cfg.deadline - latency) / self.cfg.deadline, -1.0, 1.0)

        avg_latency = float(np.mean(latency))
        avg_power = float(np.mean(power))
        violation_rate = float(np.mean(violation.astype(np.float32)))

        reward = -avg_latency - self.cfg.power_weight * avg_power - self.cfg.violation_penalty * violation_rate
        self.queue = remaining.astype(np.float32)
        self.last_power = power

        fading = self.rng.normal(0.0, 0.08, size=self.sinr.shape).astype(np.float32)
        blockage = self.rng.random(self.sinr.shape) < 0.035
        self.sinr = np.clip(0.92 * self.sinr + 0.08 * self.rng.uniform(0.3, 1.4, self.sinr.shape) + fading, 0.05, 1.8)
        self.sinr[blockage] *= 0.35

        done = self.t >= self.cfg.episode_len
        info = {
            "avg_latency": avg_latency,
            "avg_power": avg_power,
            "violation_rate": violation_rate,
            "tail_latency": float(np.quantile(latency, 0.95)),
            "queue_pressure": float(np.mean(self.queue)),
            "deadline_margin": float(np.mean(self.deadline_margin)),
            "risk_target": violation_rate,
        }
        return self._obs(), self._state(), float(reward), done, info

    def _obs(self) -> np.ndarray:
        obs = np.stack(
            [
                self.sinr[:, 0],
                self.sinr[:, 1],
                self.queue,
                self.deadline_margin,
                self.last_power,
            ],
            axis=1,
        ).astype(np.float32)
        return obs

    def _state(self) -> np.ndarray:
        obs = self._obs().reshape(-1)
        global_feats = np.array(
            [
                np.mean(self.queue),
                np.mean(self.deadline_margin),
                np.mean(self.last_power),
            ],
            dtype=np.float32,
        )
        return np.concatenate([obs, global_feats], axis=0)

    @staticmethod
    def constraint_tags_from_info(info: dict) -> list[str]:
        tags: list[str] = []
        if info.get("violation_rate", 0.0) > 0.0:
            tags.append("deadline_violation")
        if info.get("queue_pressure", 0.0) > 0.45:
            tags.append("queue_congestion")
        if info.get("avg_power", 0.0) > 0.7:
            tags.append("power_pressure")
        if not tags:
            tags.append("stable_operation")
        return tags
