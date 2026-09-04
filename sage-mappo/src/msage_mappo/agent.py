from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
import torch.nn.functional as F


@dataclass
class PPOConfig:
    obs_dim: int
    state_dim: int
    num_paths: int = 2
    memory_dim: int = 64
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    risk_coef: float = 0.35
    ppo_epochs: int = 4
    device: str = "cpu"


class HybridActor(nn.Module):
    def __init__(self, cfg: PPOConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.obs_dim + cfg.memory_dim + 4, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
        )
        self.path_head = nn.Linear(cfg.hidden_dim, cfg.num_paths)
        self.power_mean = nn.Linear(cfg.hidden_dim, 1)
        self.power_log_std = nn.Parameter(torch.tensor([-0.6], dtype=torch.float32))

    def forward(self, obs, memory, context):
        x = torch.cat([obs, memory, context], dim=-1)
        h = self.encoder(x)
        logits = self.path_head(h)
        mean = torch.sigmoid(self.power_mean(h)).squeeze(-1)
        std = F.softplus(self.power_log_std).expand_as(mean) + 1e-4
        return logits, mean, std


class CentralCritic(nn.Module):
    def __init__(self, cfg: PPOConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim + cfg.memory_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
        )
        self.return_head = nn.Linear(cfg.hidden_dim, 1)
        self.risk_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, state, global_memory):
        h = self.net(torch.cat([state, global_memory], dim=-1))
        value = self.return_head(h).squeeze(-1)
        risk = torch.sigmoid(self.risk_head(h)).squeeze(-1)
        return value, risk


class MSAGEMAPPO:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.actor = HybridActor(cfg).to(self.device)
        self.critic = CentralCritic(cfg).to(self.device)
        self.optim = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.lr,
        )

    def act(self, obs_np, memory_np, context_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        memory = torch.as_tensor(memory_np, dtype=torch.float32, device=self.device)
        context = torch.as_tensor(context_np, dtype=torch.float32, device=self.device)
        logits, power_mean, power_std = self.actor(obs, memory, context)
        path_dist = Categorical(logits=logits)
        power_dist = Normal(power_mean, power_std)
        path = path_dist.sample()
        raw_power = power_dist.rsample()
        power = torch.clamp(raw_power, 0.0, 1.0)
        log_prob = path_dist.log_prob(path) + power_dist.log_prob(raw_power)
        entropy = path_dist.entropy() + power_dist.entropy()
        return {
            "path": path.detach().cpu().numpy(),
            "power": power.detach().cpu().numpy(),
            "log_prob": log_prob.detach().cpu().numpy(),
            "entropy": entropy.detach().cpu().numpy(),
        }

    def value(self, state_np, memory_np):
        state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        memory = torch.as_tensor(memory_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value, risk = self.critic(state, memory)
        return float(value.item()), float(risk.item())

    def update(self, rollout: dict):
        obs = torch.as_tensor(np.asarray(rollout["obs"]), dtype=torch.float32, device=self.device)
        mem = torch.as_tensor(np.asarray(rollout["mem"]), dtype=torch.float32, device=self.device)
        ctx = torch.as_tensor(np.asarray(rollout["ctx"]), dtype=torch.float32, device=self.device)
        paths = torch.as_tensor(np.asarray(rollout["path"]), dtype=torch.int64, device=self.device)
        powers = torch.as_tensor(np.asarray(rollout["power"]), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(np.asarray(rollout["log_prob"]), dtype=torch.float32, device=self.device)

        states = torch.as_tensor(np.asarray(rollout["state"]), dtype=torch.float32, device=self.device)
        global_mem = torch.as_tensor(np.asarray(rollout["global_mem"]), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(np.asarray(rollout["returns"]), dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(np.asarray(rollout["advantages"]), dtype=torch.float32, device=self.device)
        risk_targets = torch.as_tensor(np.asarray(rollout["risk_targets"]), dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        losses = []
        for _ in range(self.cfg.ppo_epochs):
            logits, pmean, pstd = self.actor(obs, mem, ctx)
            path_dist = Categorical(logits=logits)
            power_dist = Normal(pmean, pstd)
            logp = path_dist.log_prob(paths) + power_dist.log_prob(powers)
            entropy = (path_dist.entropy() + power_dist.entropy()).mean()

            value, risk = self.critic(states, global_mem)
            risk_adv = risk.detach() - risk_targets
            total_adv = advantages - self.cfg.risk_coef * risk_adv
            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * total_adv
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * total_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, returns)
            risk_loss = F.binary_cross_entropy(risk, risk_targets.clamp(0.0, 1.0))
            loss = policy_loss + self.cfg.value_coef * value_loss + self.cfg.risk_coef * risk_loss - self.cfg.entropy_coef * entropy

            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 1.0)
            self.optim.step()
            losses.append(
                {
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "risk_loss": float(risk_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "total_loss": float(loss.detach().cpu()),
                }
            )
        return losses[-1]


def compute_returns_advantages(rewards, values, dones, gamma=0.97, gae_lambda=0.95):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values + [0.0], dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return returns, advantages
