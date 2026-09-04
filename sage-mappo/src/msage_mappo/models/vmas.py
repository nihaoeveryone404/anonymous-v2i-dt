from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class VMASActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, memory_dim: int, ctx_dim: int, hidden: int, log_std_init: float = -0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + memory_dim + ctx_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.obs_net = nn.Sequential(
            nn.Linear(obs_dim + ctx_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mem_proj = nn.Sequential(
            nn.Linear(memory_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
        )
        self.mem_gate = nn.Sequential(
            nn.Linear(obs_dim + memory_dim + ctx_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * float(log_std_init))

    def forward(self, obs, mem, ctx):
        x = torch.cat([obs, mem, ctx], dim=-1)
        h_base = self.obs_net(torch.cat([obs, ctx], dim=-1))
        h_mem = self.mem_proj(mem)
        gate = self.mem_gate(x)
        h = h_base + gate * h_mem
        return torch.tanh(self.mean(h)), F.softplus(self.log_std).expand(obs.shape[0], -1) + 1e-4


class VMASCritic(nn.Module):
    def __init__(self, state_dim: int, memory_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + memory_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.value = nn.Linear(hidden, 1)
        self.risk = nn.Linear(hidden, 1)

    def forward(self, state, mem):
        h = self.net(torch.cat([state, mem], dim=-1))
        return self.value(h).squeeze(-1), torch.sigmoid(self.risk(h)).squeeze(-1)
