from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class V2XActor(nn.Module):
    def __init__(self, obs_dim: int, memory_dim: int, ctx_dim: int, hidden: int, n_bs: int, n_packet_choices: int):
        super().__init__()
        self.n_bs = n_bs
        self.n_packet_choices = n_packet_choices
        self.net = nn.Sequential(
            nn.Linear(obs_dim + memory_dim + ctx_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.obs_net = nn.Sequential(
            nn.Linear(obs_dim + ctx_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mem_proj = nn.Sequential(
            nn.Linear(memory_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.mem_gate = nn.Sequential(
            nn.Linear(obs_dim + memory_dim + ctx_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.dirichlet_head = nn.Linear(hidden, n_bs)
        self.packet_head = nn.Linear(hidden, n_bs * n_packet_choices)

    def forward(self, obs, mem, ctx):
        x = torch.cat([obs, mem, ctx], dim=-1)
        h_base = self.obs_net(torch.cat([obs, ctx], dim=-1))
        h_mem = self.mem_proj(mem)
        gate = self.mem_gate(x)
        h = torch.nan_to_num(h_base + gate * h_mem, nan=0.0, posinf=8.0, neginf=-8.0)
        raw_concentration = torch.nan_to_num(self.dirichlet_head(h), nan=0.0, posinf=8.0, neginf=-8.0)
        concentration = torch.clamp(F.softplus(raw_concentration) + 0.3, 0.3, 8.0)
        packet_logits = self.packet_head(h).view(-1, self.n_bs, self.n_packet_choices)
        packet_logits = torch.nan_to_num(packet_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        return concentration, packet_logits


class V2XCritic(nn.Module):
    def __init__(self, state_dim: int, memory_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + memory_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.risk = nn.Linear(hidden, 1)

    def forward(self, state, mem):
        h = self.net(torch.cat([state, mem], dim=-1))
        return self.value(h).squeeze(-1), torch.sigmoid(self.risk(h)).squeeze(-1)


class V2XLagrangianCritic(nn.Module):
    """Centralized reward and discounted-cost critic for MAPPO-Lagrangian."""

    def __init__(self, state_dim: int, memory_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + memory_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.cost_value = nn.Linear(hidden, 1)

    def forward(self, state, mem):
        h = self.net(torch.cat([state, mem], dim=-1))
        return self.value(h).squeeze(-1), self.cost_value(h).squeeze(-1)
