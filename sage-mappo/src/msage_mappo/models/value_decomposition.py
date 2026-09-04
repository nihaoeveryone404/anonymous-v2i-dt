from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.orthogonal_(self.advantage.weight, gain=1.0)

    def forward(self, obs):
        h = self.feature(obs)
        value = self.value(h)
        advantage = self.advantage(h)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class QMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, hidden: int):
        super().__init__()
        self.n_agents = n_agents
        self.state_norm = nn.LayerNorm(state_dim)
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden)
        self.hyper_b1 = nn.Linear(state_dim, hidden)
        self.hyper_w2 = nn.Linear(state_dim, hidden)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, agent_qs, states):
        states = self.state_norm(states)
        bs = agent_qs.shape[0]
        w1 = F.softplus(self.hyper_w1(states)).view(bs, self.n_agents, -1).clamp(max=2.0)
        b1 = self.hyper_b1(states).view(bs, 1, -1)
        h = F.elu(torch.bmm(agent_qs.view(bs, 1, self.n_agents), w1) + b1)
        w2 = F.softplus(self.hyper_w2(states)).view(bs, -1, 1).clamp(max=2.0)
        b2 = self.hyper_b2(states).view(bs, 1, 1)
        return (torch.bmm(h, w2) + b2).view(bs)
