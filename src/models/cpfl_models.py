"""Neural-network components shared by CPFL training and prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleExtractor(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 8, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(8, 25, kernel_size=5, padding=2), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PFL_REMNet(nn.Module):
    """Shared PL backbone with a dynamically sized personalized head."""

    def __init__(self, input_dim: int = 102, initial_k: int = 4,
                 two_layer_head: bool = False, head_dropout: float = 0.10):
        super().__init__()
        self.input_dim = input_dim
        self.initial_k = initial_k
        self.two_layer_head = two_layer_head
        self.extractors = nn.ModuleDict({key: SimpleExtractor(1) for key in "ABCD"})
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 1024), nn.SiLU(), nn.LayerNorm(1024),
            nn.Linear(1024, 512), nn.SiLU(), nn.LayerNorm(512),
        )
        if two_layer_head:
            self.head = nn.Sequential(
                nn.Dropout(p=head_dropout), nn.Linear(512, 128), nn.SiLU(), nn.Linear(128, initial_k)
            )
        else:
            self.head = nn.Sequential(nn.Dropout(p=head_dropout), nn.Linear(512, initial_k))
        self.current_k = int(initial_k)

    def forward(self, x):
        return self.head(self.backbone(x))

    def update_head_for_k(self, new_k: int):
        new_k = int(new_k)
        if self.current_k == new_k:
            return
        old_out = self.head[-1]
        new_out = nn.Linear(old_out.in_features, new_k).to(old_out.weight.device)
        nn.init.xavier_uniform_(new_out.weight)
        nn.init.zeros_(new_out.bias)
        keep = min(self.current_k, new_k)
        with torch.no_grad():
            new_out.weight[:keep].copy_(old_out.weight[:keep])
            new_out.bias[:keep].copy_(old_out.bias[:keep])
        self.head[-1] = new_out
        self.current_k = new_k


class PFL_KPIPredictor(nn.Module):
    """Global-average-pooled KPI regressor with a personalized output head."""

    def __init__(self, input_dim: int = 102, initial_kpi_outputs: int = 15,
                 hidden_dim: int = 384, dropout: float = 0.1207568,
                 out_dim: int | None = None):
        super().__init__()
        output_dim = initial_kpi_outputs if out_dim is None else out_dim
        self.input_dim = input_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 1024), nn.SiLU(), nn.LayerNorm(1024),
            nn.Linear(1024, 512), nn.SiLU(), nn.LayerNorm(512),
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(512, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim)
        )
        self.current_kpi_outputs = int(output_dim)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(torch.mean(features, dim=0, keepdim=True))
