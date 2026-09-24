"""Exact LayerNorm/GELU Kye-MLP network."""
import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(.25),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(.25),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)
