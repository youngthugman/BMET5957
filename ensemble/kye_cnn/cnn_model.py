"""Multi-scale CNN/BiGRU/attention network from Kye-1D_CNN."""
from __future__ import annotations

from .cnn_cache import CONTEXT_WINDOWS


def _torch():
    import torch
    return torch


class MultiScaleAttentionModel(_torch().nn.Module):
    """Consume four centred contexts of 35 channels and return one logit."""
    def __init__(self, input_channels=35, hidden_size=64, dropout=.30):
        super().__init__()
        torch = _torch(); nn = torch.nn
        if input_channels != 35: raise ValueError("Kye-1D_CNN requires exactly 35 channels")
        self.context_windows = CONTEXT_WINDOWS
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Conv1d(input_channels, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                          nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                          nn.AdaptiveAvgPool1d(1)) for _ in CONTEXT_WINDOWS])
        self.bigru = nn.GRU(64, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(),
                                       nn.Linear(hidden_size, 1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size * 2, 1))

    def forward(self, contexts):
        torch = _torch()
        encoded = [encoder(value.transpose(1, 2)).squeeze(-1)
                   for encoder, value in zip(self.encoders, contexts)]
        sequence, _ = self.bigru(torch.stack(encoded, dim=1))
        weight = torch.softmax(self.attention(sequence), dim=1)
        return self.classifier((sequence * weight).sum(1)).squeeze(1)
