"""Keystroke classifier.

Summarizes each mel band of the click by its mean and standard deviation over time,
then classifies that 2·N_MELS vector with a small, heavily-regularized MLP.

Why not a CNN over the spectrogram: on real recordings (37 keys, ~100 presses each),
what distinguishes keys is their frequency fingerprint — fine resonance differences —
not the exact time course of the click within the window. Keeping the time layout
mostly added sensitivity to onset jitter; pooling it away was worth ~25 points of
held-out accuracy for a plain linear classifier (53% → 80%, trained on each key's
earlier presses and tested on its later ones). A hidden layer adds little on top of
linear for this data, so the net stays small; the dropout and weight decay do the
heavy lifting against the small per-key sample counts.
"""
import torch
import torch.nn as nn

from . import config as C


class KeystrokeNet(nn.Module):
    """Input [B, T, N_MELS] z-scored log-mel spectrogram → logits [B, n_classes]."""

    def __init__(self, n_classes: int, n_mels: int = C.N_MELS, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            # Per-feature standardization with running statistics, saved with the
            # weights — a linear layer on unstandardized log-mel stats trains poorly
            nn.BatchNorm1d(2 * n_mels),
            nn.Dropout(0.2),
            nn.Linear(2 * n_mels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, n_classes),   # logits; softmax is applied at prediction time
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=1)   # [B, 2·N_MELS]
        return self.net(stats)
