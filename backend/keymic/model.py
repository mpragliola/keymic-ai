"""CNN classifier, sized for the small, per-user datasets this app actually trains on
(tens of clips per key, not thousands) — a model with far more capacity than that
memorizes the training set instead of learning the keystroke's acoustic signature.
"""
import torch.nn as nn


def _conv_block(c_in: int, c_out: int, drop: float) -> nn.Sequential:
    # Bias is redundant before BatchNorm, which has its own shift
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(drop),   # zeroes whole feature maps, not individual pixels
    )


class KeystrokeCNN(nn.Module):
    """Input [B, 1, 64, 40] log-mel spectrogram → logits [B, n_classes]."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 16, 0.1),    # → [16, 32, 20]
            _conv_block(16, 32, 0.15),  # → [32, 16, 10]
            _conv_block(32, 64, 0.2),   # → [64, 8, 5]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(32, n_classes),  # logits; softmax is applied at prediction time
        )

    def forward(self, x):
        return self.head(self.features(x))
