"""CNN classifier — same layer stack as buildModel in src/model.js."""
import torch.nn as nn


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    # Bias is redundant before BatchNorm, which has its own shift
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class KeystrokeCNN(nn.Module):
    """Input [B, 1, 64, 40] log-mel spectrogram → logits [B, n_classes]."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 32),     # → [32, 32, 20]
            _conv_block(32, 64),    # → [64, 16, 10]
            _conv_block(64, 128),   # → [128, 8, 5]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),  # logits; softmax is applied at prediction time
        )

    def forward(self, x):
        return self.head(self.features(x))
