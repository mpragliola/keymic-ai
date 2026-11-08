"""CNN classifier, sized for the small, per-user datasets this app actually trains on
(tens of clips per key, not thousands) — a model with far more capacity than that
memorizes the training set instead of learning the keystroke's acoustic signature.

Each block below has two 3x3 convs instead of one (added depth without adding a fourth
pooling stage, which at this input size would leave almost nothing spatially — see the
shape comments). That's roughly 3x the conv parameters of the single-conv version this
replaced. Still small in absolute terms, but if a wider key/noise-class dataset ever
starts showing the train/val divergence §2.3 describes, this is the first place to look —
drop back to one conv per block before reaching for more dropout or data.
"""
import torch.nn as nn


def _conv_block(c_in: int, c_out: int, drop: float) -> nn.Sequential:
    # Bias is redundant before BatchNorm, which has its own shift. Second conv keeps
    # c_out -> c_out (depth, not width) and does not pool again.
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(drop),   # zeroes whole feature maps, not individual pixels
    )


class KeystrokeCNN(nn.Module):
    """Input [B, 1, 32, 40] log-mel spectrogram → logits [B, n_classes]."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 16, 0.1),    # → [16, 16, 20]
            _conv_block(16, 32, 0.15),  # → [32, 8, 10]
            _conv_block(32, 64, 0.2),   # → [64, 4, 5]
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
