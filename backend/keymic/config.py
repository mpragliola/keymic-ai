"""Constants shared with the browser frontend (src/config.js) — keep both in sync."""
from dataclasses import dataclass

CAPTURE_MS  = 200   # audio window per keystroke (ms)
N_MELS      = 40    # mel filterbank bands
FMIN_HZ     = 300   # mel filterbank range (Hz): below = rumble/hum, above = mostly mic hiss
FMAX_HZ     = 16000
FRAME_LEN   = 512   # STFT frame size (samples)
HOP_LEN     = 128   # STFT hop size (samples)
N_FRAMES    = 64    # fixed time-axis length fed to the CNN
MAX_SAMPLES = 50    # per key; oldest samples are dropped beyond this
MIN_SAMPLES = 8     # a key needs this many samples to be included in training
MIN_KEYS    = 2     # training needs at least this many eligible keys


@dataclass
class TrainParams:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    val_split: float = 0.2


def sample_length(sr: int) -> int:
    """Samples per keystroke clip. Rounds half up, matching Math.round in the frontend."""
    return int(CAPTURE_MS * sr / 1000 + 0.5)
