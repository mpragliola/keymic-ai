"""Constants shared with the browser frontend (src/config.js) — keep the capture settings in sync."""
from dataclasses import dataclass

CAPTURE_MS  = 200   # audio window stored per keystroke (ms)
FEATURE_MS  = 100   # only this much of each stored clip feeds the model — a key's release
                    # (timing varies with how long it's held, and can blur into the next
                    # key's press during fast typing) is excluded; only the press is used
PRE_ONSET_MS = 20   # the feature window starts this long before the detected click
ONSET_SEARCH_MS = 100   # onset re-detection (onset.py) looks for the click within this
                        # much of the stored clip, biased toward the earliest crossing so a
                        # later, louder release doesn't get mistaken for the press
ONSET_THRESHOLD_DB = 6  # click = first point this many dB above the clip's own background
# Model features (backend only — src/config.js has separate settings for the Samples
# tab's preview images). Chosen on the recorded data: a 2048-sample FFT (~23 Hz bins at
# 48 kHz) resolves the fine resonance differences between keys that a 512-sample FFT
# with 40 bands blurred together; 128 bands from 50 Hz keep the low-end body resonance.
N_MELS      = 128   # mel filterbank bands
FMIN_HZ     = 50    # mel filterbank range (Hz)
FMAX_HZ     = 16000
FRAME_LEN   = 2048  # STFT frame size (samples)
HOP_LEN     = 256   # STFT hop size (samples) — 10–11 frames per FEATURE_MS clip
MAX_SAMPLES = 1000  # per key; oldest samples are dropped beyond this (~38 MB per key at 48 kHz)
MIN_SAMPLES = 8     # a key needs this many samples to be included in training
MIN_KEYS    = 2     # training needs at least this many eligible keys

# Training-time augmentation (raw audio, applied fresh each batch so the net
# can't memorize a session's exact noise floor / timing / loudness instead of
# the keystroke's actual acoustic signature). Validation is never augmented.
AUG_NOISE_MAX  = 0.06   # added noise, as a fraction of the clip's RMS, drawn uniformly up to this
AUG_GAIN_RANGE = (0.85, 1.15)   # random amplitude scale
AUG_SHIFT_MS   = 6      # random zero-padded time shift, ± this many ms
WEIGHT_DECAY   = 1e-4   # Adam L2 penalty


@dataclass
class TrainParams:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    val_split: float = 0.2
    resume: bool = False   # warm-start from the current saved model instead of a fresh net


def sample_length(sr: int) -> int:
    """Samples per stored keystroke clip. Rounds half up, matching Math.round in the frontend."""
    return int(CAPTURE_MS * sr / 1000 + 0.5)


def feature_length(sr: int) -> int:
    """Samples actually fed to the model — a leading crop of the stored clip."""
    return int(FEATURE_MS * sr / 1000 + 0.5)
