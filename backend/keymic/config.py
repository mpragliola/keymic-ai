"""Constants shared with the browser frontend (src/config.js) — keep both in sync."""
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
N_MELS      = 40    # mel filterbank bands
FMIN_HZ     = 300   # mel filterbank range (Hz): below = rumble/hum, above = mostly mic hiss
FMAX_HZ     = 16000
FRAME_LEN   = 512   # STFT frame size (samples)
HOP_LEN     = 128   # STFT hop size (samples)
# Fixed time-axis length fed to the CNN. A FEATURE_MS=100 clip yields ~31-34 real STFT
# frames (depending on sr: 44.1 vs 48 kHz) — N_FRAMES=64 (sized for the old 200ms window)
# would zero-pad roughly half the input. 32 keeps padding to a frame or two either way,
# and divides evenly by the three 2x poolings in model.py.
N_FRAMES    = 32
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
