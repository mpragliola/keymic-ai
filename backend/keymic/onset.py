"""Re-detects each keystroke's click directly from its stored raw clip, instead of
trusting where the browser's capture-time cut happened to place it.

Every stored clip is CAPTURE_MS of raw audio (not just the cropped FEATURE_MS window),
so the actual click can be relocated here on every training run and every prediction —
which means a clip recorded with an imperfect capture-time cut (a biased latency
estimate, a borderline threshold call) gets corrected automatically every time it's
used, with no need to re-record anything.

Mirrors src/audio.js's detector in spirit (high-pass → short-time energy envelope →
background percentile → amplitude refine), not bit-for-bit. Works on a batch of clips
at once: the IIR filter has to step through time sample by sample, but each step is
one vector operation across every clip in the batch, so cost grows with clip length,
not with the number of clips.
"""
import numpy as np

from . import config as C

_DETECT_HIGHPASS_HZ = 800
_BACKGROUND_PERCENTILE = 20
_CHUNK = 1024   # clips filtered at once — bounds memory (~80 MB per float64 array at 48 kHz)


def _highpass(x: np.ndarray, sr: float, fc: float) -> np.ndarray:
    """[N, L] → [N, L] float64. 4th-order Butterworth high-pass: two cascaded RBJ biquads
    (Q = 1/sqrt(2)), matching the frontend's detector (src/audio.js's `highpass`)."""
    w0 = 2 * np.pi * fc / sr
    c, alpha = np.cos(w0), np.sin(w0) / (2 * 0.7071067811865476)
    a0 = 1 + alpha
    b0, b1, b2 = (1 + c) / 2 / a0, -(1 + c) / a0, (1 + c) / 2 / a0
    a1, a2 = -2 * c / a0, (1 - alpha) / a0

    # Time-major copy so each step reads/writes one contiguous row of N values
    y = np.ascontiguousarray(x.T, dtype=np.float64)   # [L, N]
    n = y.shape[1]
    for _ in range(2):
        out = np.empty_like(y)
        x1, x2, y1, y2 = (np.zeros(n) for _ in range(4))
        for i in range(y.shape[0]):
            v = y[i]
            o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, v, y1, o
            out[i] = o
        y = out
    return y.T


def find_clicks(clips: np.ndarray, sr: int) -> np.ndarray:
    """[N, L] clips → [N] sample index of each clip's click. Searches the first
    ONSET_SEARCH_MS for the earliest point ONSET_THRESHOLD_DB above the clip's own
    background — earliest rather than loudest, so a later release (louder, if the key
    was held a while) isn't mistaken for the press. Falls back to the loudest point in
    that range if nothing crosses the threshold."""
    n, L = clips.shape
    expected = round(C.PRE_ONSET_MS * sr / 1000)
    win = max(1, round(5 * sr / 1000))
    hi = min(L - win + 1, round(C.ONSET_SEARCH_MS * sr / 1000))
    if n == 0 or hi <= 0:
        return np.full(n, expected, dtype=np.int64)

    y = _highpass(clips, sr, _DETECT_HIGHPASS_HZ)
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(y * y, axis=1)], axis=1)
    env = cum[:, win:] - cum[:, :-win]                                   # [N, L-win+1]
    background = np.percentile(env, _BACKGROUND_PERCENTILE, axis=1, keepdims=True) + 1e-12
    env_db = 10 * np.log10((env[:, :hi] + 1e-12) / background)

    above = env_db > C.ONSET_THRESHOLD_DB
    bi = np.where(above.any(axis=1), above.argmax(axis=1), env_db.argmax(axis=1))

    # Refine to the loudest single sample in and just past the 5 ms window: the window
    # with the most energy starts a few ms after the true onset (see src/audio.js).
    span = np.arange(win + round(4 * sr / 1000))
    idx = bi[:, None] + span[None, :]
    seg = np.abs(np.take_along_axis(y, np.minimum(idx, L - 1), axis=1))
    seg[idx >= L] = -1
    return bi + seg.argmax(axis=1)


def find_click(clip: np.ndarray, sr: int) -> int:
    """Single-clip convenience wrapper around find_clicks."""
    return int(find_clicks(clip[np.newaxis], sr)[0])


def recenter(clips: np.ndarray, sr: int, out_len: int) -> np.ndarray:
    """[N, L] stored clips → [N, out_len] float32, each re-cropped to start PRE_ONSET_MS
    before its re-detected click. Zero-padded at either edge if the window runs past
    the clip."""
    n, L = clips.shape
    pre = round(C.PRE_ONSET_MS * sr / 1000)
    starts = np.concatenate([find_clicks(clips[i:i + _CHUNK], sr) for i in range(0, n, _CHUNK)]
                            or [np.zeros(0, dtype=np.int64)]) - pre
    idx = starts[:, None] + np.arange(out_len)[None, :]
    valid = (idx >= 0) & (idx < L)
    out = np.take_along_axis(clips, np.clip(idx, 0, L - 1), axis=1) * valid
    return out.astype(np.float32)
