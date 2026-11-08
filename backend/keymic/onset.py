"""Re-detects each keystroke's click directly from its stored raw clip, instead of
trusting where the browser's capture-time cut happened to place it.

Every stored clip is CAPTURE_MS of raw audio (not just the cropped FEATURE_MS window),
so the actual click can be relocated here on every training run and every prediction —
which means a clip recorded with an imperfect capture-time cut (a biased latency
estimate, a borderline threshold call) gets corrected automatically every time it's
used, with no need to re-record anything.

Mirrors src/audio.js's detector in spirit (high-pass → short-time energy envelope →
background percentile → amplitude refine), not bit-for-bit — this runs once per clip
on the CPU, not per epoch, so a plain NumPy loop is fast enough.
"""
import numpy as np

from . import config as C

_DETECT_HIGHPASS_HZ = 800
_BACKGROUND_PERCENTILE = 20


def _highpass(x: np.ndarray, sr: float, fc: float) -> np.ndarray:
    """4th-order Butterworth high-pass: two cascaded RBJ biquads (Q = 1/sqrt(2)), matching
    the frontend's detector (src/audio.js's `highpass`)."""
    w0 = 2 * np.pi * fc / sr
    c, alpha = np.cos(w0), np.sin(w0) / (2 * 0.7071067811865476)
    a0 = 1 + alpha
    b0, b1, b2 = (1 + c) / 2 / a0, -(1 + c) / a0, (1 + c) / 2 / a0
    a1, a2 = -2 * c / a0, (1 - alpha) / a0
    y = x.astype(np.float64)
    for _ in range(2):
        out = np.empty_like(y)
        x1 = x2 = y1 = y2 = 0.0
        for i, v in enumerate(y):
            o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, v, y1, o
            out[i] = o
        y = out
    return y


def find_click(clip: np.ndarray, sr: int) -> int:
    """The clip's click, as a sample index. Searches the first ONSET_SEARCH_MS of the
    clip for the earliest point ONSET_THRESHOLD_DB above the clip's own background —
    earliest rather than loudest, so a later release (louder, if the key was held a
    while) isn't mistaken for the press. Falls back to PRE_ONSET_MS (where a clip is
    expected to have its click) if nothing crosses the threshold."""
    expected = round(C.PRE_ONSET_MS * sr / 1000)
    win = max(1, round(5 * sr / 1000))
    y = _highpass(clip, sr, _DETECT_HIGHPASS_HZ)
    energy = y.astype(np.float64) ** 2
    cum = np.concatenate([[0.0], np.cumsum(energy)])
    env = cum[win:] - cum[:-win]   # env[i] = energy of samples [i, i+win)
    if len(env) == 0:
        return expected

    background = np.percentile(env, _BACKGROUND_PERCENTILE) + 1e-12
    env_db = 10 * np.log10((env + 1e-12) / background)

    hi = min(len(env_db), round(C.ONSET_SEARCH_MS * sr / 1000))
    if hi == 0:
        return expected
    above = np.flatnonzero(env_db[:hi] > C.ONSET_THRESHOLD_DB)
    bi = int(above[0]) if len(above) else int(np.argmax(env_db[:hi]))

    # Refine: the best energy *window*'s start lags the actual peak sample by a few ms —
    # pinpoint the loudest sample in and just past it.
    refine_hi = min(len(y), bi + win + round(4 * sr / 1000))
    return bi + int(np.argmax(np.abs(y[bi:refine_hi])))


def recenter(clips: np.ndarray, sr: int, out_len: int) -> np.ndarray:
    """[N, L] stored clips → [N, out_len], each re-cropped to start PRE_ONSET_MS before
    its re-detected click. Zero-padded at either edge if the window runs past the clip."""
    pre = round(C.PRE_ONSET_MS * sr / 1000)
    out = np.zeros((len(clips), out_len), dtype=np.float32)
    for i, clip in enumerate(clips):
        start = find_click(clip, sr) - pre
        src_lo, src_hi = max(0, start), max(0, min(len(clip), start + out_len))
        dst_lo = src_lo - start
        out[i, dst_lo:dst_lo + (src_hi - src_lo)] = clip[src_lo:src_hi]
    return out
