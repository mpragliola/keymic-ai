"""Log-mel spectrogram front-end, computed on the GPU in batches.

Mirrors computeMelSpec in src/features.js step by step, so spectrograms match
between browser and backend: frame → Hann window → rFFT → power → mel
filterbank → log → per-spectrogram z-score → pad/crop to N_FRAMES.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import config as C


def mel_filterbank(n_mels: int, fft_len: int, sr: int) -> np.ndarray:
    """[n_fft_bins, n_mels] matrix of triangular mel filters (same construction as the JS version)."""
    n_bins = fft_len // 2 + 1
    to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    from_mel = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mel_min, mel_max = to_mel(0.0), to_mel(sr / 2)

    # n_mels + 2 evenly spaced mel points, converted to FFT bin indices
    pts = [int(np.floor((fft_len + 1) * from_mel(mel_min + i * (mel_max - mel_min) / (n_mels + 1)) / sr))
           for i in range(n_mels + 2)]

    fb = np.zeros((n_bins, n_mels), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, c, hi = pts[m - 1], pts[m], pts[m + 1]
        for k in range(lo, min(c, n_bins)):
            fb[k, m - 1] = (k - lo) / (c - lo + 1e-9)
        for k in range(c, min(hi + 1, n_bins)):
            fb[k, m - 1] = (hi - k) / (hi - c + 1e-9)
    return fb


class MelSpectrogram:
    """Turns a batch of raw clips [B, L] into CNN input [B, 1, N_FRAMES, N_MELS]."""

    def __init__(self, sr: int, device: torch.device):
        self.sr = sr
        self.fb = torch.from_numpy(mel_filterbank(C.N_MELS, C.FRAME_LEN, sr)).to(device)
        # Periodic Hann, which is what tf.signal.hannWindow produces for even lengths
        self.window = torch.hann_window(C.FRAME_LEN, periodic=True, device=device)

    @torch.no_grad()
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        frames = audio.unfold(-1, C.FRAME_LEN, C.HOP_LEN)          # [B, T, FRAME_LEN]
        spec = torch.fft.rfft(frames * self.window, dim=-1)         # [B, T, FRAME_LEN/2+1] complex
        power = spec.real.square() + spec.imag.square()
        log_mel = torch.log(power @ self.fb + 1e-6)                 # [B, T, N_MELS]

        # z-score over each whole spectrogram (population variance, like tf.moments)
        mean = log_mel.mean(dim=(1, 2), keepdim=True)
        std = log_mel.var(dim=(1, 2), unbiased=False, keepdim=True).sqrt()
        x = (log_mel - mean) / (std + 1e-8)

        t = x.shape[1]
        x = x[:, :C.N_FRAMES] if t >= C.N_FRAMES else F.pad(x, (0, 0, 0, C.N_FRAMES - t))
        return x.unsqueeze(1)                                       # [B, 1, N_FRAMES, N_MELS]
