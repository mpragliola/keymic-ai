"""On-disk sample store.

Layout under the data directory:
    meta.json                {"sr": <sample rate of all stored clips> | null}
    samples/key_<hex>.npy    float32 [n, sample_length(sr)] — one file per key;
                             the filename is the key's UTF-8 bytes in hex, so
                             keys like ' ' are filesystem-safe.

All clips share one sample rate: mixing rates would give spectrograms with
different frequency scales. The rate resets once every sample is deleted.
"""
import json
import os
import threading
from pathlib import Path

import numpy as np

from . import config as C


class StoreError(ValueError):
    pass


class SampleRateMismatch(StoreError):
    pass


class SampleStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.dir = self.root / "samples"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "meta.json"
        self._lock = threading.RLock()

    # ── metadata ─────────────────────────────────────────────────
    def _meta(self) -> dict:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text())
        return {"sr": None}

    def _write_meta(self, meta: dict) -> None:
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, self.meta_path)

    @property
    def sr(self) -> int | None:
        with self._lock:
            return self._meta()["sr"]

    # ── paths ────────────────────────────────────────────────────
    def _path(self, key: str) -> Path:
        return self.dir / f"key_{key.encode('utf-8').hex()}.npy"

    @staticmethod
    def _key_from_path(p: Path) -> str:
        return bytes.fromhex(p.stem[len("key_"):]).decode("utf-8")

    # ── reads ────────────────────────────────────────────────────
    def counts(self) -> dict[str, int]:
        with self._lock:
            out = {}
            for p in sorted(self.dir.glob("key_*.npy")):
                n = np.load(p, mmap_mode="r").shape[0]
                if n:
                    out[self._key_from_path(p)] = int(n)
            return out

    def load(self, key: str) -> np.ndarray:
        """All clips for `key` as float32 [n, L]; empty [0, L] (or [0, 0]) if none."""
        with self._lock:
            p = self._path(key)
            if p.exists():
                return np.load(p)
            sr = self.sr
            return np.zeros((0, C.sample_length(sr) if sr else 0), np.float32)

    # ── writes ───────────────────────────────────────────────────
    def add(self, key: str, sr: int, clips: np.ndarray) -> int:
        """Appends clips [n, L] for `key`, keeping the newest MAX_SAMPLES. Returns the new count."""
        with self._lock:
            meta = self._meta()
            if meta["sr"] is None:
                meta["sr"] = int(sr)
                self._write_meta(meta)
            elif meta["sr"] != sr:
                raise SampleRateMismatch(
                    f"stored samples use {meta['sr']} Hz, got {sr} Hz — delete samples to switch rates")

            L = C.sample_length(sr)
            if clips.ndim != 2 or clips.shape[1] != L:
                raise StoreError(f"expected clips of {L} samples at {sr} Hz, got shape {clips.shape}")

            combined = np.concatenate([self.load(key), clips.astype(np.float32)])[-C.MAX_SAMPLES:]
            p = self._path(key)
            tmp = p.with_suffix(".tmp")
            with open(tmp, "wb") as f:   # file handle: np.save would append ".npy" to a str path
                np.save(f, combined)
            os.replace(tmp, p)
            return int(combined.shape[0])

    def delete(self, key: str) -> bool:
        with self._lock:
            p = self._path(key)
            existed = p.exists()
            if existed:
                p.unlink()
            self._reset_sr_if_empty()
            return existed

    def clear(self) -> None:
        with self._lock:
            for p in self.dir.glob("key_*"):
                p.unlink()
            self._reset_sr_if_empty()

    def _reset_sr_if_empty(self) -> None:
        if not any(self.dir.glob("key_*.npy")):
            self._write_meta({"sr": None})
