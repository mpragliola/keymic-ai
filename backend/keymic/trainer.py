"""Background training on the GPU, plus the inference model.

One training run at a time runs in a worker thread. The API polls status()
for logs and per-epoch metrics. stop() takes effect at the next batch; a stopped
run is discarded and the previously saved model stays in place.
"""
import threading
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .features import MelSpectrogram
from .model import KeystrokeCNN
from .store import SampleStore, SampleRateMismatch, StoreError

CHECKPOINT_FORMAT = 1


class TrainingBusy(RuntimeError):
    pass


class NoModel(LookupError):
    pass


@dataclass
class LoadedModel:
    net: KeystrokeCNN
    keys: list[str]
    sr: int
    mel: MelSpectrogram
    info: dict


class Trainer:
    def __init__(self, store: SampleStore, model_path: Path, device: torch.device):
        self.store = store
        self.model_path = Path(model_path)
        self.device = device
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reset_run_state("idle")
        self.model: LoadedModel | None = self._load_saved()
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True   # fixed input shape → let cuDNN pick fastest kernels

    # ── public API ───────────────────────────────────────────────
    def status(self, since: int = 0) -> dict:
        with self._lock:
            return {
                "state": self.state,          # idle | running | stopping | done | stopped | error
                "epoch": self.epoch,
                "epochs": self.epochs,
                "keys": self.keys,
                "error": self.error,
                "history": {k: list(v) for k, v in self.history.items()},
                "logs": self.logs[since:],
                "log_total": len(self.logs),
            }

    def start(self, params: C.TrainParams) -> dict:
        with self._lock:
            if self.state in ("running", "stopping"):
                raise TrainingBusy("a training run is already active")
            keys = self._eligible_keys()   # raises StoreError if there's not enough data
            self._reset_run_state("running")
            self.keys, self.epochs = keys, params.epochs
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, args=(params, keys), daemon=True)
            self._thread.start()
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            if self.state == "running":
                self.state = "stopping"
                self._stop.set()
                self.logs.append("Stopping after the current batch…")
        return self.status()

    def wait(self, timeout: float | None = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)

    def delete_model(self) -> None:
        """Stops any running training (so it can't save afterwards) and removes the saved model."""
        self.stop()
        self.wait(timeout=60)
        with self._lock:
            self.model = None
            self.model_path.unlink(missing_ok=True)

    def model_info(self) -> dict:
        with self._lock:
            if self.model is None:
                raise NoModel("no trained model")
            return dict(self.model.info, keys=self.model.keys, sr=self.model.sr)

    def predict(self, audio: np.ndarray, sr: int) -> dict:
        with self._lock:
            m = self.model
        if m is None:
            raise NoModel("no trained model")
        if sr != m.sr:
            raise SampleRateMismatch(f"model was trained at {m.sr} Hz, got {sr} Hz")
        L = C.sample_length(sr)
        if audio.shape != (L,):
            raise StoreError(f"expected one clip of {L} samples, got {audio.shape[0]}")

        x = m.mel(torch.from_numpy(audio).to(self.device).unsqueeze(0))
        with torch.no_grad():
            probs = torch.softmax(m.net(x), dim=1)[0].cpu().numpy()
        order = np.argsort(-probs)
        return {
            "key": m.keys[order[0]],
            "probs": [{"key": m.keys[i], "p": float(probs[i])} for i in order],
        }

    # ── internals ────────────────────────────────────────────────
    def _reset_run_state(self, state: str) -> None:
        self.state = state
        self.epoch = 0
        self.epochs = 0
        self.keys: list[str] | None = None
        self.error: str | None = None
        self.logs: list[str] = []
        self.history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    def _eligible_keys(self) -> list[str]:
        keys = sorted(k for k, n in self.store.counts().items() if n >= C.MIN_SAMPLES)
        if len(keys) < C.MIN_KEYS:
            raise StoreError(f"need at least {C.MIN_KEYS} keys with {C.MIN_SAMPLES}+ samples each "
                             f"(have {len(keys)})")
        return keys

    def _log(self, msg: str) -> None:
        with self._lock:
            self.logs.append(msg)

    def _run(self, p: C.TrainParams, keys: list[str]) -> None:
        try:
            self._train(p, keys)
        except Exception as e:  # surfaced to the client through status()
            with self._lock:
                self.state = "error"
                self.error = f"{type(e).__name__}: {e}"
                self.logs.append(f"Error: {self.error}")
                self.logs.append(traceback.format_exc().strip().splitlines()[-1])

    def _train(self, p: C.TrainParams, keys: list[str]) -> None:
        dev = self.device
        sr = self.store.sr
        name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
        self._log(f"Device: {name} · training on keys: {', '.join(repr(k) for k in keys)}")

        clips, labels = [], []
        for ci, k in enumerate(keys):
            a = self.store.load(k)
            self._log(f"  {k!r}: {len(a)} samples")
            clips.append(a)
            labels += [ci] * len(a)
        labels_np = np.asarray(labels)

        # Spectrograms for the whole dataset, computed once on the GPU
        mel = MelSpectrogram(sr, dev)
        audio = torch.from_numpy(np.concatenate(clips)).to(dev)
        x = torch.cat([mel(audio[i:i + 512]) for i in range(0, len(audio), 512)])
        y = torch.from_numpy(labels_np).to(dev)
        del audio

        # Stratified split: VAL_SPLIT of every key goes to validation, so each
        # key appears in both sets.
        rng = np.random.default_rng()
        train_idx, val_idx = [], []
        for ci in range(len(keys)):
            idx = np.flatnonzero(labels_np == ci)
            rng.shuffle(idx)
            n_val = max(1, round(len(idx) * p.val_split))
            val_idx += idx[:n_val].tolist()
            train_idx += idx[n_val:].tolist()
        train_idx = torch.tensor(train_idx, device=dev)
        val_idx = torch.tensor(val_idx, device=dev)
        self._log(f"Total {len(labels)} samples ({len(train_idx)} train / {len(val_idx)} validation)")

        net = KeystrokeCNN(len(keys)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=p.lr)
        loss_fn = nn.CrossEntropyLoss()
        t0 = time.perf_counter()

        for epoch in range(p.epochs):
            net.train()
            perm = train_idx[torch.randperm(len(train_idx), device=dev)]
            loss_sum = torch.zeros((), device=dev)
            correct = torch.zeros((), device=dev)
            for b in range(0, len(perm), p.batch_size):
                if self._stop.is_set():
                    break
                idx = perm[b:b + p.batch_size]
                logits = net(x[idx])
                loss = loss_fn(logits, y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                # Accumulate on the GPU; one host sync per epoch instead of per batch
                loss_sum += loss.detach() * len(idx)
                correct += (logits.argmax(1) == y[idx]).sum()
            if self._stop.is_set():
                break

            net.eval()
            with torch.no_grad():
                v_logits = torch.cat([net(x[val_idx[i:i + 512]]) for i in range(0, len(val_idx), 512)])
                v_loss = loss_fn(v_logits, y[val_idx]).item()
                v_acc = (v_logits.argmax(1) == y[val_idx]).float().mean().item()
            tr_loss = loss_sum.item() / len(train_idx)
            tr_acc = correct.item() / len(train_idx)

            with self._lock:
                self.epoch = epoch + 1
                for k, v in (("loss", tr_loss), ("acc", tr_acc), ("val_loss", v_loss), ("val_acc", v_acc)):
                    self.history[k].append(v)
                self.logs.append(f"Epoch {epoch + 1}/{p.epochs}  loss={tr_loss:.4f}  acc={tr_acc:.3f}  "
                                 f"val_loss={v_loss:.4f}  val_acc={v_acc:.3f}")

        if self._stop.is_set():
            return self._finish_stopped()

        info = {
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "epochs": p.epochs,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "val_acc": self.history["val_acc"][-1],
            "train_seconds": round(time.perf_counter() - t0, 2),
            "device": name,
        }
        self._save(net, keys, sr, info)

        # A stop that arrived during saving (e.g. Delete model) wins: undo the save
        if self._stop.is_set():
            self.model_path.unlink(missing_ok=True)
            return self._finish_stopped()

        net.eval()
        with self._lock:
            self.model = LoadedModel(net, keys, sr, mel, info)
            self.state = "done"
            self.logs.append(f"Done ✓ in {info['train_seconds']} s — model saved")

    def _finish_stopped(self) -> None:
        with self._lock:
            self.state = "stopped"
            self.logs.append("Training stopped — nothing saved.")

    def _save(self, net: KeystrokeCNN, keys: list[str], sr: int, info: dict) -> None:
        ckpt = {"format": CHECKPOINT_FORMAT, "state_dict": net.state_dict(), "keys": keys, "sr": sr, "info": info}
        tmp = self.model_path.with_suffix(".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(self.model_path)   # atomic: a crash never leaves a half-written model

    def _load_saved(self) -> LoadedModel | None:
        if not self.model_path.exists():
            return None
        # weights_only: the checkpoint holds only tensors and plain types, so no pickle code runs
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=True)
        net = KeystrokeCNN(len(ckpt["keys"])).to(self.device)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return LoadedModel(net, ckpt["keys"], ckpt["sr"], MelSpectrogram(ckpt["sr"], self.device), ckpt["info"])
