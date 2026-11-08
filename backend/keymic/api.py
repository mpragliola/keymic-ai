"""HTTP API (FastAPI).

Audio travels as raw little-endian float32 PCM in the request body — the same
bytes as a Float32Array in the browser — with the sample rate as ?sr=.
"""
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config as C
from .store import SampleRateMismatch, SampleStore, StoreError
from .trainer import NoModel, Trainer, TrainingBusy

DATA_DIR = Path(os.environ.get("KEYMIC_DATA", Path(__file__).resolve().parent.parent / "data"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORS_ORIGINS = os.environ.get("KEYMIC_CORS", "http://localhost:8080,http://127.0.0.1:8080").split(",")

store = SampleStore(DATA_DIR)
trainer = Trainer(store, DATA_DIR / "model.pt", DEVICE)

app = FastAPI(title="KeyMic backend", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


class TrainRequest(BaseModel):
    epochs: int = Field(60, ge=1, le=2000)
    batch_size: int = Field(32, ge=1, le=4096)
    lr: float = Field(1e-3, gt=0, lt=1)
    val_split: float = Field(0.2, gt=0, lt=0.9)


def _http(e: Exception) -> HTTPException:
    """Maps domain errors to HTTP status codes."""
    if isinstance(e, (SampleRateMismatch, TrainingBusy)):
        return HTTPException(409, str(e))
    if isinstance(e, NoModel):
        return HTTPException(404, str(e))
    return HTTPException(400, str(e))


def _check_key(key: str) -> None:
    if len(key) != 1:
        raise HTTPException(400, "key must be a single character")


async def _read_pcm(request: Request) -> np.ndarray:
    body = await request.body()
    if not body or len(body) % 4:
        raise HTTPException(400, "body must be raw float32 PCM")
    return np.frombuffer(body, dtype="<f4").copy()


@app.get("/health")
def health():
    return {
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "data_dir": str(DATA_DIR),
    }


# ── samples ───────────────────────────────────────────────────────
@app.get("/samples")
def get_samples():
    sr = store.sr
    return {"sr": sr, "sample_length": C.sample_length(sr) if sr else None, "counts": store.counts()}


@app.post("/samples/{key}")
async def add_samples(key: str, request: Request, sr: int = Query(..., gt=0)):
    """Body: one clip, or several clips concatenated (length a multiple of the clip length)."""
    _check_key(key)
    pcm = await _read_pcm(request)
    L = C.sample_length(sr)
    if len(pcm) % L:
        raise HTTPException(400, f"body has {len(pcm)} samples, not a multiple of the clip length {L}")
    try:
        count = await run_in_threadpool(store.add, key, sr, pcm.reshape(-1, L))
    except StoreError as e:
        raise _http(e)
    return {"key": key, "count": count}


@app.delete("/samples/{key}")
def delete_key_samples(key: str):
    _check_key(key)
    trainer.stop()   # a run in progress was started on data that is now gone
    return {"deleted": store.delete(key)}


@app.delete("/samples")
def delete_all_samples():
    trainer.stop()
    store.clear()
    return {"deleted": True}


# ── training ──────────────────────────────────────────────────────
@app.post("/train", status_code=202)
def start_training(req: TrainRequest | None = Body(default=None)):
    req = req or TrainRequest()
    try:
        return trainer.start(C.TrainParams(**req.model_dump()))
    except (StoreError, TrainingBusy) as e:
        raise _http(e)


@app.post("/train/stop")
def stop_training():
    return trainer.stop()


@app.get("/train/status")
def training_status(since: int = Query(0, ge=0)):
    """`since`: number of log lines already received; only newer lines are returned."""
    return trainer.status(since)


# ── model ─────────────────────────────────────────────────────────
@app.get("/model")
def get_model():
    try:
        return trainer.model_info()
    except NoModel as e:
        raise _http(e)


@app.delete("/model")
async def delete_model():
    await run_in_threadpool(trainer.delete_model)
    return {"deleted": True}


@app.post("/predict")
async def predict(request: Request, sr: int = Query(..., gt=0)):
    pcm = await _read_pcm(request)
    try:
        return await run_in_threadpool(trainer.predict, pcm, sr)
    except (StoreError, NoModel) as e:
        raise _http(e)
