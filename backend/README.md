# KeyMic backend

Python server that stores keystroke recordings, trains the keystroke CNN on
your NVIDIA GPU (PyTorch + CUDA), and serves predictions (see `../ARCHITECTURE.md`).

The browser frontend (repo root) uses this API for everything except audio capture.

## Setup (Windows, NVIDIA GPU)

```powershell
cd F:\dev\keymic\backend
python -m venv .venv
.venv\Scripts\activate

# PyTorch with CUDA — pick the command for your driver at https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Python 3.10+ required.

## Run

```powershell
python smoke_test.py   # end-to-end check with synthetic keystrokes, temp data dir
python run.py          # serves http://127.0.0.1:8000  (docs: http://127.0.0.1:8000/docs)
```

| Env var        | Default                                       | Meaning                        |
|----------------|-----------------------------------------------|--------------------------------|
| `KEYMIC_DATA`  | `backend/data`                                | samples + `model.pt`           |
| `KEYMIC_PORT`  | `8000`                                        | port                           |
| `KEYMIC_HOST`  | `127.0.0.1`                                   | bind address (localhost only)  |
| `KEYMIC_CORS`  | `http://localhost:8080,http://127.0.0.1:8080` | allowed frontend origins       |

## API

Audio is sent as **raw little-endian float32 PCM** in the request body (the
bytes of a browser `Float32Array`), with the sample rate as `?sr=`. Each clip is
`round(0.2 × sr)` samples (9600 at 48 kHz).

| Method   | Path                       | Body / params                                  | Result |
|----------|----------------------------|------------------------------------------------|--------|
| `GET`    | `/health`                  |                                                | device, GPU name, torch/CUDA versions |
| `GET`    | `/samples`                 |                                                | `{sr, sample_length, counts}` |
| `POST`   | `/samples/{key}?sr=`       | one clip, or N clips concatenated              | `{key, count}` — keeps the newest 1000 per key |
| `GET`    | `/samples/{key}`           |                                                | that key's clips as raw float32, oldest → newest; headers `X-Sample-Rate`, `X-Clip-Length`, `X-Count` (empty body if none) |
| `DELETE` | `/samples/{key}/{index}`   |                                                | deletes one clip (0 = oldest) → `{key, count}`; `404` if out of range |
| `DELETE` | `/samples/{key}`           |                                                | deletes one key's samples |
| `DELETE` | `/samples`                 |                                                | deletes all samples |
| `POST`   | `/train`                   | optional JSON `{epochs, batch_size, lr, val_split}` | `202`, starts a background run |
| `POST`   | `/train/stop`              |                                                | stops at the next batch; nothing is saved |
| `GET`    | `/train/status?since=N`    |                                                | `{state, epoch, epochs, keys, history, logs, log_total, error}` |
| `GET`    | `/model`                   |                                                | keys, sample rate, val_acc, training time; `404` if none |
| `DELETE` | `/model`                   |                                                | stops a running training, deletes the model |
| `POST`   | `/predict?sr=`             | one clip                                       | `{key, probs: [{key, p}, …]}` sorted by `p` |

`state` is one of `idle`, `running`, `stopping`, `done`, `stopped`, `error`.
Poll `/train/status?since=<log_total from last poll>` to receive only new log lines.

Errors: `400` bad input or not enough data · `404` no model · `409` sample-rate
mismatch or a run already active.

Rules worth knowing:
- All samples share one sample rate. Uploading at a different rate returns `409`;
  delete all samples to switch microphones/rates.
- Deleting samples stops a running training. Deleting the model stops it too, so a
  run can never re-save a model you just deleted.
- A stopped run is discarded; the previously saved model stays.

## Layout

```
backend/
├── run.py               starts uvicorn
├── smoke_test.py        end-to-end self-test
├── requirements.txt
└── keymic/
    ├── config.py        constants shared with src/config.js
    ├── features.py      log-mel spectrogram on the GPU (mirrors src/features.js)
    ├── model.py         KeystrokeCNN (mirrors src/model.js)
    ├── store.py         samples on disk: data/samples/key_<hex>.npy + meta.json
    ├── trainer.py       background training thread, stop, checkpointing, inference
    └── api.py           FastAPI routes
```
