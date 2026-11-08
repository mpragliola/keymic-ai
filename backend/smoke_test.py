"""End-to-end check of the backend with synthetic keystrokes (no microphone needed).

Each fake "key" is a short tone burst at its own frequency plus noise, so a
working pipeline should learn to separate them almost perfectly.

    python smoke_test.py

Uses a temporary data directory; your real samples and model are not touched.
"""
import os
import tempfile
import time

os.environ["KEYMIC_DATA"] = tempfile.mkdtemp(prefix="keymic-smoke-")

import numpy as np                                # noqa: E402
from fastapi.testclient import TestClient         # noqa: E402

from keymic import config as C                    # noqa: E402
from keymic.api import app, DEVICE                # noqa: E402

SR = 48000
L = C.sample_length(SR)
FREQS = {"a": 300.0, "s": 1200.0, "d": 3000.0}
rng = np.random.default_rng(0)


def clip(freq: float) -> np.ndarray:
    """200 ms: a 120 ms tone burst starting at a random offset, plus background noise."""
    t = np.arange(L) / SR
    start = rng.uniform(0.0, 0.05)
    env = ((t >= start) & (t < start + 0.12)).astype(np.float32)
    x = 0.4 * np.sin(2 * np.pi * freq * t) * env + rng.normal(0, 0.02, L)
    return x.astype("<f4")


def wait_for(client, states, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get("/train/status").json()
        if s["state"] in states:
            return s
        time.sleep(0.2)
    raise TimeoutError(f"training did not reach {states}: {s['state']}")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("PASS", msg)


with TestClient(app) as c:
    print("health:", c.get("/health").json())

    # Upload: one request with 12 concatenated clips per key, one single clip
    for k, f in FREQS.items():
        body = np.concatenate([clip(f) for _ in range(12)]).tobytes()
        r = c.post(f"/samples/{k}", params={"sr": SR}, content=body)
        check(r.status_code == 200 and r.json()["count"] == 12, f"bulk upload {k!r}")
    r = c.post("/samples/a", params={"sr": SR}, content=clip(FREQS["a"]).tobytes())
    check(r.json()["count"] == 13, "single-clip upload appends")
    check(c.post("/samples/a", params={"sr": 44100}, content=clip(300).tobytes()).status_code == 409,
          "mismatched sample rate rejected")
    check(c.post("/samples/a", params={"sr": SR}, content=b"\0" * 12).status_code == 400,
          "wrong clip length rejected")
    check(c.get("/samples").json()["counts"] == {"a": 13, "d": 12, "s": 12}, "counts")

    # Full training run
    r = c.post("/train", json={"epochs": 30})
    check(r.status_code == 202, "training started")
    check(c.post("/train").status_code == 409, "second concurrent run rejected")
    s = wait_for(c, {"done", "error"})
    print("\n".join(s["logs"][:6] + s["logs"][-2:]))
    check(s["state"] == "done", f"training finished ({s['error']})")
    check(s["history"]["val_acc"][-1] >= 0.8, f"val_acc {s['history']['val_acc'][-1]:.2f} ≥ 0.8")
    info = c.get("/model").json()
    check(info["keys"] == ["a", "d", "s"], f"model info: {info}")

    # Predictions on fresh clips
    hits = 0
    for k, f in FREQS.items():
        for _ in range(5):
            p = c.post("/predict", params={"sr": SR}, content=clip(f).tobytes()).json()
            hits += p["key"] == k
    check(hits >= 13, f"predictions on fresh clips: {hits}/15 correct")

    # Stop: a long run is cancelled and the earlier model survives
    c.post("/train", json={"epochs": 2000})
    wait_for(c, {"running"})
    time.sleep(1)
    c.post("/train/stop")
    s = wait_for(c, {"stopped", "done", "error"})
    check(s["state"] == "stopped", "stop cancels the run")
    check(c.get("/model").status_code == 200, "previous model kept after a stopped run")

    # Delete model, then samples
    check(c.delete("/model").status_code == 200 and c.get("/model").status_code == 404, "model deleted")
    check(c.post("/predict", params={"sr": SR}, content=clip(300).tobytes()).status_code == 404,
          "predict without model → 404")
    c.delete("/samples/s")
    check("s" not in c.get("/samples").json()["counts"], "single key deleted")
    c.delete("/samples")
    check(c.get("/samples").json() == {"sr": None, "sample_length": None, "counts": {}}, "all samples deleted")

print(f"\nAll checks passed on {DEVICE}.")
