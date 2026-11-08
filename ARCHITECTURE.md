# KeyMic — Architecture

## Overview

KeyMic trains a Convolutional Neural Network (CNN) to classify keyboard keys
from their acoustic fingerprint. Two local processes, nothing leaves the machine:

| Part | Does |
|------|------|
| Browser (`index.html`, `src/`) | microphone capture, cutting each keystroke clip at its click (§1.1), sample browser, training progress UI, prediction UI |
| Backend (`backend/`, PyTorch) | sample storage, spectrograms (§1.2–1.6), CNN training on the GPU (§2), prediction |

```
keydown ─► browser: cut 200 ms clip at the click ─► POST /samples/{key}   (record)
                                                 └► POST /predict          (predict → key + probabilities)
Train tab ─► POST /train ─► poll GET /train/status every 400 ms (log lines, per-epoch loss/acc) ─► GET /model
```

The browser still computes spectrograms with TensorFlow.js, only to draw the
previews in the Samples tab; the JS and Python versions produce the same
features (identical filterbank, same normalisation).

---

## 1  Feature pipeline: keystroke clip → log-mel spectrogram

Raw audio → 2-D image that the CNN can learn from.

### 1.1  Signal capture

| Component | Detail |
|-----------|--------|
| API | `navigator.mediaDevices.getUserMedia` |
| Processing flags | `echoCancellation: false`, `noiseSuppression: false`, `autoGainControl: false` — raw signal only |
| Ring buffer | `Float32Array`, depth 4 s, written by a `ScriptProcessorNode` (chosen over `AudioWorklet` to avoid the HTTPS requirement) |
| Keystroke cut | see §1.1.1 |

#### 1.1.1  Cutting the keystroke clip

Audio reaches the ring buffer later than the key event (processing chunk of
2048 frames + device input latency), so the clip can't simply be "the last
200 ms" at keydown — that window would end before the click arrives.
`AudioCapture.captureKeystroke` instead:

1. maps the event's `timeStamp` to an absolute stream frame (frames written so
   far + time since the last chunk);
2. waits until the audio after the event has arrived;
3. high-passes −250…+400 ms of audio around the event (4th-order
   Butterworth, 800 Hz) so fans, hum and rumble don't mask the click, and
   computes a short-time energy envelope (5 ms windows, 1 ms hop) in dB above
   the background (20th percentile of the envelope);
4. takes the envelope peak inside the search window as the click. Before
   calibration the window is −100…+300 ms; once 5 clicks were accepted it
   narrows to ±80 ms around the **measured latency** (median event→click
   offset of the last 20 clicks). If the narrow window finds nothing, the wide
   one is searched too, so a wrong calibration corrects itself;
   the press is rejected if the peak is below the threshold (default 6 dB,
   adjustable on the Record tab, which also plots the envelope of the last press);
5. returns 200 ms starting 20 ms before the peak, so every clip has its click
   at the same position.

### 1.2  STFT

Each 200 ms window is framed with:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Frame length | 512 samples | ≈ 11 ms at 44 100 Hz — shorter than a keystroke |
| Hop length | 128 samples | 75 % overlap for temporal smoothness |
| Window function | Hann | Reduces spectral leakage at frame boundaries |
| Transform | Real FFT (`torch.fft.rfft`; `tf.spectral.rfft` for browser previews) | Complex spectrum; power = `re² + im²` |

### 1.3  Mel filterbank

The power spectrum (257 bins) is projected onto **40 triangular mel-scale
filters spanning 300 Hz – 16 kHz** (`buildFilterbank`) represented as a
`[257 × 40]` matrix. Below 300 Hz is rumble and mains hum; above 16 kHz is
mostly microphone hiss. Restricting the range keeps that noise out of the
features and out of the per-spectrogram normalisation. Stored clips are
unfiltered, so the range can change without re-recording (but requires
retraining).
Matrix-multiplication (`tf.matMul`) produces one mel-energy vector per frame.

Mel scale formula:
```
mel(f) = 2595 · log₁₀(1 + f / 700)
```

### 1.4  Log compression

```
logMel = log(melPower + 10⁻⁶)
```

The log compresses the wide dynamic range of acoustic energy; the small
offset avoids `log(0)` in silent bins.

### 1.5  Per-spectrogram normalisation

```
z = (logMel − mean) / (std + ε)
```

Z-score computed per spectrogram (not across the dataset) so that absolute
loudness level and room noise floor are factored out.  This is essential for
generalising across recording sessions.

### 1.6  Temporal alignment

After normalisation, the time axis is padded with zeros or cropped to exactly
**64 frames** (a 200 ms clip yields 72 frames at 48 kHz, so it is cropped).
This fixes the CNN input shape.

**Final input tensor shape: `[64, 40, 1]`** (frames × mel-bands × channel).

---

## 2  Convolutional Neural Network (backend, PyTorch)

### 2.1  Architecture

```
Input          [64 × 40 × 1]

Conv Block 1
  Conv2D        32 filters, 3×3, same padding
  BatchNorm
  ReLU
  MaxPool2D     2×2 → [32 × 20 × 32]

Conv Block 2
  Conv2D        64 filters, 3×3, same padding
  BatchNorm
  ReLU
  MaxPool2D     2×2 → [16 × 10 × 64]

Conv Block 3
  Conv2D        128 filters, 3×3, same padding
  BatchNorm
  ReLU
  MaxPool2D     2×2 → [8 × 5 × 128]

GlobalAveragePooling2D  → [128]

Dense           128 units, ReLU
Dropout         0.4

Dense           N classes, Softmax
```

### 2.2  Design choices

| Choice | Reason |
|--------|--------|
| Three conv blocks | Enough depth to capture local spectral-temporal patterns without overfitting on small datasets |
| Batch normalisation | Stabilises training and allows higher learning rates |
| GlobalAveragePooling instead of Flatten | Reduces parameter count; more robust to slight temporal misalignment |
| Dropout 0.4 | Main regulariser — the dataset is typically < 500 samples total |
| Softmax output | Multi-class probability vector; argmax gives the predicted key (applied at prediction time; training uses logits with `CrossEntropyLoss`) |
| BatchNorm momentum 0.1 (PyTorch default) | Running statistics used at inference converge within the few hundred steps a small dataset gives |

### 2.3  Training

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam |
| Learning rate | 0.001 |
| Loss | `CrossEntropyLoss` |
| Metric | Accuracy |
| Epochs | 60 (adjustable in the Train tab) |
| Batch size | 32 |
| Validation split | 20 % per key (stratified) |
| Weight decay | 1e-4 (Adam L2 penalty) |

Minimum to start: **8 samples per key**, **2 distinct keys**.
Maximum stored per key: 50 (oldest dropped on overflow).

**Training-time augmentation.** Each batch's raw clips are perturbed fresh before the
spectrogram is computed — a different random perturbation every epoch:

| Augmentation | Range |
|---|---|
| Gain | ×0.85–1.15 |
| Added noise | up to 6 % of the clip's own RMS |
| Time shift | ± 6 ms, zero-padded |

The validation set is never augmented, so `val_acc` reflects the model's accuracy on
clean, unperturbed audio. Without this, a small per-key sample count and subtle
inter-key acoustic differences let the network memorise incidental details of one
recording session (exact noise floor, exact timing) instead of the keystroke's actual
acoustic signature — it looks fine on a validation split of that same session and then
performs at chance on new keystrokes.

A true-key × predicted-key confusion matrix over the final validation set is logged at
the end of training (and returned from `/model` as `info.confusion`), since the
aggregate `val_acc` number can't distinguish "genuinely random across all keys" from
"biased toward one key" or "two specific keys keep getting swapped."

### 2.4  Persistence (backend)

| Artefact | Storage |
|----------|---------|
| Raw audio clips | `backend/data/samples/key_<hex>.npy` — float32 `[n, clip_length]` per key |
| Sample rate | `backend/data/meta.json` |
| Trained model | `backend/data/model.pt` — weights, key list, sample rate, metrics |

Training runs in a background thread on the GPU (spectrograms of the whole
dataset in one batched pass). The API exposes start / stop / status / delete;
a stopped run saves nothing. Endpoints and setup: `backend/README.md`.

Samples recorded in the browser before the backend existed (IndexedDB) are
offered for one-click upload on the Record tab (`src/legacy-store.js`).

---

## 3  File structure

```
F:/dev/keymic/
├── index.html            HTML shell + CDN script tags
├── styles/
│   └── main.css          Dark-theme stylesheet
├── src/
│   ├── config.js         Shared constants and keyboard layout
│   ├── audio.js          AudioCapture — ring buffer, click detection, latency calibration
│   ├── api.js            Backend client (samples, training, model, predict)
│   ├── features.js       Mel spectrogram in TF.js (Samples tab previews)
│   ├── legacy-store.js   One-time upload of samples kept in IndexedDB by older versions
│   ├── ui.js             All DOM helpers (badges, keyboard, charts, capture view)
│   ├── samples.js        Samples tab: paginated clip browser (waveform, spectrogram, play, delete)
│   └── app.js            Application orchestrator — state, event handlers
├── backend/              Python GPU backend (see backend/README.md)
└── ARCHITECTURE.md       This document
```

---

## 4  Running locally

See the root `README.md`: start the backend (`python run.py` in `backend/`),
then serve the frontend (`python -m http.server 8080`) and open http://localhost:8080.
