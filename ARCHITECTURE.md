# KeyMic — Architecture

## Overview

KeyMic trains a Convolutional Neural Network (CNN) entirely in the browser
to classify keyboard keys from their acoustic fingerprint.  No data ever
leaves the machine; everything — capture, feature extraction, training, and
inference — runs client-side with the Web Audio API and TensorFlow.js.

---

## 1  Front-end: Mel Spectrogram

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
| Transform | Real FFT (`tf.spectral.rfft`) | Complex spectrum; power = `re² + im²` |

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

## 2  Back-end: Convolutional Neural Network

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
| Softmax output | Multi-class probability vector; argmax gives the predicted key |

### 2.3  Training

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam |
| Learning rate | 0.001 |
| Loss | `sparseCategoricalCrossentropy` |
| Metric | Accuracy |
| Epochs | 60 |
| Batch size | 32 |
| Validation split | 20 % per key (stratified) |

Minimum to start: **8 samples per key**, **2 distinct keys**.
Maximum stored per key: 50 (oldest dropped on overflow).

### 2.4  Persistence

| Artefact | Storage |
|----------|---------|
| Raw audio samples | IndexedDB via `localForage` (`ks-{key}` keys, ArrayBuffer) |
| Sample counts | IndexedDB (`ks-counts`, JSON) |
| Sample rate | IndexedDB (`ks-sr`) |
| Trained model weights | IndexedDB via `model.save('indexeddb://keystroke-cnn')` |
| Model key list | IndexedDB (`ks-model-keys`, JSON) |

Both samples and model survive page reloads and browser restarts.

---

## 3  Python backend (GPU training)

`backend/` re-implements the same pipeline in PyTorch so training runs on CUDA
instead of WebGL. Spectrogram construction mirrors §1 (the mel filterbank is
bit-identical to the JS one) and `KeystrokeCNN` mirrors the layer stack in §2.
Differences:

| Aspect | Browser (TF.js) | Backend (PyTorch) |
|--------|-----------------|-------------------|
| Spectrograms | per clip, during dataset build | whole dataset in one batched GPU pass |
| Output layer | softmax inside the model | logits; `CrossEntropyLoss` / softmax at prediction |
| Validation | stratified 20 % per key | same |
| Storage | IndexedDB | `backend/data/samples/*.npy` + `model.pt` |

Training runs in a background thread; the HTTP API exposes start / stop /
status / delete. Endpoints and setup: `backend/README.md`.

---

## 4  File structure

```
F:/dev/keymic/
├── index.html            HTML shell + CDN script tags
├── styles/
│   └── main.css          Dark-theme stylesheet
├── src/
│   ├── config.js         Shared constants and keyboard layout
│   ├── audio.js          AudioCapture — ring buffer, mic init, snapshot
│   ├── features.js       Mel filterbank (singleton) + computeMelSpec
│   ├── store.js          IndexedDB persistence layer (localForage wrapper)
│   ├── model.js          buildModel · trainFromStore · loadSavedModel · deleteSavedModel
│   ├── ui.js             All DOM helpers (badges, keyboard, charts, capture view)
│   ├── samples.js        Samples tab: paginated clip browser (waveform, spectrogram, play, delete)
│   └── app.js            Application orchestrator — state, event handlers
├── backend/              Python GPU backend (see backend/README.md)
└── ARCHITECTURE.md       This document
```

---

## 5  Running locally

```bash
cd F:/dev/keymic
python -m http.server 8080
# then open  http://localhost:8080
```

A local HTTP server is required for ES module imports and `getUserMedia`.
Opening `index.html` directly as a `file://` URL will block both.
