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
5. refines that peak: a 5 ms energy window scores highest once it fully
   contains the click's decay tail, which lands a few ms *after* the true
   onset. `peakAmplitudeFrame` rescans the few ms around the coarse window
   for the actual highest-amplitude sample and uses that instead — this is
   what fixed the systematic "peak always comes a bit late" bias visible in
   the Samples tab;
6. returns 200 ms starting 20 ms before the refined peak, so every clip has
   its click at the same position.

The blind scanner (`AudioCapture.startBlindScan`, §1.1.3) applies the same
`peakAmplitudeFrame` refinement to its own coarse hits.

#### 1.1.2  Backend-side onset re-detection and the 100 ms feature window

A keystroke has two acoustic events: the press and the release. The release's
timing depends on how long the key is held, and in fast typing it can blur
into the *next* key's press — so only the press should feed the model.
`FEATURE_MS = 100` (`backend/keymic/config.py`) crops the input to a leading
100 ms window, well clear of the release.

Rather than trust where the browser's capture-time cut placed the click, the
backend re-detects it directly from the stored raw clip, identically at train
and predict time (`backend/keymic/onset.py`):

1. high-pass filter (800 Hz, same cascaded-biquad design as the frontend);
2. short-time energy envelope, background-percentile noise floor;
3. the **earliest** point `ONSET_THRESHOLD_DB` (6 dB) above that background,
   within the first `ONSET_SEARCH_MS` (100 ms) — earliest rather than loudest,
   so a louder release (if the key was held a while) isn't mistaken for the
   press;
4. amplitude-refined to the exact peak sample, as in §1.1.1.

The clip is then re-cropped to start `PRE_ONSET_MS` (20 ms) before that point
and run through the usual 100 ms feature window. Every stored clip is still
the full 200 ms `CAPTURE_MS` of raw audio, so this re-detection — and
therefore the correction — applies to clips recorded before this existed,
with no need to re-record anything.

#### 1.1.3  Mic-only typing (blind scan)

The Type tab classifies keystrokes with no keyboard events at all —
`AudioCapture.startBlindScan` runs a continuous trailing-window energy scan
(hop, background window, and refractory period all in `src/config.js`,
`BLIND_*`) instead of the keydown-anchored capture in §1.1.1, with a
refractory period after each detected click so its own decay tail isn't
mistaken for a second click. Each detected click is sent to `/predict` the
same way as the Predict tab.

Because every loud sound in range gets classified as *something* — the
model is a closed-set softmax over trained keys — a reserved sentinel class,
`NOISE_KEY = '~'` (`src/config.js`), covers mouse clicks, coughs, and other
non-keystroke transients. The Record tab's "Start capturing" button reuses
this same blind scanner (pointed at storage instead of `/predict`): once
started, every transient it detects above the threshold is stored under
`NOISE_KEY` automatically until "Stop capturing" is pressed — no per-sound
button press. Only one blind scan runs at a time, so starting this disables
the Type tab's listening toggle and vice versa. `NOISE_KEY` is stored and
trained like any other key (a single character, so it needs no
special-casing in `backend/keymic/store.py`'s hex-encoded filenames); the
Type tab just drops a prediction instead of appending it when the predicted
key is `NOISE_KEY`.

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
**32 frames**, fixing the CNN input shape. A `FEATURE_MS = 100` clip (§1.1.2) yields
~31 real STFT frames at 44.1 kHz or ~34 at 48 kHz — `N_FRAMES` is set to 32 so padding
never amounts to more than a frame or two either way. (It was 64 before `FEATURE_MS`
existed, sized for the then-200ms window's ~72 frames; left at 64 after the crop to
100ms, it would have padded roughly *half* the input with zeros.)

**Final input tensor shape: `[32, 40, 1]`** (frames × mel-bands × channel).

---

## 2  Convolutional Neural Network (backend, PyTorch)

### 2.1  Architecture

```
Input          [32 × 40 × 1]

Conv Block 1
  Conv2D        16 filters, 3×3, same padding
  BatchNorm · ReLU
  Conv2D        16 filters, 3×3, same padding   (depth, not a second pooling stage)
  BatchNorm · ReLU
  MaxPool2D     2×2 → [16 × 20 × 16]
  Dropout2D     0.1

Conv Block 2
  Conv2D        32 filters, 3×3, same padding
  BatchNorm · ReLU
  Conv2D        32 filters, 3×3, same padding
  BatchNorm · ReLU
  MaxPool2D     2×2 → [8 × 10 × 32]
  Dropout2D     0.15

Conv Block 3
  Conv2D        64 filters, 3×3, same padding
  BatchNorm · ReLU
  Conv2D        64 filters, 3×3, same padding
  BatchNorm · ReLU
  MaxPool2D     2×2 → [4 × 5 × 64]
  Dropout2D     0.2

GlobalAveragePooling2D  → [64]

Dense           32 units, ReLU
Dropout         0.4

Dense           N classes, Softmax
```

A fourth pooling stage was deliberately not added — at a 32-frame input, three 2× poolings
already leave only 4 frames; a fourth would leave 2, too little spatial extent for a 3×3
kernel to mean much. The extra depth instead comes from a second 3×3 conv *within* each
block (c_out → c_out, no further downsampling) — roughly 3x the conv parameters of the
single-conv version this replaced, still small in absolute terms.

### 2.2  Design choices

| Choice | Reason |
|--------|--------|
| Three conv blocks, 16→32→64 filters, two convs each | Deliberately small — a per-user dataset is tens of clips per key, not thousands; a wider net (the original 32→64→128, one conv each) memorizes the training set instead of generalizing. If a larger dataset (more keys, the noise class, more samples per key) starts showing §2.3's overfitting signature again, drop back to one conv per block before adding dropout or data |
| Batch normalisation | Stabilises training and allows higher learning rates |
| Dropout2D after every conv block | Drops whole feature maps during training, not individual pixels — regularises the conv stack itself, not just the dense head |
| GlobalAveragePooling instead of Flatten | Reduces parameter count; more robust to slight temporal misalignment |
| Dropout 0.4 in the head | Additional regulariser on the final dense layer |
| Softmax output | Multi-class probability vector; argmax gives the predicted key (applied at prediction time; training uses logits with `CrossEntropyLoss`) |
| BatchNorm momentum 0.1 (PyTorch default) | Running statistics used at inference converge within the few hundred steps a small dataset gives |
| Training-time augmentation + weight decay + early stopping | See §2.3 — the main defense against overfitting given how little data there is per key |

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
Maximum stored per key: 1000 (oldest dropped on overflow).

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

**Resuming.** By default every run starts from a freshly-initialized network — raising
the epoch count and training again restarts from scratch, it doesn't continue where the
last run left off. Checking "Resume from current model" (Train tab) warm-starts the new
network from the currently saved model's weights instead (`Trainer.start` clones its
`state_dict` before the training thread starts, `_train` loads it in place of the random
init). This only works if the key set is unchanged since that model was saved — rejected
with a 400 otherwise, since the output layer's size depends on the number of keys. The
optimizer's own state (Adam's momentum) isn't carried over, and each run draws a fresh
random train/validation split, so early stopping's `best_val_loss` always starts over at
epoch 1 of the new run rather than continuing a curve from the old one. The key-set check
only catches a changed *key set* — a model saved under a since-edited `KeystrokeCNN`
(different layer sizes, same keys) fails `load_state_dict` with a shape mismatch instead;
`_train` catches that `RuntimeError` and falls back to a fresh network with a logged
explanation rather than crashing the run.

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
