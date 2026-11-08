# KeyMic — Architecture

## Overview

KeyMic trains a small neural network to classify keyboard keys
from their acoustic fingerprint. Two local processes, nothing leaves the machine:

| Part | Does |
|------|------|
| Browser (`index.html`, `src/`) | microphone capture, cutting each keystroke clip at its click (§1.1), sample browser, training progress UI, prediction UI |
| Backend (`backend/`, PyTorch) | sample storage, spectrograms (§1.2–1.6), training on the GPU (§2), prediction |

```
keydown ─► browser: cut 200 ms clip at the click ─► POST /samples/{key}   (record)
                                                 └► POST /predict          (predict → key + probabilities)
Train tab ─► POST /train ─► poll GET /train/status every 400 ms (log lines, per-epoch loss/acc) ─► GET /model
```

The browser still computes spectrograms with TensorFlow.js, only to draw the
previews in the Samples tab, with its own display-oriented settings (§1.6).

---

## 1  Feature pipeline: keystroke clip → log-mel spectrogram

Raw audio → log-mel spectrogram → per-band statistics the classifier learns from.

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

The 100 ms feature window (§1.1.2) is framed with:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Frame length | 2048 samples | ≈ 43 ms at 48 kHz; frequency bins ≈ 23 Hz apart — fine enough to resolve the small resonance differences between keys |
| Hop length | 256 samples | 10–11 frames per 100 ms window |
| Window function | Hann | Reduces spectral leakage at frame boundaries |
| Transform | Real FFT (`torch.fft.rfft`) | Complex spectrum; power = `re² + im²` |

### 1.3  Mel filterbank

The power spectrum (1025 bins) is projected onto **128 triangular mel-scale
filters spanning 50 Hz – 16 kHz** (`mel_filterbank`), giving one 128-band
mel-energy vector per frame. Stored clips are unfiltered, so the range can change
without re-recording (but requires retraining).

Mel scale formula:
```
mel(f) = 2595 · log₁₀(1 + f / 700)
```

**Why these settings.** They were chosen by measuring on real recordings (37 keys,
~100 presses each), training a plain linear classifier on each key's earlier presses
and testing on its later ones:

| Features | Held-out accuracy |
|---|---|
| Previous: 512-sample FFT, 40 mels 300 Hz–16 kHz, full time layout | 53 % |
| 2048-sample FFT, 128 mels from 50 Hz, mean + std over time | 80 % |
| …plus per-spectrogram z-score (§1.5) | 83 % |

The 512-sample FFT's ~94 Hz bins, squeezed into 40 bands, blurred together exactly
the fine spectral differences that distinguish keys.

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
loudness — how hard the key was hit, how far the mic is — is factored out.

### 1.6  No fixed time length

The time axis is left at its natural length (10–11 frames). The model summarizes
each band over time itself (§2.1), so there is no padding to dilute those
statistics. **Model input: `[T, 128]`** (frames × mel bands).

The browser's TensorFlow.js spectrogram (`src/features.js`) keeps its own, finer-in-time
settings — it only draws the Samples tab previews and is not the model's input.

---

## 2  Classifier (backend, PyTorch)

### 2.1  Architecture

```
Input                [T × 128]  z-scored log-mel
Per band over time:  mean, std  → [256]
BatchNorm1d          (learned per-feature standardization)
Dropout              0.2
Dense                128 units, ReLU
Dropout              0.5
Dense                N classes   (softmax at prediction time)
```

### 2.2  Design choices

| Choice | Reason |
|--------|--------|
| Mean + std over time instead of a CNN over the spectrogram | Measured on real recordings: what distinguishes keys is their frequency fingerprint, not the exact time course of the click inside the window. Keeping the time layout mostly added sensitivity to onset jitter — pooling it away was worth ~25 points (§1.3's table). The spectrogram CNNs used before never reached usable live accuracy on these recordings |
| One small hidden layer | A hidden layer adds little over a linear classifier on this data (~81 % vs ~80–83 %), so the net is kept small; dropout and weight decay do the work against small per-key sample counts |
| BatchNorm1d on the input | A linear layer on unstandardized log-mel statistics trains poorly; BatchNorm's running statistics are saved with the weights, so the same standardization applies at prediction time |
| Softmax output | Multi-class probability vector; argmax gives the predicted key (training uses logits with `CrossEntropyLoss`) |
| Training-time augmentation + weight decay + early stopping | See §2.3 |

### 2.3  Training

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam |
| Learning rate | 0.001 |
| Loss | `CrossEntropyLoss` |
| Metric | Accuracy |
| Epochs | 60 (adjustable in the Train tab) |
| Batch size | 32 |
| Validation split | each key's most recent 20 % of presses |
| Weight decay | 1e-4 (Adam L2 penalty) |

Minimum to start: **8 samples per key**, **2 distinct keys**.
Maximum stored per key: 1000 (oldest dropped on overflow).

**Validation = most recent presses, not a random sample.** Keys are recorded in runs of
consecutive presses, so mic position, background noise and typing force drift *within*
each key's samples — on real recordings, a classifier could guess the key 3.5× above
chance from the background noise alone, with no keystroke in it. A random split lets
validation share that drift with training: on the same data and model, a random split
reported 89 % where testing on later presses gave 83 %. Validating on each key's latest
presses keeps `val_acc` closer to what live typing will see, and makes early stopping pick
the epoch that generalizes across that drift. The split is deterministic (it depends only
on recording order), so a resumed run validates on the same clips as the run before.

**Training-time augmentation.** Each batch's raw clips are perturbed fresh before the
spectrogram is computed — a different random perturbation every epoch:

| Augmentation | Range |
|---|---|
| Gain | ×0.85–1.15 |
| Added noise | up to 6 % of the clip's own RMS |
| Time shift | ± 6 ms, zero-padded |

The validation set is never augmented, so `val_acc` reflects the model's accuracy on
clean, unperturbed audio.

**Early stopping.** The checkpoint with the lowest validation loss is saved, not the last
epoch's.

A true-key × predicted-key confusion matrix over the final validation set is logged at
the end of training (and returned from `/model` as `info.confusion`), since the
aggregate `val_acc` number can't distinguish "genuinely random across all keys" from
"biased toward one key" or "two specific keys keep getting swapped."

**Resuming.** By default every run starts from a freshly-initialized network. Checking
"Resume from current model" (Train tab) warm-starts the new network from the currently
saved model's weights instead (`Trainer.start` clones its `state_dict` before the
training thread starts, `_train` loads it in place of the random init). This only works
if the key set is unchanged since that model was saved — rejected with a 400 otherwise,
since the output layer's size depends on the number of keys. Adam's own state isn't
carried over, and early stopping's `best_val_loss` starts over with the new run. A model
saved under a since-changed architecture (same keys, different layers — e.g. a
`KeystrokeCNN` checkpoint from before §2.2's switch) fails `load_state_dict`; `_train`
catches that and falls back to a fresh network with a logged explanation.

### 2.4  Persistence (backend)

| Artefact | Storage |
|----------|---------|
| Raw audio clips | `backend/data/samples/key_<hex>.npy` — float32 `[n, clip_length]` per key |
| Sample rate | `backend/data/meta.json` |
| Trained model | `backend/data/model.pt` — weights, key list, sample rate, metrics |

Training runs in a background thread on the GPU. The API exposes start / stop / status / delete;
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
