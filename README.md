# KeyMic

Learns to recognise which keyboard key was pressed from its sound alone. You
record labelled keystrokes through the microphone, train a CNN on their log-mel
spectrograms, then type and compare the model's guess against the real key.

- **Frontend** (repo root): plain HTML + ES modules. Records keystrokes from the
  microphone, cuts each clip at the detected click, and shows samples, training
  progress and predictions.
- **Backend** (`backend/`): Python + PyTorch server. Stores the samples, trains
  the CNN on the GPU and runs predictions.

## Run

1. **Backend** — first-time setup in [backend/README.md](backend/README.md), then:
   ```powershell
   cd backend
   .venv\Scripts\activate
   python run.py          # http://127.0.0.1:8000
   ```
2. **Frontend** — in the repo root:
   ```bash
   python -m http.server 8080
   # open http://localhost:8080
   ```
   A local server is required (ES modules and microphone access don't work on `file://`).
   The header shows the backend's GPU once connected; a red banner means it isn't running.

## Use

1. **Record** — enable the mic and press each key you want learned (≥ 8 times; 20–30 is better).
   The "Last press" chart shows where each click was found; red keys mean the press was ignored.
2. **Samples** — browse each key's clips (waveform, spectrogram, playback), delete bad ones.
3. **Train** — runs on the GPU with live log and accuracy chart; stop at any time, or delete the model.
4. **Predict** — type; the guess, the real key and the top confidences are shown.

How the model works: [ARCHITECTURE.md](ARCHITECTURE.md).
