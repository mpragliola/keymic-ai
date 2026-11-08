# KeyMic

Learns to recognise which keyboard key was pressed from its sound alone. You
record labelled keystrokes through the microphone, train a CNN on their log-mel
spectrograms, then type and compare the model's guess against the real key.

- **Frontend** (repo root): plain HTML + ES modules + TensorFlow.js. Records,
  trains and predicts entirely in the browser; data persists in IndexedDB.
- **Backend** (`backend/`): Python + PyTorch server that trains on an NVIDIA GPU.
  Not yet wired to the frontend.

## Run the frontend

```bash
python -m http.server 8080
# open http://localhost:8080
```

A local server is required (ES modules and microphone access don't work on `file://`).

1. **Record** — enable the mic and press each key you want learned (≥ 8 times; 15–30 is better).
2. **Samples** — browse each key's clips (waveform, spectrogram, playback), delete bad ones.
3. **Train** — start, stop at any time, or delete the model.
4. **Predict** — type; the guess, the real key and the top confidences are shown.

## Run the backend

See [backend/README.md](backend/README.md).

How the model works: [ARCHITECTURE.md](ARCHITECTURE.md).
