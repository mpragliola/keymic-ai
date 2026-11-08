// Shared configuration constants and keyboard layout.

export const CFG = {
  // Backend
  BACKEND_URL: 'http://127.0.0.1:8000',
  POLL_MS:     400,   // training status poll interval while a run is active
  DEFAULT_EPOCHS: 60,

  // Audio capture
  CAPTURE_MS:  200,   // clip length per keystroke (ms)
  PRE_ONSET_MS: 20,   // clip starts this long before the detected click
  ANALYSIS_RANGE_MS: [-250, 400], // audio analysed around each key event (and shown in the capture view)
  SEARCH_WINDOW_MS: [-100, 300],  // click search window around the key event, before calibration
  CALIBRATED_HALF_WINDOW_MS: 80, // after calibration: ± this around the measured latency
  CALIBRATION_STROKES: 5,        // accepted clicks needed before latency counts as measured
  CALIBRATION_HISTORY: 20,       // latency = median offset of the last N clicks
  MIN_PEAK_DB: 6,     // default click threshold (dB over background); adjustable in the UI
  BACKGROUND_PERCENTILE: 20, // background level = this percentile of the energy envelope
  DETECT_HIGHPASS_HZ: 800,   // click detection runs on high-passed audio (removes fan/hum/rumble)
  RING_S:        4,   // ring buffer depth (seconds)

  // Blind click detection (no keyboard event — "type via mic" mode)
  BLIND_HOP_MS:        15,  // how often the stream is scanned for a new click
  BLIND_BACKGROUND_MS: 400, // trailing window used to estimate the noise floor
  BLIND_REFRACTORY_MS: 150, // minimum spacing between two accepted clicks
  BLIND_PEAK_SEARCH_MS: 30, // once a threshold crossing fires, look this far ahead for the true peak

  // Feature extraction
  N_MELS:       40,   // mel filterbank bands
  FMIN_HZ:     300,   // mel filterbank range: below = rumble/hum, above = mostly mic hiss
  FMAX_HZ:   16000,
  FRAME_LEN:   512,   // STFT frame size (samples)
  HOP_LEN:     128,   // STFT hop size (samples)
  N_FRAMES:     64,   // fixed time-axis length fed to the CNN

  // Training (backend enforces these; kept here for UI hints)
  MAX_SAMPLES:  50,
  MIN_SAMPLES:   8,
  MIN_KEYS:      2,
};

export const KEYBOARD_ROWS = [
  ['1','2','3','4','5','6','7','8','9','0'],
  ['q','w','e','r','t','y','u','i','o','p'],
  ['a','s','d','f','g','h','j','k','l'],
  ['z','x','c','v','b','n','m'],
  [' '],
];

// All keys we track (flat set for O(1) lookup)
export const TRACKED = new Set(KEYBOARD_ROWS.flat());

// Visual label for a key
export const displayKey = k => k === ' ' ? 'SPC' : k.toUpperCase();
