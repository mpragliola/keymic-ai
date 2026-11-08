// Shared configuration constants and keyboard layout.

export const CFG = {
  // Audio capture
  CAPTURE_MS:  200,   // window length to extract per keystroke (ms)
  RING_S:        4,   // ring buffer depth (seconds)

  // Feature extraction
  N_MELS:       40,   // mel filterbank bands
  FRAME_LEN:   512,   // STFT frame size (samples)
  HOP_LEN:     128,   // STFT hop size (samples)
  N_FRAMES:     64,   // fixed time-axis length fed to the CNN

  // Training
  MAX_SAMPLES:  50,   // hard cap per key
  MIN_SAMPLES:   8,   // minimum samples for a key to enter training
  MIN_KEYS:      2,   // minimum distinct keys to allow training
  EPOCHS:       60,
  BATCH_SIZE:   32,
  VAL_SPLIT:   0.2,
  LR:         0.001,
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
