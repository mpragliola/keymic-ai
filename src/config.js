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
  PEAK_SEPARATION_MS: 15,        // envelope peaks closer than this count as one click
  MIN_PEAK_DB: 6,     // default click threshold (dB over background); adjustable in the UI
  BACKGROUND_PERCENTILE: 20, // background level = this percentile of the energy envelope
  DETECT_HIGHPASS_HZ: 800,   // click detection runs on high-passed audio (removes fan/hum/rumble)
  RING_S:        4,   // ring buffer depth (seconds)

  // Blind click detection (no keyboard event — "type via mic" mode)
  BLIND_HOP_MS:        15,  // how often the stream is scanned for a new click
  BLIND_BACKGROUND_MS: 400, // trailing window used to estimate the noise floor
  BLIND_REFRACTORY_MS:  50, // minimum spacing between two accepted clicks — short enough for
                            // fast typing. Key releases get detected too; the model drops them
                            // as non-keystroke sounds (recorded on keyup, see recordRelease in app.js)
  BLIND_PEAK_SEARCH_MS: 30, // once a threshold crossing fires, look this far ahead for the true peak

  // Samples-tab spectrogram previews only — the model's features are set in backend/keymic/config.py
  N_MELS:       40,   // mel filterbank bands
  FMIN_HZ:     300,   // mel filterbank range: below = rumble/hum, above = mostly mic hiss
  FMAX_HZ:   16000,
  FRAME_LEN:   512,   // STFT frame size (samples)
  HOP_LEN:     128,   // STFT hop size (samples)
  N_FRAMES:     64,   // time-axis length of the preview image

  // Training (backend enforces these; kept here for UI hints)
  MAX_SAMPLES:  1000,
  MIN_SAMPLES:   8,
  MIN_KEYS:      2,
};

// ─── Keys ────────────────────────────────────────────────────────────────────
// Keys are physical positions (KeyboardEvent.code), so Shift+1 is still the "1" key.
// Each is stored under a single character: the key's unshifted character in the user's
// layout (navigator.keyboard.getLayoutMap, where available) — the same character the
// old e.key-based recording produced for letters and digits, so existing samples keep
// matching. Keys without a character, and characters that can't sit in a URL path
// segment ('/' and '.'), use private-use characters instead.

// Standard ISO/ANSI main block, row by row. Codes a keyboard doesn't have (e.g.
// IntlBackslash on ANSI) are dropped at runtime if their character duplicates another key.
export const KEY_CODE_ROWS = [
  ['Backquote','Digit1','Digit2','Digit3','Digit4','Digit5','Digit6','Digit7','Digit8','Digit9','Digit0','Minus','Equal','Backspace'],
  ['Tab','KeyQ','KeyW','KeyE','KeyR','KeyT','KeyY','KeyU','KeyI','KeyO','KeyP','BracketLeft','BracketRight','Backslash'],
  ['CapsLock','KeyA','KeyS','KeyD','KeyF','KeyG','KeyH','KeyJ','KeyK','KeyL','Semicolon','Quote','Enter'],
  ['ShiftLeft','IntlBackslash','KeyZ','KeyX','KeyC','KeyV','KeyB','KeyN','KeyM','Comma','Period','Slash','ShiftRight'],
  ['Space'],
];

// Reserved "this is not a keystroke" class (mouse clicks, coughs, key releases). Not
// produced by any key — see keyChar below.
export const NOISE_KEY = '~';

// Keys without a typed character → private-use stand-ins, with label and typed text
const NAMED = {
  Backspace:  { ch: '', label: '⌫',     text: '\b' },
  Tab:        { ch: '', label: 'TAB',   text: '\t' },
  CapsLock:   { ch: '', label: 'CAPS',  text: '' },
  Enter:      { ch: '', label: '⏎',     text: '\n' },
  ShiftLeft:  { ch: '', label: '⇧',     text: '' },
  ShiftRight: { ch: '', label: '⇧R',    text: '' },
};
export const isNamedKey = k => Object.values(NAMED).some(n => n.ch === k);

// Characters that would break the /samples/{key} URL path segment, and NOISE_KEY's '~'
const ESCAPED = { '/': '', '.': '', '~': '' };

// US-layout fallback where navigator.keyboard.getLayoutMap isn't available (Firefox, Safari)
const US = { Backquote:'`', Minus:'-', Equal:'=', BracketLeft:'[', BracketRight:']', Backslash:'\\',
             Semicolon:';', Quote:"'", Comma:',', Period:'.', Slash:'/', IntlBackslash:'\\', Space:' ' };
for (let i = 0; i < 26; i++) US['Key' + String.fromCharCode(65 + i)] = String.fromCharCode(97 + i);
for (let i = 0; i < 10; i++) US['Digit' + i] = String(i);

let layout = null;   // Map code → unshifted character, once loaded
export async function loadKeyboardLayout() {
  try { layout = await navigator.keyboard.getLayoutMap(); } catch { layout = null; }
}

// Storage character for a physical key code, or null if it isn't one we track
export function keyChar(code) {
  if (NAMED[code]) return NAMED[code].ch;
  const c = (code === 'Space' ? ' ' : (layout?.get(code) ?? US[code]))?.toLowerCase();
  if (c === undefined || [...c].length !== 1) return null;
  return ESCAPED[c] ?? c;
}

// The keyboard actually shown and tracked: KEY_CODE_ROWS as storage characters, with any
// key whose character duplicates an earlier one dropped. Call after loadKeyboardLayout.
export function keyboardRows() {
  const seen = new Set();
  return KEY_CODE_ROWS.map(row => row.map(keyChar).filter(ch => ch !== null && !seen.has(ch) && seen.add(ch)));
}

const LABEL = new Map([...Object.values(NAMED).map(n => [n.ch, n.label]),
                       ...Object.entries(ESCAPED).map(([c, ch]) => [ch, c])]);
const TEXT  = new Map([...Object.values(NAMED).map(n => [n.ch, n.text]),
                       ...Object.entries(ESCAPED).map(([c, ch]) => [ch, c])]);

// Visual label for a stored key
export const displayKey = k => k === ' ' ? 'SPC' : k === NOISE_KEY ? 'NOISE' : LABEL.get(k) ?? k.toUpperCase();

// What typing this key produces in a text box ('\b' = delete the previous character)
export const keyText = k => TEXT.get(k) ?? k;

// For inserting labels into innerHTML
export const escapeHtml = s => s.replace(/[&<>"']/g, c => `&#${c.charCodeAt(0)};`);
