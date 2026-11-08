import { CFG } from './config.js?v=11';

// Continuously fills a ring buffer from the microphone and cuts keystroke
// clips aligned to the sound itself.
//
// Positions are absolute stream frame indices (frames written since init), so
// a moment like "keydown" can be pinned to the stream and read back later,
// once the audio for it has actually arrived. That matters because audio
// reaches the ring buffer with a delay (processing chunk + input latency):
// at keydown time, the keystroke sound is not in the buffer yet.
export class AudioCapture {
  constructor() {
    this.ctx       = null;
    this.analyser  = null;
    this.processor = null;
    this.ring      = null;   // Float32Array ring buffer
    this.written   = 0;      // total frames written since init
    this.waiters   = [];     // [{ frame, resolve }] pending waitUntil() calls
    this.lastChunkAt = 0;    // performance.now() of the most recent chunk
    this.offsets   = [];     // recent key-event → click offsets (ms), for calibration
    this.sr        = 44100;  // actual sample rate from AudioContext
    this.ready     = false;
  }

  async init() {
    // Disable browser-side processing so we capture the raw keystroke sound
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      video: false,
    });

    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.sr  = this.ctx.sampleRate;
    this.ring = new Float32Array(Math.ceil(this.sr * CFG.RING_S));

    const src = this.ctx.createMediaStreamSource(stream);

    // AnalyserNode drives the waveform display
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    src.connect(this.analyser);

    // ScriptProcessorNode fills the ring buffer. Chosen over AudioWorklet,
    // which needs a secure context and a separate module file.
    this.processor = this.ctx.createScriptProcessor(2048, 1, 1);
    this.processor.onaudioprocess = (e) => {
      const data = e.inputBuffer.getChannelData(0);
      const N = this.ring.length;
      for (let i = 0; i < data.length; i++) this.ring[(this.written + i) % N] = data[i];
      this.written += data.length;
      this.lastChunkAt = performance.now();
      if (this.waiters.length) {
        this.waiters = this.waiters.filter(w => {
          if (this.written < w.frame) return true;
          w.resolve();
          return false;
        });
      }
    };
    src.connect(this.processor);
    // Must connect to an output node for onaudioprocess to fire in Chrome
    this.processor.connect(this.ctx.destination);

    this.ready = true;
  }

  // Resolves once the stream has reached absolute frame `frame`.
  waitUntil(frame) {
    if (this.written >= frame) return Promise.resolve();
    return new Promise(resolve => this.waiters.push({ frame, resolve }));
  }

  // Frames [start, start+len) by absolute index; zeros where the ring no
  // longer (or not yet) holds them.
  readAbsolute(start, len) {
    const out = new Float32Array(len);
    const N = this.ring.length;
    for (let i = 0; i < len; i++) {
      const a = start + i;
      if (a >= 0 && a < this.written && a >= this.written - N) out[i] = this.ring[a % N];
    }
    return out;
  }

  // Converts an event timestamp (performance.now() timebase) to an absolute
  // stream frame. `written` only advances once per processing chunk (~43 ms),
  // so the time elapsed since the last chunk is added to place the event
  // between chunk boundaries.
  frameAt(eventTimeMs) {
    const sinceChunk = Math.max(0, eventTimeMs - this.lastChunkAt);
    return this.written + Math.round(sinceChunk * this.sr / 1000);
  }

  // Measured key-event → click offset (ms), the median of recent accepted
  // clicks; null until CALIBRATION_STROKES clicks have been seen.
  get latencyMs() {
    if (this.offsets.length < CFG.CALIBRATION_STROKES) return null;
    const s = [...this.offsets].sort((x, y) => x - y);
    return s[Math.floor(s.length / 2)];
  }

  // Cuts the clip for a key event. Searches the audio around the event for
  // the keystroke click (short-time energy peak) and returns CAPTURE_MS of
  // audio starting PRE_ONSET_MS before it, so every clip has its click at the
  // same position. Before calibration the search window is wide; afterwards it
  // is centred on the measured latency. Returns null when no click stands out
  // from the background (MIN_PEAK_DB), so silent presses don't become samples.
  async captureKeystroke(eventTimeMs = performance.now()) {
    const ms = t => Math.round(t * this.sr / 1000);
    const at = this.frameAt(eventTimeMs);
    const lat = this.latencyMs;
    const [w0, w1] = lat === null
      ? CFG.SEARCH_WINDOW_MS
      : [lat - CFG.CALIBRATED_HALF_WINDOW_MS, lat + CFG.CALIBRATED_HALF_WINDOW_MS];
    const winStart = at + ms(w0), winEnd = at + ms(w1);
    const clip = ms(CFG.CAPTURE_MS), pre = ms(CFG.PRE_ONSET_MS);

    // Wait until the window and a full clip after its end have arrived
    await this.waitUntil(winEnd + clip);

    // Short-time energy (5 ms windows, 1 ms hop) over the window plus 150 ms
    // of context before it; the context makes the median a background estimate.
    const ctxStart = winStart - ms(150);
    const region = this.readAbsolute(ctxStart, winEnd - ctxStart);
    const win = ms(5), hop = Math.max(1, ms(1));
    const cum = new Float64Array(region.length + 1);
    for (let i = 0; i < region.length; i++) cum[i + 1] = cum[i] + region[i] * region[i];
    const energies = [];
    let best = -1, bestE = -1;
    for (let s0 = 0; s0 + win <= region.length; s0 += hop) {
      const e = cum[s0 + win] - cum[s0];
      energies.push(e);
      if (s0 >= winStart - ctxStart && e > bestE) { bestE = e; best = s0; }
    }
    energies.sort((x, y) => x - y);
    const background = energies[Math.floor(energies.length / 2)] + 1e-12;
    const peakDb = 10 * Math.log10(bestE / background);
    if (best < 0 || peakDb < CFG.MIN_PEAK_DB) return { audio: null, peakDb, latencyMs: lat };

    const clickFrame = ctxStart + best;
    this.offsets.push((clickFrame - at) * 1000 / this.sr);
    if (this.offsets.length > CFG.CALIBRATION_HISTORY) this.offsets.shift();

    return { audio: this.readAbsolute(clickFrame - pre, clip), peakDb, latencyMs: this.latencyMs };
  }

  // Time-domain data for the waveform display canvas
  getWaveform() {
    const d = new Float32Array(this.analyser.frequencyBinCount);
    this.analyser.getFloatTimeDomainData(d);
    return d;
  }
}
