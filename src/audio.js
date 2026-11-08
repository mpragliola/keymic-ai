import { CFG } from './config.js?v=14';

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
    this.minPeakDb = CFG.MIN_PEAK_DB;  // click detection threshold (dB over background)
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

  // Cuts the clip for a key event.
  //
  // Analyses ANALYSIS_RANGE_MS of audio around the event, high-passed at
  // DETECT_HIGHPASS_HZ so fans, hum and rumble don't mask the click: a short-time energy
  // envelope (5 ms windows, 1 ms hop) in dB above the background level (a low
  // percentile of the envelope, so other keystrokes in range don't raise it).
  // The click is the envelope peak inside the search window: wide before
  // calibration, ± CALIBRATED_HALF_WINDOW_MS around the measured latency
  // after. If the narrow search finds nothing, the wide window is tried too,
  // so a wrong calibration recovers instead of rejecting every press.
  //
  // Returns { audio, ...diagnostics }; audio is CAPTURE_MS starting
  // PRE_ONSET_MS before the click, or null if no peak reaches minPeakDb.
  async captureKeystroke(eventTimeMs = performance.now()) {
    const ms = t => Math.round(t * this.sr / 1000);
    const at = this.frameAt(eventTimeMs);
    const [r0, r1] = CFG.ANALYSIS_RANGE_MS;
    const clip = ms(CFG.CAPTURE_MS), pre = ms(CFG.PRE_ONSET_MS);

    // Wait until the analysis range and a full clip after its end have arrived
    await this.waitUntil(at + ms(r1) + clip);

    // Read 30 ms extra before the range so the filter's start-up transient is discarded
    const warm = ms(30);
    const region = highpass(this.readAbsolute(at + ms(r0) - warm, ms(r1 - r0) + warm),
                            this.sr, CFG.DETECT_HIGHPASS_HZ).subarray(warm);
    const win = ms(5), hop = ms(1);
    const cum = new Float64Array(region.length + 1);
    for (let i = 0; i < region.length; i++) cum[i + 1] = cum[i] + region[i] * region[i];
    const env = [];
    for (let s0 = 0; s0 + win <= region.length; s0 += hop) env.push(cum[s0 + win] - cum[s0]);
    const sorted = [...env].sort((x, y) => x - y);
    const background = sorted[Math.floor(sorted.length * CFG.BACKGROUND_PERCENTILE / 100)] + 1e-12;
    const envDb = Float32Array.from(env, e => 10 * Math.log10((e + 1e-12) / background));

    // envDb[i] covers r0 + i ms … r0 + i + 5 ms relative to the event
    const findPeak = ([w0, w1]) => {
      let bi = -1, bd = -Infinity;
      for (let i = Math.max(0, w0 - r0); i <= Math.min(envDb.length - 1, w1 - r0); i++)
        if (envDb[i] > bd) { bd = envDb[i]; bi = i; }
      return { ms: r0 + bi, db: bd };
    };

    const lat = this.latencyMs;
    let window = lat === null
      ? CFG.SEARCH_WINDOW_MS
      : [Math.round(lat - CFG.CALIBRATED_HALF_WINDOW_MS), Math.round(lat + CFG.CALIBRATED_HALF_WINDOW_MS)];
    let peak = findPeak(window);
    if (lat !== null && peak.db < this.minPeakDb) {
      window = CFG.SEARCH_WINDOW_MS;
      peak = findPeak(window);
    }

    const diag = { envDb, rangeMs: [r0, r1], window, peakMs: peak.ms, peakDb: peak.db,
                   threshold: this.minPeakDb, latencyMs: lat };
    if (peak.db < this.minPeakDb) return { audio: null, ...diag };

    this.offsets.push(peak.ms);
    if (this.offsets.length > CFG.CALIBRATION_HISTORY) this.offsets.shift();
    return { audio: this.readAbsolute(at + ms(peak.ms) - pre, clip), ...diag, latencyMs: this.latencyMs };
  }

  // Time-domain data for the waveform display canvas
  getWaveform() {
    const d = new Float32Array(this.analyser.frequencyBinCount);
    this.analyser.getFloatTimeDomainData(d);
    return d;
  }
}

// 4th-order Butterworth high-pass: two cascaded RBJ biquads (Q = 1/√2 each).
// Used only for click detection; stored clips stay unfiltered.
function highpass(x, sr, fc) {
  const w0 = 2 * Math.PI * fc / sr, c = Math.cos(w0), alpha = Math.sin(w0) / (2 * Math.SQRT1_2);
  const a0 = 1 + alpha;
  const b0 = (1 + c) / 2 / a0, b1 = -(1 + c) / a0, b2 = (1 + c) / 2 / a0;
  const a1 = -2 * c / a0, a2 = (1 - alpha) / a0;
  let y = x;
  for (let pass = 0; pass < 2; pass++) {
    const out = new Float32Array(y.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < y.length; i++) {
      const v = b0 * y[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1; x1 = y[i]; y2 = y1; y1 = v; out[i] = v;
    }
    y = out;
  }
  return y;
}
