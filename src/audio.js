import { CFG } from './config.js?v=9';

// Continuously fills a ring buffer from the microphone.
// Provides snapshot extraction keyed to any past moment.
export class AudioCapture {
  constructor() {
    this.ctx       = null;
    this.analyser  = null;
    this.processor = null;
    this.ring      = null;   // Float32Array ring buffer
    this.pos       = 0;      // next write position
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

    // ScriptProcessorNode fills the ring buffer sample by sample.
    // We deliberately use this deprecated API instead of AudioWorklet
    // because AudioWorklet requires HTTPS — ScriptProcessorNode works
    // on plain http://localhost and file:// without extra server config.
    this.processor = this.ctx.createScriptProcessor(2048, 1, 1);
    this.processor.onaudioprocess = (e) => {
      const data = e.inputBuffer.getChannelData(0);
      for (let i = 0; i < data.length; i++) {
        this.ring[this.pos] = data[i];
        this.pos = (this.pos + 1) % this.ring.length;
      }
    };
    src.connect(this.processor);
    // Must connect to an output node for onaudioprocess to fire in Chrome
    this.processor.connect(this.ctx.destination);

    this.ready = true;
  }

  // Returns durationMs of audio that ended delayMs ago.
  // extract(0, 200) → the most recent 200 ms.
  extract(delayMs, durationMs) {
    const dSamples = Math.round(durationMs * this.sr / 1000);
    const offset   = Math.round(delayMs   * this.sr / 1000) + dSamples;
    const result   = new Float32Array(dSamples);
    for (let i = 0; i < dSamples; i++) {
      let p = this.pos - offset + i;
      if (p < 0) p += this.ring.length;
      result[i] = this.ring[p % this.ring.length];
    }
    return result;
  }

  // Time-domain data for the waveform display canvas
  getWaveform() {
    const d = new Float32Array(this.analyser.frequencyBinCount);
    this.analyser.getFloatTimeDomainData(d);
    return d;
  }
}
