import { CFG } from './config.js?v=15';

// Lazy singleton — rebuilt only when the sample rate changes.
let _fb = null;

export function getFilterbank(sr) {
  if (_fb && _fb.sr === sr) return _fb.tensor;
  if (_fb) _fb.tensor.dispose();
  _fb = { tensor: buildFilterbank(CFG.N_MELS, CFG.FRAME_LEN, sr), sr };
  return _fb.tensor;
}

// [nFftBins × nMels] triangular mel filterbank matrix spanning fmin…fmax Hz.
export function buildFilterbank(nMels, fftLen, sr, fmin = CFG.FMIN_HZ, fmax = CFG.FMAX_HZ) {
  const nBins  = Math.floor(fftLen / 2) + 1;
  const toMel  = hz => 2595 * Math.log10(1 + hz / 700);
  const fromMel = m => 700 * (Math.pow(10, m / 2595) - 1);
  const melMin = toMel(fmin), melMax = toMel(Math.min(fmax, sr / 2));

  const pts = Array.from({ length: nMels + 2 }, (_, i) =>
    Math.floor((fftLen + 1) * fromMel(melMin + (i * (melMax - melMin)) / (nMels + 1)) / sr)
  );

  const data = new Float32Array(nBins * nMels);
  for (let m = 1; m <= nMels; m++) {
    const lo = pts[m - 1], c = pts[m], hi = pts[m + 1];
    for (let k = lo; k < c  && k < nBins; k++) data[k * nMels + (m-1)] = (k - lo)  / (c  - lo + 1e-9);
    for (let k = c;  k <= hi && k < nBins; k++) data[k * nMels + (m-1)] = (hi - k)  / (hi - c  + 1e-9);
  }
  return tf.tensor2d(data, [nBins, nMels]);
}

// Returns [N_FRAMES, N_MELS, 1] log-mel spectrogram tensor.
export function computeMelSpec(audio, filterbank) {
  return tf.tidy(() => {
    const sig = tf.tensor1d(audio);

    // Frame + Hann window
    const frames   = tf.signal.frame(sig, CFG.FRAME_LEN, CFG.HOP_LEN);
    const window   = tf.signal.hannWindow(CFG.FRAME_LEN);
    const windowed = tf.mul(frames, window);

    // RFFT lives in tf.spectral (tf.signal only has frame/window/stft helpers).
    // Result is complex64 [nF, FRAME_LEN/2+1]; power = re² + im².
    const rfft  = tf.spectral.rfft(windowed);
    const power = tf.add(tf.square(tf.real(rfft)), tf.square(tf.imag(rfft)));

    // Mel filterbank + log compression
    const mel    = tf.matMul(power, filterbank);   // [nF, N_MELS]
    const logMel = tf.log(tf.add(mel, 1e-6));

    // Per-spectrogram z-score (removes room-noise offset between sessions)
    const mean   = logMel.mean();
    const std    = tf.sqrt(tf.moments(logMel).variance);
    const normed = tf.div(tf.sub(logMel, mean), tf.add(std, 1e-8));

    // Pad/crop time axis to exactly N_FRAMES
    const nF = normed.shape[0];
    const shaped = nF >= CFG.N_FRAMES
      ? normed.slice([0, 0], [CFG.N_FRAMES, CFG.N_MELS])
      : tf.concat([normed, tf.zeros([CFG.N_FRAMES - nF, CFG.N_MELS])], 0);

    return shaped.expandDims(2);  // [N_FRAMES, N_MELS, 1]
  });
}
