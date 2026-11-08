import { displayKey } from './config.js?v=15';

const PAGE_SIZE = 12;

// Samples tab: browse one key's recorded clips page by page (newest first),
// each with waveform, log-mel spectrogram preview, playback and delete.
//
// deps: { getCounts() → {key: n}, getSamples(key) → Promise<Float32Array[]>,
//         deleteSample(key, index) → Promise, getSr() → Promise<number|null>,
//         getFilterbank(sr), computeMelSpec(audio, fb) }
export function initSamplesTab(deps) {
  const keySel = document.getElementById('samples-key');
  const grid   = document.getElementById('samples-grid');
  const label  = document.getElementById('samples-page');
  const prev   = document.getElementById('samples-prev');
  const next   = document.getElementById('samples-next');
  let page = 0;
  let playCtx = null;   // created on first Play click (needs a user gesture)

  const safeRender = () => render().catch(e => { label.textContent = 'Error: ' + e.message; });
  keySel.addEventListener('change', () => { page = 0; safeRender(); });
  prev.addEventListener('click', () => { page--; safeRender(); });
  next.addEventListener('click', () => { page++; safeRender(); });

  // Errors (e.g. backend offline) are shown in the page label instead of thrown
  async function refresh() {
    try { await refreshInner(); } catch (e) { label.textContent = 'Error: ' + e.message; }
  }

  async function refreshInner() {
    const counts = deps.getCounts();
    const keys = Object.keys(counts).filter(k => counts[k] > 0).sort();
    const current = keySel.value;
    keySel.innerHTML = keys.map(k =>
      `<option value="${k === ' ' ? 'SPACE' : k}">${displayKey(k)} (${counts[k]})</option>`).join('');
    if (keys.some(k => (k === ' ' ? 'SPACE' : k) === current)) keySel.value = current;
    await render();
  }

  async function render() {
    grid.innerHTML = '';
    if (!keySel.value) {
      label.textContent = 'No samples recorded';
      prev.disabled = next.disabled = true;
      return;
    }
    const key = keySel.value === 'SPACE' ? ' ' : keySel.value;
    const samples = await deps.getSamples(key);
    const pages = Math.max(1, Math.ceil(samples.length / PAGE_SIZE));
    page = Math.min(Math.max(0, page), pages - 1);
    label.textContent = `Page ${page + 1} / ${pages} · ${samples.length} samples`;
    prev.disabled = page === 0;
    next.disabled = page >= pages - 1;

    const sr = await deps.getSr();
    if (!sr) return;
    const fb = deps.getFilterbank(sr);
    // Newest first: display position i ↔ storage index samples.length-1-i
    for (let i = page * PAGE_SIZE; i < Math.min(samples.length, (page + 1) * PAGE_SIZE); i++) {
      const idx = samples.length - 1 - i;
      grid.append(card(key, idx, samples[idx], sr, fb));
    }
  }

  function card(key, idx, audio, sr, fb) {
    const el = document.createElement('div');
    el.className = 'sample-card';
    el.innerHTML = `
      <div class="sample-head"><span>#${idx + 1}</span><span class="sample-peak"></span></div>
      <canvas class="sample-wave"></canvas>
      <canvas class="sample-spec"></canvas>
      <div class="btn-row">
        <button class="btn-ghost sample-play">▶ Play</button>
        <button class="btn-danger sample-del">Delete</button>
      </div>`;
    grid.append(el);   // attach first so canvases have a layout size

    let peak = 0;
    for (const v of audio) peak = Math.max(peak, Math.abs(v));
    el.querySelector('.sample-peak').textContent = `peak ${(20 * Math.log10(peak + 1e-9)).toFixed(0)} dBFS`;
    drawWave(el.querySelector('.sample-wave'), audio, peak);
    const spec = deps.computeMelSpec(audio, fb);
    drawSpec(el.querySelector('.sample-spec'), spec.dataSync(), spec.shape[0], spec.shape[1]);
    spec.dispose();

    el.querySelector('.sample-play').addEventListener('click', () => play(audio, sr));
    el.querySelector('.sample-del').addEventListener('click', async () => {
      try { await deps.deleteSample(key, idx); }
      catch (e) { label.textContent = 'Delete failed: ' + e.message; return; }
      await refresh();
    });
    return el;
  }

  function play(audio, sr) {
    playCtx ??= new AudioContext();
    const buf = playCtx.createBuffer(1, audio.length, sr);
    buf.copyToChannel(audio, 0);
    const src = playCtx.createBufferSource();
    src.buffer = buf;
    src.connect(playCtx.destination);
    src.start();
  }

  return { refresh };
}

function sizeCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.round(c.clientWidth * dpr);
  c.height = Math.round(c.clientHeight * dpr);
  return c.getContext('2d');
}

// Waveform normalised to the clip's own peak, so quiet clicks stay visible.
// The marker at PRE_ONSET (20 ms of 200 ms) is where the detected click should start.
function drawWave(c, audio, peak) {
  const ctx = sizeCanvas(c), W = c.width, H = c.height;
  ctx.fillStyle = '#0c0c10'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(245,166,35,.5)';
  ctx.beginPath(); ctx.moveTo(W * 0.1, 0); ctx.lineTo(W * 0.1, H); ctx.stroke();
  ctx.strokeStyle = '#5b8def'; ctx.beginPath();
  const step = audio.length / W, g = (H / 2 - 2) / (peak || 1);
  for (let x = 0; x < W; x++) {
    let lo = 0, hi = 0;
    for (let j = Math.floor(x * step); j < Math.floor((x + 1) * step); j++) { lo = Math.min(lo, audio[j]); hi = Math.max(hi, audio[j]); }
    ctx.moveTo(x, H / 2 - hi * g); ctx.lineTo(x, H / 2 - lo * g);
  }
  ctx.stroke();
}

// Spectrogram [frames × mels] as a heatmap: time → x, mel band → y (low at bottom).
function drawSpec(c, data, nFrames, nMels) {
  const ctx = sizeCanvas(c), W = c.width, H = c.height;
  const img = ctx.createImageData(nFrames, nMels);
  for (let t = 0; t < nFrames; t++) for (let m = 0; m < nMels; m++) {
    const z = data[t * nMels + m];                     // z-scored value
    const v = Math.max(0, Math.min(1, (z + 2) / 5));   // map ≈[-2, 3] → [0, 1]
    const o = ((nMels - 1 - m) * nFrames + t) * 4;
    img.data[o]     = Math.round(255 * Math.min(1, v * 1.6));
    img.data[o + 1] = Math.round(255 * Math.max(0, v * 1.4 - 0.4));
    img.data[o + 2] = Math.round(255 * Math.max(0, 0.6 - v) );
    img.data[o + 3] = 255;
  }
  const off = new OffscreenCanvas(nFrames, nMels);
  off.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, W, H);
}
