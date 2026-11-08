import { CFG, displayKey } from './config.js?v=15';

// ─── Badge helpers ────────────────────────────────────────────────────────────
export function setBadge(id, text, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.className = 'badge ' + (cls || '');
}

// ─── Training log ─────────────────────────────────────────────────────────────
export function log(msg) {
  const el = document.getElementById('train-log');
  if (!el) return;
  el.textContent += msg + '\n';
  el.scrollTop = el.scrollHeight;
}

export function clearLog() {
  const el = document.getElementById('train-log');
  if (el) el.textContent = '';
}

// ─── Keyboard grid ────────────────────────────────────────────────────────────
export function buildKeyboardDOM(rows, container) {
  container.innerHTML = '';
  for (const row of rows) {
    const rowEl = document.createElement('div');
    rowEl.className = 'kb-row';
    for (const k of row) {
      const btn = document.createElement('div');
      btn.className = 'kb-key';
      btn.id = `key-${k === ' ' ? 'space' : k}`;
      btn.dataset.key = k;

      const label = document.createElement('span');
      label.className = 'key-label';
      label.textContent = displayKey(k);

      const cnt = document.createElement('span');
      cnt.className = 'key-count';
      cnt.textContent = '0';

      btn.append(label, cnt);
      rowEl.append(btn);
    }
    container.append(rowEl);
  }
}

// Iterates every key element (not just keys in `counts`) so keys whose
// samples were deleted are reset to 0.
export function updateKeyboard(counts) {
  for (const el of document.querySelectorAll('.kb-key')) {
    const n   = counts[el.dataset.key] || 0;
    const cnt = el.querySelector('.key-count');
    if (cnt) cnt.textContent = n;
    el.classList.toggle('has-data', n >= CFG.MIN_SAMPLES);
    el.classList.toggle('partial',  n > 0 && n < CFG.MIN_SAMPLES);
  }
}

export function flashKey(k, cls) {
  const id = `key-${k === ' ' ? 'space' : k}`;
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.add(cls);
  setTimeout(() => el.classList.remove(cls), 180);
}

// ─── Record tab stats ─────────────────────────────────────────────────────────
export function updateRecordStats(counts) {
  const keys  = Object.entries(counts).filter(([, n]) => n >= CFG.MIN_SAMPLES).length;
  const total = Object.values(counts).reduce((s, n) => s + n, 0);
  const el = document.getElementById('record-stats');
  if (el) el.textContent = `${keys} key${keys !== 1 ? 's' : ''} ready · ${total} sample${total !== 1 ? 's' : ''} total`;
}

// Matches the canvas drawing buffer to its CSS size (× devicePixelRatio) so
// drawings aren't stretched, and returns a context in CSS-pixel coordinates.
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width  = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W, H };
}

// ─── Waveform canvas ─────────────────────────────────────────────────────────
export function drawWaveform(canvas, data) {
  const { ctx, W, H } = fitCanvas(canvas);
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = getComputedStyle(document.documentElement)
    .getPropertyValue('--accent').trim() || '#5b8def';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = (i / data.length) * W;
    const y = H / 2 + data[i] * (H / 2 - 4);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// ─── Capture view ────────────────────────────────────────────────────────────
// Energy envelope around the last key event: search window (shaded), key
// event (grey line at 0 ms), threshold (dashed), detected click (green if
// kept, red if ignored).
export function drawCapture(canvas, cut) {
  const { ctx, W, H } = fitCanvas(canvas);
  ctx.clearRect(0, 0, W, H);
  const pad = { l: 34, r: 8, t: 8, b: 16 };
  const [r0, r1] = cut.rangeMs, DB_MAX = 40;
  const x = t => pad.l + (t - r0) / (r1 - r0) * (W - pad.l - pad.r);
  const y = db => pad.t + (1 - Math.max(0, Math.min(DB_MAX, db)) / DB_MAX) * (H - pad.t - pad.b);

  ctx.fillStyle = 'rgba(91,141,239,.12)';
  ctx.fillRect(x(cut.window[0]), pad.t, x(cut.window[1]) - x(cut.window[0]), H - pad.t - pad.b);

  ctx.font = '10px sans-serif';
  ctx.fillStyle = 'rgba(255,255,255,.4)';
  ctx.textAlign = 'right';
  for (const db of [0, 20, 40]) ctx.fillText(`${db}dB`, pad.l - 4, y(db) + 3);
  ctx.textAlign = 'center';
  for (let t = Math.ceil(r0 / 100) * 100; t <= r1; t += 100) ctx.fillText(`${t}`, x(t), H - 3);

  ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x(0), pad.t); ctx.lineTo(x(0), H - pad.b); ctx.stroke();

  ctx.setLineDash([4, 3]); ctx.strokeStyle = '#f5a623';
  ctx.beginPath(); ctx.moveTo(pad.l, y(cut.threshold)); ctx.lineTo(W - pad.r, y(cut.threshold)); ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = '#5b8def'; ctx.lineWidth = 1.2; ctx.beginPath();
  cut.envDb.forEach((db, i) => { const px = x(r0 + i), py = y(db); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
  ctx.stroke();

  ctx.strokeStyle = cut.audio ? '#3ecf8e' : '#e05252'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x(cut.peakMs), pad.t); ctx.lineTo(x(cut.peakMs), H - pad.b); ctx.stroke();
}

// ─── Training accuracy chart ─────────────────────────────────────────────────
export function drawChart(canvas, chartData) {
  const { ctx, W, H } = fitCanvas(canvas);
  const pad = { t: 10, r: 10, b: 24, l: 36 };
  const iW = W - pad.l - pad.r;
  const iH = H - pad.t - pad.b;

  ctx.clearRect(0, 0, W, H);
  if (!chartData.trainAcc.length) return;

  const epochs = chartData.trainAcc.length;
  const xScale = iW / Math.max(epochs - 1, 1);

  const drawLine = (data, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad.l + i * xScale;
      const y = pad.t + iH * (1 - v);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  [0, 0.25, 0.5, 0.75, 1].forEach(v => {
    const y = pad.t + iH * (1 - v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + iW, y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`${(v * 100).toFixed(0)}%`, pad.l - 4, y + 3);
  });

  drawLine(chartData.trainAcc, '#5b8def');
  if (chartData.valAcc.length) drawLine(chartData.valAcc, '#3ecf8e');

  // Legend
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'left';
  [[`#5b8def`, 'Train acc'], [`#3ecf8e`, 'Val acc']].forEach(([c, l], i) => {
    ctx.fillStyle = c;
    ctx.fillRect(pad.l + 4 + i * 90, pad.t + 2, 10, 10);
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fillText(l, pad.l + 18 + i * 90, pad.t + 11);
  });
}

// ─── Prediction display ───────────────────────────────────────────────────────
export function renderConfidence(container, probs, keys) {
  container.innerHTML = '';
  const sorted = keys.map((k, i) => [k, probs[i]]).sort((a, b) => b[1] - a[1]);
  for (const [k, p] of sorted.slice(0, 6)) {
    const row = document.createElement('div');
    row.className = 'conf-row';
    row.innerHTML = `
      <span class="conf-key">${displayKey(k)}</span>
      <div class="conf-bar-wrap"><div class="conf-bar" style="width:${(p * 100).toFixed(1)}%"></div></div>
      <span class="conf-pct">${(p * 100).toFixed(1)}%</span>`;
    container.append(row);
  }
}

export function updatePredDisplay(predKey, actualKey) {
  const pd = document.getElementById('pred-display');
  const pa = document.getElementById('pred-actual');
  if (pd) pd.textContent = predKey ? displayKey(predKey) : '—';
  if (pa) pa.textContent = actualKey ? displayKey(actualKey) : '—';
  if (pd && predKey && actualKey)
    pd.classList.toggle('correct', predKey === actualKey);
}

export function updatePredStats(total, correct) {
  const el = document.getElementById('pred-stats');
  if (!el) return;
  const pct = total ? ((correct / total) * 100).toFixed(1) : '—';
  el.textContent = `${correct}/${total} correct (${pct}%)`;
}

export function prependHistory(container, actualKey, predKey, probs, keys) {
  const row = document.createElement('div');
  row.className = 'hist-row' + (actualKey === predKey ? ' correct' : ' wrong');
  const top2 = keys.map((k, i) => [k, probs[i]]).sort((a, b) => b[1] - a[1]).slice(0, 2);
  row.innerHTML = `
    <span class="hist-actual">${displayKey(actualKey)}</span>
    <span class="hist-arrow">→</span>
    <span class="hist-pred">${displayKey(predKey)}</span>
    <span class="hist-probs">${top2.map(([k, p]) => `${displayKey(k)} ${(p*100).toFixed(0)}%`).join(' · ')}</span>`;
  container.prepend(row);
  // Keep list short
  while (container.children.length > 20) container.lastChild.remove();
}
