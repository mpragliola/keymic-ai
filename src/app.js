import { CFG, KEYBOARD_ROWS, TRACKED } from './config.js?v=15';
import { AudioCapture } from './audio.js?v=15';
import { getFilterbank, computeMelSpec } from './features.js?v=15';
import { api } from './api.js?v=15';
import { LegacyStore } from './legacy-store.js?v=15';
import { initSamplesTab } from './samples.js?v=15';
import {
  setBadge, log, clearLog,
  buildKeyboardDOM, updateKeyboard, flashKey, updateRecordStats,
  drawWaveform, drawChart, drawCapture,
  renderConfidence, updatePredDisplay, updatePredStats, prependHistory,
} from './ui.js?v=15';

// Samples, training and prediction live in the Python backend; the browser
// captures and cuts keystroke audio, shows progress and draws previews.

// ─── Shared state ─────────────────────────────────────────────────────────────
const state = {
  audio:       null,     // AudioCapture instance
  micActive:   false,
  backend:     null,     // /health response; null while the backend is unreachable
  counts:      {},       // { key → n } as stored in the backend
  sr:          null,     // sample rate of the stored samples
  model:       null,     // /model info (keys, sr, val_acc, …); null when none
  train:       { state: 'idle', logTotal: 0, polling: false },
  predTotal:   0,
  predCorrect: 0,
  animFrame:   null,
  rejected:    0,        // presses ignored because no click was detected
};

const isActive = s => s === 'running' || s === 'stopping';
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ─── Tab switching ────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === target));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${target}`));
    if (target === 'record') startWaveformLoop();
    else stopWaveformLoop();
    if (target === 'samples') samplesTab.refresh();
    if (target === 'predict') {
      updatePredStatus();
      if (state.model && !state.micActive) startMic();
    }
  });
});

// ─── Mic ──────────────────────────────────────────────────────────────────────
// Shared by the Record and Predict tabs; called from click handlers because
// browsers only start an AudioContext from a user gesture.
async function startMic() {
  if (state.micActive) return true;
  try {
    state.audio = new AudioCapture();
    await state.audio.init();
    state.audio.minPeakDb = Number(thresholdInput.value);
    state.micActive = true;
    setBadge('badge-mic', `mic ${state.audio.sr / 1000} kHz`, 'green');
    startWaveformLoop();
    return true;
  } catch (e) {
    setBadge('badge-mic', 'mic error', 'red');
    alert('Microphone access denied: ' + e.message);
    return false;
  } finally {
    updateMicButtons();
    updatePredStatus();
  }
}

function updateMicButtons() {
  for (const id of ['btn-mic', 'btn-mic-predict']) {
    const b = document.getElementById(id);
    b.disabled = state.micActive;
    b.textContent = state.micActive ? 'Mic on' : 'Enable mic';
  }
}

document.getElementById('btn-mic').addEventListener('click', startMic);
document.getElementById('btn-mic-predict').addEventListener('click', startMic);

// ─── Click threshold ─────────────────────────────────────────────────────────
// Persisted per browser; localStorage can throw (e.g. storage disabled).
const thresholdInput = document.getElementById('threshold');
function loadThreshold() {
  try { const v = Number(localStorage.getItem('keymic-threshold-db')); if (v) return v; } catch {}
  return CFG.MIN_PEAK_DB;
}
function applyThreshold(db) {
  thresholdInput.value = db;
  document.getElementById('threshold-val').textContent = `${db} dB`;
  if (state.audio) state.audio.minPeakDb = db;
}
thresholdInput.addEventListener('input', () => {
  const db = Number(thresholdInput.value);
  applyThreshold(db);
  try { localStorage.setItem('keymic-threshold-db', String(db)); } catch {}
});
applyThreshold(loadThreshold());

// ─── Waveform animation ───────────────────────────────────────────────────────
const waveCanvas = document.getElementById('waveform');

function startWaveformLoop() {
  if (state.animFrame) return;
  const tick = () => {
    if (state.micActive) drawWaveform(waveCanvas, state.audio.getWaveform());
    state.animFrame = requestAnimationFrame(tick);
  };
  state.animFrame = requestAnimationFrame(tick);
}

function stopWaveformLoop() {
  if (state.animFrame) { cancelAnimationFrame(state.animFrame); state.animFrame = null; }
}

// ─── Keyboard capture ─────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.matches?.('input, select, textarea')) return;   // typing in a form field
  const k = e.key === ' ' ? ' ' : e.key.toLowerCase();
  if (!TRACKED.has(k)) return;
  e.preventDefault();
  if (e.repeat) return;   // held key: auto-repeat events have no new sound

  const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;
  if (activeTab === 'record' && state.micActive) recordKey(k, e.timeStamp);
  else if (activeTab === 'predict' && state.micActive && state.model) predictKey(k, e.timeStamp);
});

// ─── Capture stats ───────────────────────────────────────────────────────────
// Shows the measured mic latency and how many presses were ignored.
function updateCaptureStats(cut) {
  if (!cut.audio) state.rejected++;
  drawCapture(document.getElementById('capture-view'), cut);
  const lat = cut.latencyMs;
  document.getElementById('capture-stats').textContent =
    (lat === null ? 'Latency: calibrating…' : `Latency: ${Math.round(lat)} ms`) +
    ` · last click ${cut.peakDb.toFixed(0)} dB over background` +
    (state.rejected ? ` · ${state.rejected} press${state.rejected > 1 ? 'es' : ''} ignored (no click)` : '');
}

function showRecordError(msg) {
  document.getElementById('record-error').textContent = msg || '';
}

// ─── Samples (backend) ───────────────────────────────────────────────────────
function renderCounts() {
  updateKeyboard(state.counts);
  updateRecordStats(state.counts);
  const total = Object.values(state.counts).reduce((s, n) => s + n, 0);
  setBadge('badge-data', total ? `${total} samples` : 'no data', total ? 'green' : '');
}

async function refreshSamples() {
  const s = await api.samples();
  state.counts = s.counts;
  state.sr = s.sr;
  renderCounts();
}

async function deleteSample(key, index) {
  const r = await api.deleteSample(key, index);
  if (r.count) state.counts[key] = r.count;
  else delete state.counts[key];
  renderCounts();
}

const samplesTab = initSamplesTab({
  getCounts:  () => state.counts,
  getSamples: key => api.keySamples(key),
  deleteSample,
  getSr:      async () => state.sr,
  getFilterbank,
  computeMelSpec,
});

async function deleteSamples() {
  if (!confirm('Delete all recorded samples on the backend? The trained model is kept.')) return;
  try {
    await api.deleteSamples();   // also stops a running training
    state.counts = {};
    state.sr = null;
    renderCounts();
    log('Samples deleted.');
  } catch (e) {
    alert('Could not delete samples: ' + e.message);
  }
}
document.getElementById('btn-clear').addEventListener('click', deleteSamples);
document.getElementById('btn-delete-samples').addEventListener('click', deleteSamples);

// ─── Record ───────────────────────────────────────────────────────────────────
async function recordKey(k, eventTime) {
  const cut = await state.audio.captureKeystroke(eventTime);
  updateCaptureStats(cut);
  if (!cut.audio) { flashKey(k, 'rejected'); return; }
  try {
    const r = await api.addSamples(k, state.audio.sr, cut.audio);
    flashKey(k, 'recording');
    state.counts[k] = r.count;
    state.sr = state.audio.sr;
    renderCounts();
    showRecordError('');
  } catch (e) {
    flashKey(k, 'rejected');
    showRecordError('Not saved: ' + e.message);
  }
}

// ─── Predict ─────────────────────────────────────────────────────────────────
function updatePredStatus() {
  const el = document.getElementById('pred-status');
  if (!state.backend)        el.textContent = 'Backend offline — start it with "python run.py" in backend/.';
  else if (!state.model)     el.textContent = 'No trained model — train one in the Train tab.';
  else if (!state.micActive) el.textContent = 'Enable the mic, then type a key.';
  else if (state.model.sr !== state.audio.sr)
    el.textContent = `Model was trained at ${state.model.sr / 1000} kHz but the mic runs at ${state.audio.sr / 1000} kHz.`;
  else                       el.textContent = '';
}

async function predictKey(actualKey, eventTime) {
  const cut = await state.audio.captureKeystroke(eventTime);
  updateCaptureStats(cut);
  if (!cut.audio) {
    document.getElementById('pred-status').textContent = 'No keystroke sound detected — press ignored.';
    return;
  }
  let r;
  try {
    r = await api.predict(state.audio.sr, cut.audio);
  } catch (e) {
    document.getElementById('pred-status').textContent = 'Prediction failed: ' + e.message;
    return;
  }
  updatePredStatus();
  const keys  = r.probs.map(p => p.key);
  const probs = r.probs.map(p => p.p);

  state.predTotal++;
  if (r.key === actualKey) state.predCorrect++;
  flashKey(actualKey, 'predicting');
  renderConfidence(document.getElementById('conf-bars'), probs, keys);
  updatePredDisplay(r.key, actualKey);
  updatePredStats(state.predTotal, state.predCorrect);
  prependHistory(document.getElementById('pred-history'), actualKey, r.key, probs, keys);
}

document.getElementById('btn-reset-stats').addEventListener('click', () => {
  state.predTotal = state.predCorrect = 0;
  updatePredStats(0, 0);
  document.getElementById('pred-history').innerHTML = '';
});

// ─── Model ───────────────────────────────────────────────────────────────────
async function refreshModel() {
  state.model = await api.model();
  const m = state.model;
  setBadge('badge-model',
    m ? `model: ${m.keys.length} keys · val ${(m.val_acc * 100).toFixed(0)}%` : 'no model',
    m ? 'green' : '');
  updateTrainButtons();
  updatePredStatus();
}

// ─── Train (backend, polled for live progress) ───────────────────────────────
const btnTrain    = document.getElementById('btn-train');
const btnStop     = document.getElementById('btn-stop-train');
const btnDelete   = document.getElementById('btn-delete-model');
const epochsInput = document.getElementById('train-epochs');
epochsInput.value = CFG.DEFAULT_EPOCHS;

// Train only when idle; Stop only while running; Delete when a model exists
// or a run is active (deleting stops the run so it can't save afterwards).
function updateTrainButtons() {
  const active = isActive(state.train.state);
  btnTrain.disabled  = !state.backend || active;
  btnStop.disabled   = !state.backend || state.train.state !== 'running';
  btnDelete.disabled = !state.backend || (!active && !state.model);
}

// Applies a /train/status response: new log lines, chart, progress line.
function applyStatus(s) {
  for (const line of s.logs) log(line);
  state.train.logTotal = s.log_total;
  state.train.state = s.state;
  drawChart(document.getElementById('train-chart'), { trainAcc: s.history.acc, valAcc: s.history.val_acc });
  const last = i => s.history[i].at(-1);
  document.getElementById('train-progress').textContent =
    s.state === 'idle' ? '' :
    `${s.state} · epoch ${s.epoch}/${s.epochs}` +
    (s.history.acc.length ? ` · acc ${(last('acc') * 100).toFixed(1)}% · val_acc ${(last('val_acc') * 100).toFixed(1)}%` : '');
  updateTrainButtons();
}

// Polls the run until it ends. Only new log lines are requested (since=N);
// if the backend restarted, its log is shorter than ours and is re-read whole.
async function pollTraining() {
  if (state.train.polling) return;
  state.train.polling = true;
  try {
    while (true) {
      const s = await api.trainStatus(state.train.logTotal);
      if (s.log_total < state.train.logTotal) { clearLog(); state.train.logTotal = 0; continue; }
      applyStatus(s);
      if (!isActive(s.state)) break;
      await sleep(CFG.POLL_MS);
    }
    await refreshModel();
  } catch (e) {
    log('Lost contact with the backend: ' + e.message);
    connectBackend();
  } finally {
    state.train.polling = false;
  }
}

btnTrain.addEventListener('click', async () => {
  btnTrain.disabled = true;
  clearLog();
  state.train.logTotal = 0;
  try {
    applyStatus(await api.train({ epochs: Number(epochsInput.value) || CFG.DEFAULT_EPOCHS }));
    pollTraining();
  } catch (e) {
    log('Error: ' + e.message);
    updateTrainButtons();
  }
});

btnStop.addEventListener('click', async () => {
  btnStop.disabled = true;
  try { await api.stopTraining(); } catch (e) { log('Error: ' + e.message); }
});

btnDelete.addEventListener('click', async () => {
  if (!confirm('Delete the trained model? Recorded samples are kept.')) return;
  try {
    await api.deleteModel();   // also stops a running training
    log('Model deleted.');
  } catch (e) {
    log('Error: ' + e.message);
  }
  await refreshModel().catch(() => {});
});

// ─── Samples recorded in the browser before the backend ──────────────────────
async function checkLegacy() {
  const counts = await LegacyStore.getCounts();
  const keys = Object.keys(counts).filter(k => counts[k] > 0);
  const total = keys.reduce((s, k) => s + counts[k], 0);
  const card = document.getElementById('legacy-card');
  card.hidden = total === 0;
  if (total) document.getElementById('legacy-text').textContent =
    `${total} samples for ${keys.length} key${keys.length > 1 ? 's' : ''} are stored in this browser from before ` +
    `the backend. Upload them, or discard them if they were recorded before the click-detection fix.`;
  document.getElementById('btn-legacy-upload').disabled = !state.backend;
}

document.getElementById('btn-legacy-upload').addEventListener('click', async () => {
  const msg = document.getElementById('legacy-text');
  try {
    const sr = await LegacyStore.getSr();
    const counts = await LegacyStore.getCounts();
    for (const key of Object.keys(counts)) {
      const clips = await LegacyStore.loadKey(key);
      if (!clips.length) continue;
      const all = new Float32Array(clips.length * clips[0].length);
      clips.forEach((c, i) => all.set(c, i * c.length));
      msg.textContent = `Uploading ${clips.length} samples for "${key === ' ' ? 'SPC' : key}"…`;
      await api.addSamples(key, sr, all);   // one request per key: clips concatenated
    }
    await LegacyStore.clearAll();
    await refreshSamples();
    await checkLegacy();
  } catch (e) {
    msg.textContent = 'Upload failed: ' + e.message + ' — nothing was deleted from the browser.';
  }
});

document.getElementById('btn-legacy-discard').addEventListener('click', async () => {
  if (!confirm('Discard the samples stored in this browser? Samples on the backend are not affected.')) return;
  await LegacyStore.clearAll();
  await checkLegacy();
});

// ─── Backend connection ──────────────────────────────────────────────────────
// Retries every 3 s while unreachable; on (re)connect loads samples, model and
// any training run in progress (e.g. started before a page reload).
let retryTimer = null;
async function connectBackend() {
  clearTimeout(retryTimer);
  try {
    state.backend = await api.health();
    const gpu = state.backend.gpu;
    setBadge('badge-backend', gpu ? `GPU: ${gpu}` : 'backend: CPU only', gpu ? 'green' : 'yellow');
    document.getElementById('backend-banner').hidden = true;
    await Promise.all([refreshSamples(), refreshModel()]);
    const s = await api.trainStatus(0);
    clearLog();
    state.train.logTotal = 0;
    applyStatus(s);
    if (isActive(s.state)) pollTraining();
  } catch (e) {
    state.backend = null;
    setBadge('badge-backend', 'backend offline', 'red');
    const banner = document.getElementById('backend-banner');
    banner.hidden = false;
    banner.textContent = `Backend not reachable at ${CFG.BACKEND_URL}. Start it: cd backend → python run.py. Retrying…`;
    retryTimer = setTimeout(connectBackend, 3000);
  }
  updateTrainButtons();
  updatePredStatus();
  checkLegacy();
}

// ─── Init ─────────────────────────────────────────────────────────────────────
buildKeyboardDOM(KEYBOARD_ROWS, document.getElementById('keyboard'));
renderCounts();
connectBackend();
