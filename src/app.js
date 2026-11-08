import { CFG, KEYBOARD_ROWS, TRACKED } from './config.js?v=14';
import { AudioCapture } from './audio.js?v=14';
import { getFilterbank, computeMelSpec } from './features.js?v=14';
import { Store } from './store.js?v=14';
import { initSamplesTab } from './samples.js?v=14';
import { trainFromStore, loadSavedModel, deleteSavedModel } from './model.js?v=14';
import {
  setBadge, log, clearLog,
  buildKeyboardDOM, updateKeyboard, flashKey, updateRecordStats,
  drawWaveform, drawChart, drawCapture,
  renderConfidence, updatePredDisplay, updatePredStats, prependHistory,
} from './ui.js?v=14';

// ─── Shared state ─────────────────────────────────────────────────────────────
const state = {
  audio:        null,     // AudioCapture instance
  micActive:    false,
  samples:      {},       // { key → Float32Array[] }  samples for each key
  loading:      {},       // { key → Promise<Float32Array[]> }  in-flight loads
  counts:       {},       // { key → n }  persisted counts
  model:        null,     // tf.LayersModel
  modelKeys:    null,     // string[]
  predTotal:    0,
  predCorrect:  0,
  chartData:    { trainAcc: [], valAcc: [] },
  animFrame:    null,
  rejected:     0,        // presses ignored because no click was detected
  trainAbort:   null,     // AbortController of the running training, if any
};

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
    setBadge('badge-mic', 'mic on', 'green');
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

// Tells the user what is missing before predictions can run
function updatePredStatus() {
  const el = document.getElementById('pred-status');
  if (!state.model)          el.textContent = 'No trained model — train one in the Train tab.';
  else if (!state.micActive) el.textContent = 'Enable the mic, then type a key.';
  else                       el.textContent = '';
}

document.getElementById('btn-mic').addEventListener('click', startMic);

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
document.getElementById('btn-mic-predict').addEventListener('click', startMic);

// ─── Waveform animation ───────────────────────────────────────────────────────
const waveCanvas = document.getElementById('waveform');

function startWaveformLoop() {
  if (state.animFrame) return;
  const tick = () => {
    if (state.micActive && waveCanvas) {
      drawWaveform(waveCanvas, state.audio.getWaveform());
    }
    state.animFrame = requestAnimationFrame(tick);
  };
  state.animFrame = requestAnimationFrame(tick);
}

function stopWaveformLoop() {
  if (state.animFrame) { cancelAnimationFrame(state.animFrame); state.animFrame = null; }
}

// ─── Keyboard capture ─────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  const k = e.key === ' ' ? ' ' : e.key.toLowerCase();
  if (!TRACKED.has(k)) return;
  e.preventDefault();
  if (e.repeat) return;   // held key: auto-repeat events have no new sound

  // Determine current mode
  const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;

  if (activeTab === 'record' && state.micActive) {
    recordKey(k, e.timeStamp);
  } else if (activeTab === 'predict' && state.model) {
    predictKey(k, e.timeStamp);
  }
});

// ─── Capture stats ───────────────────────────────────────────────────────────
// Shows the measured mic latency and how many presses were ignored.
function updateCaptureStats(cut) {
  if (!cut.audio) state.rejected++;
  drawCapture(document.getElementById('capture-view'), cut);
  const lat = cut.latencyMs;
  const el = document.getElementById('capture-stats');
  el.textContent =
    (lat === null ? 'Latency: calibrating…' : `Latency: ${Math.round(lat)} ms`) +
    ` · last click ${cut.peakDb.toFixed(0)} dB over background` +
    (state.rejected ? ` · ${state.rejected} press${state.rejected > 1 ? 'es' : ''} ignored (no click)` : '');
}

// ─── Sample access ───────────────────────────────────────────────────────────
// Samples persisted in earlier sessions are loaded before use, so writes save
// the full set instead of overwriting it with this session only. One shared
// load promise per key: rapid repeat presses must not each load (and then
// overwrite) the stored set independently.
async function getKeySamples(k) {
  if (!state.samples[k]) {
    state.loading[k] ??= Store.loadKey(k);
    const loaded = await state.loading[k];
    state.samples[k] ??= loaded;
  }
  return state.samples[k];
}

async function deleteSample(k, index) {
  const arr = await getKeySamples(k);
  arr.splice(index, 1);
  if (arr.length) {
    state.counts[k] = arr.length;
    await Store.saveKey(k, arr);
  } else {
    delete state.counts[k];
    await Store.removeKey(k);
  }
  await Store.setCounts(state.counts);
  updateKeyboard(state.counts);
  updateRecordStats(state.counts);
}

const samplesTab = initSamplesTab({
  getCounts:  () => state.counts,
  getSamples: getKeySamples,
  deleteSample,
  getSr:      () => Store.getSr(),
  getFilterbank,
  computeMelSpec,
});

// ─── Record ───────────────────────────────────────────────────────────────────
async function recordKey(k, eventTime) {
  const cut = await state.audio.captureKeystroke(eventTime);
  updateCaptureStats(cut);
  if (!cut.audio) { flashKey(k, 'rejected'); return; }
  flashKey(k, 'recording');
  const audio = cut.audio;

  await getKeySamples(k);
  if (state.samples[k].length >= CFG.MAX_SAMPLES) state.samples[k].shift();
  state.samples[k].push(audio);
  state.counts[k] = state.samples[k].length;

  updateKeyboard(state.counts);
  updateRecordStats(state.counts);

  // Persist
  await Store.saveKey(k, state.samples[k]);
  await Store.setCounts(state.counts);
  await Store.saveSr(state.audio.sr);
}

// ─── Predict ─────────────────────────────────────────────────────────────────
async function predictKey(actualKey, eventTime) {
  if (!state.model || !state.micActive) return;
  const cut = await state.audio.captureKeystroke(eventTime);
  updateCaptureStats(cut);
  if (!cut.audio) {
    document.getElementById('pred-status').textContent = 'No keystroke sound detected — press ignored.';
    return;
  }
  updatePredStatus();
  const audio = cut.audio;
  const fb    = getFilterbank(state.audio.sr);
  const spec  = computeMelSpec(audio, fb);
  const input = spec.expandDims(0);               // [1, N_FRAMES, N_MELS, 1]
  const out   = state.model.predict(input);
  const probs = Array.from(out.dataSync());
  input.dispose(); spec.dispose(); out.dispose();

  const predIdx = probs.indexOf(Math.max(...probs));
  const predKey = state.modelKeys[predIdx];

  state.predTotal++;
  if (predKey === actualKey) state.predCorrect++;

  flashKey(actualKey, 'predicting');

  renderConfidence(document.getElementById('conf-bars'), probs, state.modelKeys);
  updatePredDisplay(predKey, actualKey);
  updatePredStats(state.predTotal, state.predCorrect);
  prependHistory(document.getElementById('pred-history'), actualKey, predKey, probs, state.modelKeys);
}

// ─── Train ────────────────────────────────────────────────────────────────────
const btnTrain  = document.getElementById('btn-train');
const btnStop   = document.getElementById('btn-stop-train');
const btnDelete = document.getElementById('btn-delete-model');

// Train is disabled while a run is active; Stop only while one is active;
// Delete whenever a model exists or a run is active.
function updateTrainButtons() {
  const training = !!state.trainAbort;
  btnTrain.disabled  = training;
  btnStop.disabled   = !training;
  btnDelete.disabled = !training && !state.model;
}

btnTrain.addEventListener('click', async () => {
  const abort = new AbortController();
  state.trainAbort = abort;
  updateTrainButtons();
  clearLog();
  state.chartData = { trainAcc: [], valAcc: [] };

  try {
    const result = await trainFromStore(
      msg => log(msg),
      (epoch, logs) => {
        state.chartData.trainAcc.push(logs.acc    ?? logs.accuracy    ?? 0);
        state.chartData.valAcc.push(logs.val_acc ?? logs.val_accuracy ?? 0);
        const c = document.getElementById('train-chart');
        if (c) drawChart(c, state.chartData);
      },
      abort.signal,
    );
    if (result) {
      state.model?.dispose();
      state.model     = result.model;
      state.modelKeys = result.keys;
      setBadge('badge-model', `model: ${result.keys.length} keys`, 'green');
    }
    updatePredStatus();
  } catch (e) {
    log('Error: ' + e.message);
  } finally {
    state.trainAbort = null;
    updateTrainButtons();
  }
});

btnStop.addEventListener('click', () => {
  if (!state.trainAbort) return;
  log('Stopping after the current batch…');
  state.trainAbort.abort();
  btnStop.disabled = true;
});

btnDelete.addEventListener('click', async () => {
  if (!confirm('Delete the trained model? Recorded samples are kept.')) return;
  // A running training would otherwise save a new model right after deletion
  state.trainAbort?.abort();
  state.model?.dispose();
  state.model     = null;
  state.modelKeys = null;
  await deleteSavedModel();
  setBadge('badge-model', 'no model', '');
  log('Model deleted.');
  updateTrainButtons();
  updatePredStatus();
});

// ─── Delete samples ───────────────────────────────────────────────────────────
// Available from both Record and Train tabs. A running training is stopped
// first; the saved model is kept (it has its own Delete button).
async function deleteSamples() {
  if (!confirm('Delete all recorded samples? The trained model is kept.')) return;
  state.trainAbort?.abort();
  await Store.clearAll();
  state.samples = {};
  state.loading = {};
  state.counts  = {};
  updateKeyboard({});
  updateRecordStats({});
  setBadge('badge-data', 'no data', '');
  log('Samples deleted.');
}

document.getElementById('btn-clear').addEventListener('click', deleteSamples);
document.getElementById('btn-delete-samples').addEventListener('click', deleteSamples);

// ─── Reset stats ──────────────────────────────────────────────────────────────
document.getElementById('btn-reset-stats')?.addEventListener('click', () => {
  state.predTotal = state.predCorrect = 0;
  updatePredStats(0, 0);
  const h = document.getElementById('pred-history');
  if (h) h.innerHTML = '';
});

// ─── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  buildKeyboardDOM(KEYBOARD_ROWS, document.getElementById('keyboard'));

  // Restore persisted state
  // Recompute counts from stored buffers (fixes counts that drifted in earlier versions)
  state.counts = await Store.repairCounts();
  updateKeyboard(state.counts);
  updateRecordStats(state.counts);

  const total = Object.values(state.counts).reduce((s, n) => s + n, 0);
  if (total > 0)
    setBadge('badge-data', `${total} samples`, 'green');

  // Try loading saved model
  const saved = await loadSavedModel();
  if (saved) {
    state.model     = saved.model;
    state.modelKeys = saved.keys;
    setBadge('badge-model', `model: ${saved.keys.length} keys`, 'green');
    log(`Restored model for keys: ${saved.keys.join(', ')}`);
  } else {
    setBadge('badge-model', 'no model', '');
  }
  updateTrainButtons();
  updatePredStatus();
}

init();
