import { CFG } from './config.js?v=16';

// Client for the Python backend (backend/keymic/api.py).
// Audio goes over the wire as raw little-endian float32 PCM — the bytes of a
// Float32Array — with the sample rate as ?sr=.

export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

async function call(path, { method = 'GET', body, json } = {}) {
  let res;
  try {
    res = await fetch(CFG.BACKEND_URL + path, {
      method,
      body: json !== undefined ? JSON.stringify(json) : body,
      headers: json !== undefined ? { 'Content-Type': 'application/json' }
             : body ? { 'Content-Type': 'application/octet-stream' } : undefined,
    });
  } catch {
    throw new ApiError(0, `backend not reachable at ${CFG.BACKEND_URL} — start it with "python run.py" in backend/`);
  }
  if (!res.ok) {
    let msg = res.statusText;
    try { const d = (await res.json()).detail; msg = typeof d === 'string' ? d : JSON.stringify(d); } catch {}
    throw new ApiError(res.status, msg);
  }
  return res;
}

const json = r => r.json();
const k = key => encodeURIComponent(key);

export const api = {
  health:        ()              => call('/health').then(json),

  samples:       ()              => call('/samples').then(json),
  addSamples:    (key, sr, pcm)  => call(`/samples/${k(key)}?sr=${sr}`, { method: 'POST', body: pcm }).then(json),
  deleteSample:  (key, index)    => call(`/samples/${k(key)}/${index}`, { method: 'DELETE' }).then(json),
  deleteSamples: ()              => call('/samples', { method: 'DELETE' }).then(json),

  // All clips of one key, oldest first, as Float32Array[]
  async keySamples(key) {
    const res = await call(`/samples/${k(key)}`);
    const n = Number(res.headers.get('X-Count'));
    const L = Number(res.headers.get('X-Clip-Length'));
    const all = new Float32Array(await res.arrayBuffer());
    const clips = [];
    for (let i = 0; i < n; i++) clips.push(all.subarray(i * L, (i + 1) * L));
    return clips;
  },

  train:         params => call('/train', { method: 'POST', json: params }).then(json),
  stopTraining:  ()     => call('/train/stop', { method: 'POST' }).then(json),
  trainStatus:   since  => call(`/train/status?since=${since}`).then(json),

  // null when no model has been trained
  async model() {
    try { return await call('/model').then(json); }
    catch (e) { if (e.status === 404) return null; throw e; }
  },
  deleteModel:   ()        => call('/model', { method: 'DELETE' }).then(json),
  predict:       (sr, pcm) => call(`/predict?sr=${sr}`, { method: 'POST', body: pcm }).then(json),
};
