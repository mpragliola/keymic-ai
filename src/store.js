// Persistence layer backed by IndexedDB via localForage.
//
// Key layout:
//   ks-counts   JSON map { key → sampleCount }
//   ks-sr       number (sample rate)
//   ks-{key}    ArrayBuffer   (concatenated Float32Array samples, raw PCM)
//   ks-model-keys  JSON array of key strings the trained model covers

import { CFG } from './config.js?v=9';

const sampleLength = sr => Math.round(CFG.CAPTURE_MS * sr / 1000);

export const Store = {

  // ─── Counts ──────────────────────────────────────────────────
  async getCounts() {
    const v = await localforage.getItem('ks-counts');
    return v ? JSON.parse(v) : {};
  },

  async setCounts(counts) {
    await localforage.setItem('ks-counts', JSON.stringify(counts));
  },

  // ─── Sample rate ─────────────────────────────────────────────
  async getSr() {
    return await localforage.getItem('ks-sr');
  },

  async saveSr(sr) {
    await localforage.setItem('ks-sr', sr);
  },

  // ─── Per-key raw audio ───────────────────────────────────────
  // Each key stores samples as a single concatenated Float32Array
  // serialised as an ArrayBuffer for efficient IndexedDB storage.
  // Every sample is exactly CAPTURE_MS long at the recording sample rate, so the
  // buffer is split by that fixed length rather than by the stored count. This
  // also recovers buffers whose count drifted out of sync with their contents.
  async loadKey(key) {
    const buf = await localforage.getItem(`ks-${key}`);
    if (!buf) return [];
    const sr = await this.getSr();
    if (!sr) return [];
    const all = new Float32Array(buf);
    const sampLen = sampleLength(sr);
    const n = Math.floor(all.length / sampLen);
    const out = [];
    for (let i = 0; i < n; i++)
      out.push(all.slice(i * sampLen, (i + 1) * sampLen));
    return out;
  },

  // Rewrites ks-counts from the buffers actually stored, so counts always
  // equal the number of samples loadKey will return.
  async repairCounts() {
    const counts = await this.getCounts();
    const fixed = {};
    for (const k of Object.keys(counts)) {
      const n = (await this.loadKey(k)).length;
      if (n > 0) fixed[k] = n;
    }
    await this.setCounts(fixed);
    return fixed;
  },

  async saveKey(key, samples) {
    if (!samples.length) return;
    const total = samples.reduce((s, a) => s + a.length, 0);
    const buf   = new Float32Array(total);
    let off = 0;
    for (const s of samples) { buf.set(s, off); off += s.length; }
    await localforage.setItem(`ks-${key}`, buf.buffer);
  },

  // ─── Model keys ──────────────────────────────────────────────
  async saveModelKeys(keys) {
    await localforage.setItem('ks-model-keys', JSON.stringify(keys));
  },

  async removeModelKeys() {
    await localforage.removeItem('ks-model-keys');
  },

  async loadModelKeys() {
    const v = await localforage.getItem('ks-model-keys');
    return v ? JSON.parse(v) : null;
  },

  // ─── Clear ───────────────────────────────────────────────────
  async clearAll() {
    const counts = await this.getCounts();
    for (const k of Object.keys(counts))
      await localforage.removeItem(`ks-${k}`);
    await localforage.removeItem('ks-counts');
    await localforage.removeItem('ks-sr');
    // Don't clear the model — user might want to re-record without losing it
  },
};
