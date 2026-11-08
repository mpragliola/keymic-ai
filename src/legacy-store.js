import { CFG } from './config.js?v=15';

// Read access to samples recorded before the backend existed, when they were
// kept in IndexedDB (localForage keys: ks-counts, ks-sr, ks-{key}). Used only
// to upload them to the backend once, then clear them.
export const LegacyStore = {
  async getCounts() {
    const v = await localforage.getItem('ks-counts');
    return v ? JSON.parse(v) : {};
  },

  async getSr() {
    return await localforage.getItem('ks-sr');
  },

  // Clips of one key; each is exactly CAPTURE_MS long at the stored sample rate
  async loadKey(key) {
    const buf = await localforage.getItem(`ks-${key}`);
    const sr = await this.getSr();
    if (!buf || !sr) return [];
    const all = new Float32Array(buf);
    const L = Math.round(CFG.CAPTURE_MS * sr / 1000);
    const out = [];
    for (let i = 0; i + L <= all.length; i += L) out.push(all.subarray(i, i + L));
    return out;
  },

  // Removes all browser-side samples and the old in-browser model
  async clearAll() {
    for (const key of await localforage.keys())
      if (key.startsWith('ks-')) await localforage.removeItem(key);
    try { await tf.io.removeModel('indexeddb://keystroke-cnn'); } catch { /* none saved */ }
  },
};
