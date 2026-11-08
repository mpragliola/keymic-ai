import { CFG } from './config.js?v=14';
import { getFilterbank, computeMelSpec } from './features.js?v=14';
import { Store } from './store.js?v=14';

// ─── Build the CNN ────────────────────────────────────────────────────────────
// Architecture:
//   Input  [N_FRAMES=64, N_MELS=40, 1]
//   Block 1: Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
//   Block 2: Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
//   Block 3: Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
//   GlobalAveragePooling2D
//   Dense(128) → ReLU → Dropout(0.4)
//   Dense(nClasses) → Softmax
export function buildModel(nClasses) {
  const inp = tf.input({ shape: [CFG.N_FRAMES, CFG.N_MELS, 1] });

  const convBlock = (x, filters) => {
    let h = tf.layers.conv2d({ filters, kernelSize: 3, padding: 'same',
                               activation: 'linear', useBias: false }).apply(x);
    h = tf.layers.batchNormalization().apply(h);
    h = tf.layers.activation({ activation: 'relu' }).apply(h);
    h = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(h);
    return h;
  };

  let h = convBlock(inp, 32);
  h = convBlock(h, 64);
  h = convBlock(h, 128);
  h = tf.layers.globalAveragePooling2d({}).apply(h);
  h = tf.layers.dense({ units: 128, activation: 'relu' }).apply(h);
  h = tf.layers.dropout({ rate: 0.4 }).apply(h);
  const out = tf.layers.dense({ units: nClasses, activation: 'softmax' }).apply(h);

  const model = tf.model({ inputs: inp, outputs: out });
  model.compile({
    optimizer: tf.train.adam(CFG.LR),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

// ─── Train from persisted data ───────────────────────────────────────────────
// Loads all recorded samples from IndexedDB, converts each to a mel spectrogram,
// stacks them into a tensor dataset, and trains a fresh CNN.
// After training, saves the model back to IndexedDB.
//
// Callbacks:
//   onLog(msg)            → append a line to the training log
//   onProgress(ep, logs)  → update epoch progress bar / chart
//
// signal (AbortSignal): aborting it stops the run at the next batch boundary.
// A stopped run is discarded — nothing is saved — and the function returns null.
export async function trainFromStore(onLog, onProgress, signal) {
  onLog('Loading samples from IndexedDB…');

  const counts  = await Store.repairCounts();
  const sr      = await Store.getSr();
  const fb      = getFilterbank(sr);

  const eligibleKeys = Object.entries(counts)
    .filter(([, n]) => n >= CFG.MIN_SAMPLES)
    .map(([k]) => k)
    .sort();

  if (eligibleKeys.length < CFG.MIN_KEYS) {
    throw new Error(`Need at least ${CFG.MIN_KEYS} keys with ${CFG.MIN_SAMPLES}+ samples each.`);
  }

  onLog(`Training on keys: ${eligibleKeys.join(', ')}`);

  const specs  = [];
  const labels = [];
  let xs = null, ys = null, xv = null, yv = null, model = null;

  try {
    for (let ci = 0; ci < eligibleKeys.length; ci++) {
      if (signal?.aborted) return stopped(onLog);
      const key     = eligibleKeys[ci];
      const samples = await Store.loadKey(key);
      onLog(`  ${key}: ${samples.length} samples`);
      for (const audio of samples) {
        specs.push(computeMelSpec(audio, fb));  // [N_FRAMES, N_MELS, 1]
        labels.push(ci);
      }
      // Yield to the event loop so a Stop click is handled during preprocessing
      await tf.nextFrame();
    }

    // Stratified split: hold out VAL_SPLIT of each key's samples for validation.
    // (fit's validationSplit takes the last rows before shuffling; since rows are
    // grouped by key, that would validate only on keys never seen in training.)
    const byKey = new Map();
    labels.forEach((c, i) => { if (!byKey.has(c)) byKey.set(c, []); byKey.get(c).push(i); });
    const trainIdx = [], valIdx = [];
    for (const idx of byKey.values()) {
      shuffleInPlace(idx);
      const nVal = Math.max(1, Math.round(idx.length * CFG.VAL_SPLIT));
      valIdx.push(...idx.slice(0, nVal));
      trainIdx.push(...idx.slice(nVal));
    }
    shuffleInPlace(trainIdx);

    const pick = idx => tf.stack(idx.map(i => specs[i]));
    // float32, not int32: TF.js's sparse accuracy metric applies floor() to labels,
    // which rejects int32 tensors.
    const pickY = idx => tf.tensor1d(idx.map(i => labels[i]), 'float32');
    xs = pick(trainIdx);  ys = pickY(trainIdx);
    xv = pick(valIdx);    yv = pickY(valIdx);

    onLog(`Total samples: ${labels.length} (${trainIdx.length} train / ${valIdx.length} validation) — building model…`);
    model = buildModel(eligibleKeys.length);

    await model.fit(xs, ys, {
      epochs:          CFG.EPOCHS,
      batchSize:       CFG.BATCH_SIZE,
      validationData:  [xv, yv],
      shuffle:         true,
      callbacks: {
        // stopTraining is checked by fit() after each batch
        onBatchEnd: () => { if (signal?.aborted) model.stopTraining = true; },
        onEpochEnd: (epoch, logs) => {
          if (signal?.aborted) return;
          onLog(`Epoch ${epoch + 1}/${CFG.EPOCHS}  loss=${logs.loss.toFixed(4)}  acc=${(logs.acc ?? logs.accuracy ?? 0).toFixed(3)}  val_acc=${(logs.val_acc ?? logs.val_accuracy ?? 0).toFixed(3)}`);
          onProgress(epoch + 1, logs);
        },
      },
    });

    if (signal?.aborted) { model.dispose(); return stopped(onLog); }

    onLog('Saving model to IndexedDB…');
    await model.save('indexeddb://keystroke-cnn');
    await Store.saveModelKeys(eligibleKeys);

    // Abort may arrive while saving (e.g. Delete model clicked) — undo the save
    if (signal?.aborted) {
      await deleteSavedModel();
      model.dispose();
      return stopped(onLog);
    }

    onLog('Done ✓');
    return { model, keys: eligibleKeys };
  } finally {
    specs.forEach(t => t.dispose());
    xs?.dispose(); ys?.dispose();
    xv?.dispose(); yv?.dispose();
  }
}

// Fisher–Yates
function shuffleInPlace(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function stopped(onLog) {
  onLog('Training stopped — nothing saved.');
  return null;
}

// ─── Delete saved model ───────────────────────────────────────────────────────
// Removes the trained weights and the key list; recorded samples are kept.
export async function deleteSavedModel() {
  try { await tf.io.removeModel('indexeddb://keystroke-cnn'); } catch { /* none saved */ }
  await Store.removeModelKeys();
}

// ─── Load saved model ─────────────────────────────────────────────────────────
export async function loadSavedModel() {
  try {
    const model = await tf.loadLayersModel('indexeddb://keystroke-cnn');
    const keys  = await Store.loadModelKeys();
    if (!keys) return null;
    return { model, keys };
  } catch {
    return null;
  }
}
