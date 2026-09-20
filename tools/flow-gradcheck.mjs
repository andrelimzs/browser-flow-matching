// Two independent checks of MLP.backward.
//
// 1. Against a naive Float64 reference: no unrolling, no shared code. This is
//    the primary check and is shape-independent, agreeing to ~1e-6.
// 2. Against finite differences, on shallow shapes only. This is what validates
//    the reference itself — a reference mirroring the implementation could share
//    a conceptual error, where FD cannot. It is restricted to shallow shapes
//    because with float32 parameters FD noise grows with depth and swamps the
//    signal well before a real error would.
import { MLP } from "../src/flow/mlp.js";

function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

// Deliberately naive: plain loops, Float64, written from the definitions.
function reference(model, inputs, targets, batch) {
  const sizes = model.sizes;
  const layers = sizes.length - 1;
  const weights = model.weights.map((w) => Float64Array.from(w));
  const biases = model.biases.map((b) => Float64Array.from(b));
  const acts = [Float64Array.from(inputs)];

  for (let l = 0; l < layers; l++) {
    const inN = sizes[l], outN = sizes[l + 1];
    const next = new Float64Array(batch * outN);
    for (let s = 0; s < batch; s++)
      for (let j = 0; j < outN; j++) {
        let sum = biases[l][j];
        for (let i = 0; i < inN; i++) sum += weights[l][j * inN + i] * acts[l][s * inN + i];
        next[s * outN + j] = l === layers - 1 ? sum : Math.tanh(sum);
      }
    acts.push(next);
  }

  const outN = sizes[layers];
  let delta = new Float64Array(batch * outN);
  for (let i = 0; i < batch * outN; i++) delta[i] = (acts[layers][i] - targets[i]) / batch;

  const wGrads = [], bGrads = [];
  for (let l = 0; l < layers; l++) {
    wGrads.push(new Float64Array(sizes[l + 1] * sizes[l]));
    bGrads.push(new Float64Array(sizes[l + 1]));
  }

  for (let l = layers - 1; l >= 0; l--) {
    const inN = sizes[l], outN2 = sizes[l + 1];
    for (let s = 0; s < batch; s++)
      for (let j = 0; j < outN2; j++) {
        const d = delta[s * outN2 + j];
        bGrads[l][j] += d;
        for (let i = 0; i < inN; i++) wGrads[l][j * inN + i] += d * acts[l][s * inN + i];
      }
    if (l === 0) break;
    const prev = new Float64Array(batch * inN);
    for (let s = 0; s < batch; s++) {
      for (let j = 0; j < outN2; j++) {
        const d = delta[s * outN2 + j];
        for (let i = 0; i < inN; i++) prev[s * inN + i] += weights[l][j * inN + i] * d;
      }
      for (let i = 0; i < inN; i++) {
        const a = acts[l][s * inN + i];
        prev[s * inN + i] *= 1 - a * a;
      }
    }
    delta = prev;
  }

  const flat = [];
  for (let l = 0; l < layers; l++) flat.push(wGrads[l], bGrads[l]);
  return flat;
}

const SHAPES = [
  { sizes: [4, 6, 6, 3], batch: 5 },
  { sizes: [3, 5], batch: 4 },            // no hidden layer
  { sizes: [7, 9, 2], batch: 1 },         // single hidden, batch 1
  { sizes: [5, 4, 4, 4, 6], batch: 3 },   // deeper
  { sizes: [13, 8, 11], batch: 8 },       // widths not multiples of 4
  { sizes: [1, 1, 1, 1], batch: 2 },      // width-1 layers
  { sizes: [11, 7, 5, 2], batch: 8 },     // every mod-4 residue
];

let failed = 0;
console.log("against a naive Float64 reference:");
for (const { sizes, batch } of SHAPES) {
  const random = makeRandom(7);
  const model = new MLP({ sizes, maxBatch: batch, random });
  const outWidth = sizes[sizes.length - 1];
  const inputs = Float32Array.from({ length: batch * sizes[0] }, () => random() * 2 - 1);
  const targets = Float32Array.from({ length: batch * outWidth }, () => random() * 2 - 1);

  model.zeroGrad();
  model.inputBuffer().set(inputs);
  const out = model.outputs(model.forward(batch) && batch);
  const gradOutput = new Float32Array(batch * outWidth);
  for (let i = 0; i < gradOutput.length; i++) gradOutput[i] = (out[i] - targets[i]) / batch;
  model.backward(gradOutput, batch);

  const exact = reference(model, inputs, targets, batch);
  // Combined tolerance, |a - b| <= atol + rtol * |b|. A pure relative bound is
  // meaningless on the smallest gradients: these shapes span four orders of
  // magnitude, and float32 storage carries ~3e-8 of absolute precision at the
  // top of that range, so a 1e-5 gradient cannot be compared relatively.
  const atol = 1e-6;
  const rtol = 1e-4;
  let worst = 0;
  let count = 0;
  for (let p = 0; p < model.grads.length; p++)
    for (let i = 0; i < model.grads[p].length; i++) {
      const a = model.grads[p][i], b = exact[p][i];
      const allowed = atol + rtol * Math.abs(b);
      worst = Math.max(worst, Math.abs(a - b) / allowed);
      count += 1;
    }
  const ok = worst < 1;
  if (!ok) failed += 1;
  console.log(`  ${String(sizes.join("x")).padEnd(14)} batch ${batch}  ${String(count).padStart(4)} grads  worst ${(worst * 100).toFixed(1)}% of tolerance  ${ok ? "ok" : "FAIL"}`);
}

// Finite differences validate the reference on shapes where FD is trustworthy.
console.log("\nfinite differences (shallow shapes, validating the reference):");
for (const { sizes, batch } of SHAPES.slice(0, 3)) {
  const random = makeRandom(7);
  const model = new MLP({ sizes, maxBatch: batch, random });
  const outWidth = sizes[sizes.length - 1];
  const inputs = Float32Array.from({ length: batch * sizes[0] }, () => random() * 2 - 1);
  const targets = Float32Array.from({ length: batch * outWidth }, () => random() * 2 - 1);
  const loss = () => {
    model.inputBuffer().set(inputs);
    model.forward(batch);
    const o = model.outputs(batch);
    let total = 0;
    for (let i = 0; i < batch * outWidth; i++) { const e = o[i] - targets[i]; total += 0.5 * e * e; }
    return total / batch;
  };
  model.zeroGrad();
  loss();
  const o = model.outputs(batch);
  const gradOutput = new Float32Array(batch * outWidth);
  for (let i = 0; i < gradOutput.length; i++) gradOutput[i] = (o[i] - targets[i]) / batch;
  model.backward(gradOutput, batch);
  const analytic = model.grads.map((g) => Float32Array.from(g));

  const epsilon = 3e-2;
  let worstLarge = 0, large = 0;
  for (let p = 0; p < model.params.length; p++) {
    const values = model.params[p];
    for (let i = 0; i < values.length; i++) {
      const original = values[i];
      values[i] = original + epsilon; const plus = loss();
      values[i] = original - epsilon; const minus = loss();
      values[i] = original;
      const numeric = (plus - minus) / (2 * epsilon);
      const exact = analytic[p][i];
      if (Math.abs(exact) <= 1e-2) continue;
      large += 1;
      worstLarge = Math.max(worstLarge, Math.abs(numeric - exact) / (Math.abs(numeric) + Math.abs(exact)));
    }
  }
  const ok = worstLarge < 2e-3;
  if (!ok) failed += 1;
  console.log(`  ${String(sizes.join("x")).padEnd(14)} batch ${batch}  ${String(large).padStart(4)} grads>1e-2  worst ${worstLarge.toExponential(2)}  ${ok ? "ok" : "FAIL"}`);
}

console.log(failed ? "\nFAIL" : "\nPASS");
process.exit(failed ? 1 : 0);
