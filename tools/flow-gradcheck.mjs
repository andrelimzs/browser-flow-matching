// Finite-difference check of MLP.backward against the loss it claims to
// differentiate. Buffers are Float32, so the achievable agreement is limited by
// float32 rounding rather than by the maths; 1e-2 relative is the practical bar.
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

const random = makeRandom(7);
const batch = 5;
const sizes = [4, 6, 6, 3];
const model = new MLP({ sizes, maxBatch: batch, random });

const inputs = Float32Array.from({ length: batch * sizes[0] }, () => random() * 2 - 1);
const targets = Float32Array.from({ length: batch * sizes[3] }, () => random() * 2 - 1);

function loss() {
  model.inputBuffer().set(inputs);
  const out = model.forward(batch);
  let total = 0;
  for (let i = 0; i < batch * sizes[3]; i++) {
    const error = out[i] - targets[i];
    total += 0.5 * error * error;
  }
  return total / batch;
}

// Analytic gradients.
model.zeroGrad();
model.inputBuffer().set(inputs);
const out = model.forward(batch);
const gradOutput = new Float32Array(batch * sizes[3]);
for (let i = 0; i < gradOutput.length; i++) gradOutput[i] = (out[i] - targets[i]) / batch;
model.backward(gradOutput, batch);
const analytic = model.grads.map((g) => Float32Array.from(g));

// Numeric gradients over every parameter.
let worst = 0;
let worstWhere = "";
let checked = 0;
// Tracked separately: on gradients large enough that float32 rounding is not
// the dominant term, agreement should be far tighter than the overall bar.
let worstLarge = 0;
let largeCount = 0;
const epsilon = 3e-2;
for (let p = 0; p < model.params.length; p++) {
  const values = model.params[p];
  for (let i = 0; i < values.length; i++) {
    const original = values[i];
    values[i] = original + epsilon;
    const plus = loss();
    values[i] = original - epsilon;
    const minus = loss();
    values[i] = original;
    const numeric = (plus - minus) / (2 * epsilon);
    const exact = analytic[p][i];
    const scale = Math.max(1e-4, Math.abs(numeric) + Math.abs(exact));
    const relative = Math.abs(numeric - exact) / scale;
    checked += 1;
    if (relative > worst) { worst = relative; worstWhere = `param ${p}[${i}] analytic ${exact.toExponential(3)} numeric ${numeric.toExponential(3)}`; }
    if (Math.abs(exact) > 1e-2) { largeCount += 1; worstLarge = Math.max(worstLarge, relative); }
  }
}

console.log(`checked ${checked} parameters across ${sizes.length - 1} layers, batch ${batch}`);
console.log(`worst relative error ${worst.toExponential(3)}`);
console.log(`  at ${worstWhere}`);
console.log(`worst among the ${largeCount} gradients above 1e-2: ${worstLarge.toExponential(3)}`);
const pass = worst < 1e-2 && worstLarge < 1e-3;
console.log(pass ? "PASS" : "FAIL");
process.exit(pass ? 0 : 1);
