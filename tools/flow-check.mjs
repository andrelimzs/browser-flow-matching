// End-to-end check of the flow core on a bimodal target.
//
// Transports uniform 2D samples onto two separated clusters. This is the exact
// property the project exists to demonstrate: a model that regressed the mean
// would put everything between the clusters, so populating both is the test.
import { MLP } from "../src/flow/mlp.js";
import { Adam } from "../src/flow/adam.js";
import { makeFourierEncoder } from "../src/flow/features.js";
import { FlowTrainer, sampleTime, eulerIntegrate } from "../src/flow/flow.js";

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

const random = makeRandom(1234);
const gaussian = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());

const MODES = [[-0.6, 0.0], [0.6, 0.0]];
const SPREAD = 0.07;
const BATCH = 256;
const STEPS = Number(process.env.STEPS ?? 4000);

const encoder = makeFourierEncoder({
  inputWidth: 3, // x, y, t
  bands: [
    { index: 0, frequencies: [1, 2, 4] },
    { index: 1, frequencies: [1, 2, 4] },
    { index: 2, frequencies: [2] },
  ],
});
const model = new MLP({ sizes: [encoder.size, 64, 64, 2], maxBatch: BATCH, random });
const optimizer = new Adam(model.params, { learningRate: 0.003 });
const trainer = new FlowTrainer({ model, optimizer });

const targets = new Float32Array(BATCH * 2);
const raw = new Float32Array(3);
const input = model.inputBuffer();

const started = process.hrtime.bigint();
let loss = 0;
for (let step = 0; step < STEPS; step++) {
  for (let sample = 0; sample < BATCH; sample++) {
    const sourceX = random() * 2 - 1;
    const sourceY = random() * 2 - 1;
    const mode = MODES[random() < 0.5 ? 0 : 1];
    const targetX = mode[0] + gaussian() * SPREAD;
    const targetY = mode[1] + gaussian() * SPREAD;
    const time = sampleTime(random);
    raw[0] = sourceX * (1 - time) + targetX * time;
    raw[1] = sourceY * (1 - time) + targetY * time;
    raw[2] = time;
    encoder.encode(raw, input, sample * encoder.size);
    targets[sample * 2] = targetX - sourceX;
    targets[sample * 2 + 1] = targetY - sourceY;
  }
  loss = trainer.step(targets, BATCH);
}
const trainSeconds = Number(process.hrtime.bigint() - started) / 1e9;

// Sample the learned field.
const single = new MLP({ sizes: model.sizes, maxBatch: 1, random });
single.loadJSON(model.toJSON());
const singleInput = single.inputBuffer();
const velocityAt = (state, time, out) => {
  raw[0] = state[0]; raw[1] = state[1]; raw[2] = time;
  encoder.encode(raw, singleInput, 0);
  const result = single.forward(1);
  out[0] = result[0]; out[1] = result[1];
};

const counts = [0, 0];
let stray = 0;
let distanceSum = 0;
const DRAWS = 1000;
for (let draw = 0; draw < DRAWS; draw++) {
  const state = Float32Array.from([random() * 2 - 1, random() * 2 - 1]);
  eulerIntegrate(velocityAt, state, 20, 2);
  const d0 = Math.hypot(state[0] - MODES[0][0], state[1] - MODES[0][1]);
  const d1 = Math.hypot(state[0] - MODES[1][0], state[1] - MODES[1][1]);
  const nearest = Math.min(d0, d1);
  distanceSum += nearest;
  if (nearest > 0.25) stray += 1;
  else counts[d0 < d1 ? 0 : 1] += 1;
}

const share = counts[0] / Math.max(1, counts[0] + counts[1]);
console.log(`trained ${STEPS} steps x batch ${BATCH} in ${trainSeconds.toFixed(2)}s  (final loss ${loss.toFixed(4)})`);
console.log(`${DRAWS} samples: mode A ${counts[0]}, mode B ${counts[1]}, between/stray ${stray}`);
console.log(`mean distance to nearest mode ${(distanceSum / DRAWS).toFixed(4)}  (cluster spread ${SPREAD})`);
console.log(`mode balance ${(share * 100).toFixed(1)}% / ${((1 - share) * 100).toFixed(1)}%`);

const balanced = share > 0.3 && share < 0.7;
const tight = distanceSum / DRAWS < 0.15;
const clean = stray / DRAWS < 0.1;
console.log(`\nbimodal: ${balanced ? "yes" : "NO"}   on-target: ${tight ? "yes" : "NO"}   few strays: ${clean ? "yes" : "NO"}`);
console.log(balanced && tight && clean ? "PASS" : "FAIL");
process.exit(balanced && tight && clean ? 0 : 1);
