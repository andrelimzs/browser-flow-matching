// Debugging ladder for the behaviour-cloning pipeline, no obstacles.
//
//  rung 0  replay the expert's recorded actions         -> validates the eval harness
//  rung 1  replay them through the chunk encode/decode  -> validates the data path
//  rung 2  MSE on a few episodes, same start poses      -> validates learning end to end
//  rung 3  MSE on many, training then fresh starts      -> generalization
//
// Each rung must pass before the next one means anything.
import { PushWorld, createRandom, SUCCESS_COVERAGE } from "../src/pusht/sim.js";
import { ScriptedExpert } from "../src/pusht/expert.js";
import { buildDataset, egocentricObservation, intoBlockFrame, outOfBlockFrame,
         CHUNK, CHUNK_WIDTH, ACTION_DIM } from "../src/pusht/policy.js";
import { MLP } from "../src/flow/mlp.js";
import { Adam } from "../src/flow/adam.js";

const OBSTACLES = 0;
const CAP = 900;
const EXECUTE = 16;
const random = createRandom(1717);
const world = new PushWorld({ random, obstacleCount: OBSTACLES });
const expert = new ScriptedExpert({ random });

function collect(n) {
  const eps = [];
  for (let e = 0; e < n; e++) {
    world.reset({ obstacleCount: OBSTACLES });
    const initial = world.snapshot();
    expert.reset();
    const observations = [], actions = [];
    for (let s = 0; s < CAP; s++) {
      if (world.coverage() >= SUCCESS_COVERAGE) break;
      observations.push(Array.from(world.writeObservation()));
      const [x, y, lift] = expert.act(world);
      actions.push([x, y, lift]);
      world.step(x, y, lift);
    }
    eps.push({ initial, observations, actions, success: world.coverage() >= SUCCESS_COVERAGE,
               observationSize: observations[0].length });
  }
  return eps;
}

// Runs a closed loop where `chunkFor(world)` supplies a fresh chunk every EXECUTE steps.
function rollout(episodes, chunkFor, label) {
  let wins = 0;
  for (const ep of episodes) {
    world.restore(ep.initial);
    let chunk = null, cursor = 1e9;
    for (let s = 0; s < CAP; s++) {
      if (world.coverage() >= SUCCESS_COVERAGE) break;
      if (cursor >= EXECUTE) { chunk = chunkFor(world, ep, s); cursor = 0; }
      const base = cursor * ACTION_DIM;
      world.step(chunk[base], chunk[base + 1], chunk[base + 2]);
      cursor += 1;
    }
    if (world.coverage() >= SUCCESS_COVERAGE) wins += 1;
  }
  console.log(`${label.padEnd(44)} ${wins}/${episodes.length}`);
  return wins / episodes.length;
}

const train = collect(150);
const solved = train.filter((e) => e.success);
const lengths = solved.map((e) => e.observations.length).sort((a, b) => a - b);
console.log(`collected ${train.length}, solved ${solved.length}, median length ${lengths[lengths.length >> 1]}\n`);

// rung 0: the expert's own actions, indexed by step, through the same loop.
rollout(solved.slice(0, 40), (w, ep, s) => {
  const chunk = new Float32Array(CHUNK_WIDTH);
  for (let k = 0; k < CHUNK; k++) {
    const a = ep.actions[Math.min(ep.actions.length - 1, s + k)];
    chunk[k * ACTION_DIM] = a[0];
    chunk[k * ACTION_DIM + 1] = a[1];
    chunk[k * ACTION_DIM + 2] = a[2];
  }
  return chunk;
}, "rung 0: expert actions, replayed");

// rung 1: same, but round-tripped through the encoding inference uses.
rollout(solved.slice(0, 40), (w, ep, s) => {
  const raw = w.writeObservation();
  const chunk = new Float32Array(CHUNK_WIDTH);
  for (let k = 0; k < CHUNK; k++) {
    const i = Math.min(ep.observations.length - 1, s + k);
    const a = ep.actions[Math.min(ep.actions.length - 1, s + k)];
    const o = ep.observations[Math.min(ep.observations.length - 1, s)];
    const [dx, dy] = intoBlockFrame(a[0] - o[0], a[1] - o[1], o[4], o[5]);
    const [bx, by] = outOfBlockFrame(dx, dy, raw[4], raw[5]);
    chunk[k * ACTION_DIM] = raw[0] + bx;
    chunk[k * ACTION_DIM + 1] = raw[1] + by;
    chunk[k * ACTION_DIM + 2] = a[2];
  }
  return chunk;
}, "rung 1: via chunk encode/decode");

function trainMSE(episodes, steps) {
  const ds = buildDataset(episodes);
  const OBS = ds.observationSize;
  const BATCH = Math.min(256, ds.count);
  const model = new MLP({ sizes: [OBS, 256, 256, CHUNK_WIDTH], maxBatch: BATCH, random });
  const opt = new Adam(model.params, { learningRate: 0.002 });
  const targets = new Float32Array(BATCH * CHUNK_WIDTH), grad = new Float32Array(BATCH * CHUNK_WIDTH);
  const input = model.inputBuffer();
  let loss = 0;
  for (let step = 0; step < steps; step++) {
    for (let b = 0; b < BATCH; b++) {
      const row = Math.floor(random() * ds.count);
      for (let i = 0; i < OBS; i++) input[b * OBS + i] = ds.observations[row * OBS + i];
      for (let i = 0; i < CHUNK_WIDTH; i++) targets[b * CHUNK_WIDTH + i] = ds.chunks[row * CHUNK_WIDTH + i];
    }
    model.zeroGrad(); model.forward(BATCH);
    const out = model.outputs(BATCH);
    loss = 0;
    for (let i = 0; i < BATCH * CHUNK_WIDTH; i++) { const e = out[i] - targets[i]; loss += 0.5 * e * e; grad[i] = e / BATCH; }
    loss /= BATCH;
    model.backward(grad, BATCH); opt.step(model.grads);
  }
  const ego = new Float32Array(OBS);
  const chunkFor = (w) => {
    const raw = w.writeObservation();
    egocentricObservation(raw, ego);
    input.set(ego, 0);
    model.forward(1);
    const o = model.outputs(1);
    const chunk = new Float32Array(CHUNK_WIDTH);
    for (let k = 0; k < CHUNK; k++) {
      const scale = ds.scales[k];
      const [dx, dy] = outOfBlockFrame(o[k * ACTION_DIM] * scale, o[k * ACTION_DIM + 1] * scale, raw[4], raw[5]);
      chunk[k * ACTION_DIM] = raw[0] + dx;
      chunk[k * ACTION_DIM + 1] = raw[1] + dy;
      chunk[k * ACTION_DIM + 2] = o[k * ACTION_DIM + 2] > 0 ? 1 : 0;
    }
    return chunk;
  };
  return { chunkFor, loss, count: ds.count };
}

const few = solved.slice(0, 8);
const fewModel = trainMSE(few, 4000);
console.log(`\n8 episodes, ${fewModel.count} transitions, loss ${fewModel.loss.toFixed(4)}`);
rollout(few, fewModel.chunkFor, "rung 2: MSE on 8 eps, same starts");

const allModel = trainMSE(solved, 8000);
console.log(`\n${solved.length} episodes, ${allModel.count} transitions, loss ${allModel.loss.toFixed(4)}`);
rollout(solved.slice(0, 40), allModel.chunkFor, "rung 3a: MSE all, training starts");
rollout(collect(40).filter((e) => e.success), allModel.chunkFor, "rung 3b: MSE all, fresh starts");
