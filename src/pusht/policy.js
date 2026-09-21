// Flow-matching behaviour-cloning policy for the pushing task.
//
// The policy models a distribution over short action chunks conditioned on the
// current observation, rather than a single next action. Chunking is what lets
// a sampled trajectory commit: the decision this task turns on — which way to
// take the block past an obstacle — is only visible over several steps, and a
// policy re-sampling every step would keep re-deciding it.
//
// Training is ordinary conditional flow matching. The source is Gaussian noise
// over the chunk, the target is the expert's actual chunk, and the model
// regresses the velocity along the straight path between them, conditioned on
// the observation and the flow time. Sampling integrates that field from fresh
// noise, which is what lets it land on one mode instead of the average of two.

import { MLP } from "../flow/mlp.js";
import { Adam } from "../flow/adam.js";
import { makeFourierEncoder } from "../flow/features.js";
import { FlowTrainer, sampleTime } from "../flow/flow.js";

// The chunk has to span the decision it is meant to commit to, or resampling
// re-rolls that decision several times inside a single manoeuvre. Taking the
// pusher around the block — the choice this task turns on — is a half orbit,
// about 33 steps at 0.017 per step; the expert's approach phases run a median
// of 15 and a p90 of 52. A chunk of 8 covered a quarter of that and was
// resampled every 4, so a bimodal choice got re-flipped roughly eight times
// per commitment.
export const CHUNK = 32;
// x, y, reserved lift channel. Lift is temporarily disabled, but retaining the
// third channel keeps the ablation easy to reverse without changing the stored
// action shape or every chunk consumer.
export const ACTION_DIM = 3;
export const CHUNK_WIDTH = CHUNK * ACTION_DIM;

// XY actions are first differences: the first target is relative to the current
// pusher and later targets are relative to the preceding command. Each chunk
// position is normalized by its dataset RMS; lift uses -1/+1.
export const DELTA_SCALE = 0.06;

// Raw layout is 11 fixed values (pusher xy, block xy, block cos/sin, goal xy,
// goal cos/sin, lifted) plus 3 per obstacle.
export function normalizedObservationSize(rawSize) {
  return rawSize;
}

export function normalizeObservation(raw, out = new Float32Array(normalizedObservationSize(raw.length))) {
  // Absolute state in [-1, 1]. Positions are affine-mapped from the unit arena;
  // cos/sin already have the right range, and binary lift becomes -1/+1.
  out[0] = raw[0] * 2 - 1;
  out[1] = raw[1] * 2 - 1;
  out[2] = raw[2] * 2 - 1;
  out[3] = raw[3] * 2 - 1;
  out[4] = raw[4];
  out[5] = raw[5];
  out[6] = raw[6] * 2 - 1;
  out[7] = raw[7] * 2 - 1;
  out[8] = raw[8];
  out[9] = raw[9];
  out[10] = raw[10] * 2 - 1;

  let cursor = 11;
  for (let index = 11; index < raw.length; index += 3) {
    out[cursor++] = raw[index] * 2 - 1;
    out[cursor++] = raw[index + 1] * 2 - 1;
    out[cursor++] = (raw[index + 2] - 0.05) * 20;
  }
  return out;
}

export function intoBlockFrame(dx, dy, cos, sin) {
  return [dx * cos + dy * sin, -dx * sin + dy * cos];
}

export function outOfBlockFrame(dx, dy, cos, sin) {
  return [dx * cos - dy * sin, dx * sin + dy * cos];
}

// Input is [chunk, observation, time]; only time gets a Fourier lift. The
// planner needed bands on its spatial inputs because it was fitting a thin
// curve; this is a much lower-frequency function of position.
// observationSize here is the normalized state width.
export function makePolicy({ observationSize, width = 128, maxBatch = 256, random = Math.random }) {
  const rawWidth = CHUNK_WIDTH + observationSize + 1;
  const timeIndex = rawWidth - 1;
  const encoder = makeFourierEncoder({
    inputWidth: rawWidth,
    bands: [{ index: timeIndex, frequencies: [1, 2, 4] }],
  });
  const model = new MLP({ sizes: [encoder.size, width, width, CHUNK_WIDTH], maxBatch, random });
  const raw = new Float32Array(rawWidth);

  // Writes one encoded row into `out` at `offset`.
  function encode(chunk, chunkOffset, observation, observationOffset, time, out, offset) {
    for (let index = 0; index < CHUNK_WIDTH; index++) raw[index] = chunk[chunkOffset + index];
    for (let index = 0; index < observationSize; index++) {
      raw[CHUNK_WIDTH + index] = observation[observationOffset + index];
    }
    raw[timeIndex] = time;
    encoder.encode(raw, out, offset);
  }

  return { model, encoder, encode, observationSize, rawWidth, width };
}

// Flattens episodes into normalized observations and the action chunk that
// follows each one. Chunks that run past the end of an episode repeat its last
// action, which is what the expert effectively does once it has stopped.
// Failed episodes are excluded by default, and this matters far more than the
// episode count suggests: a failure runs to the step cap while a success takes
// a few hundred steps, so at a 92% expert success rate the 8% of episodes that
// failed contribute 26% of the transitions — and every one of those steps is the
// expert stuck, which is exactly the behaviour a policy must not learn.
export function buildDataset(episodes, { observationSize, includeFailures = false } = {}) {
  const usable = episodes.filter(
    (episode) =>
      episode.observations?.length &&
      (includeFailures || episode.success !== false) &&
      (observationSize === undefined || (episode.observationSize ?? episode.observations[0].length) === observationSize),
  );
  if (!usable.length) return null;
  const rawWidth = observationSize ?? usable[0].observationSize ?? usable[0].observations[0].length;
  const width = normalizedObservationSize(rawWidth);
  const count = usable.reduce((total, episode) => total + episode.observations.length, 0);

  const observations = new Float32Array(count * width);
  const chunks = new Float32Array(count * CHUNK_WIDTH);
  const scratch = new Float32Array(width);
  let cursor = 0;

  for (const episode of usable) {
    const length = episode.observations.length;
    for (let index = 0; index < length; index++) {
      const raw = episode.observations[index];
      normalizeObservation(raw, scratch);
      observations.set(scratch, cursor * width);
      let previousX = raw[0];
      let previousY = raw[1];
      const cos = raw[4];
      const sin = raw[5];
      for (let step = 0; step < CHUNK; step++) {
        const source = episode.actions[Math.min(length - 1, index + step)];
        const [dx, dy] = intoBlockFrame(source[0] - previousX, source[1] - previousY, cos, sin);
        chunks[cursor * CHUNK_WIDTH + step * ACTION_DIM] = dx;
        chunks[cursor * CHUNK_WIDTH + step * ACTION_DIM + 1] = dy;
        chunks[cursor * CHUNK_WIDTH + step * ACTION_DIM + 2] = -1;
        previousX = source[0];
        previousY = source[1];
      }
      cursor += 1;
    }
  }

  const scales = new Float32Array(CHUNK);
  for (let step = 0; step < CHUNK; step++) {
    let total = 0;
    for (let row = 0; row < count; row++) {
      const base = row * CHUNK_WIDTH + step * ACTION_DIM;
      total += chunks[base] * chunks[base] + chunks[base + 1] * chunks[base + 1];
    }
    scales[step] = Math.max(1e-4, Math.sqrt(total / (count * 2)));
  }
  for (let row = 0; row < count; row++) {
    for (let step = 0; step < CHUNK; step++) {
      const base = row * CHUNK_WIDTH + step * ACTION_DIM;
      chunks[base] /= scales[step];
      chunks[base + 1] /= scales[step];
    }
  }

  return { observations, chunks, scales, count, observationSize: width, episodes: usable.length };
}

export class PolicyTrainer {
  constructor({
    policy,
    batch = 256,
    learningRate = 0.002,
    random = Math.random,
    stateNoise = 0,
    actionNoise = 0,
  }) {
    this.policy = policy;
    this.batch = batch;
    this.random = random;
    this.stateNoise = stateNoise;
    this.actionNoise = actionNoise;
    this.optimizer = new Adam(policy.model.params, { learningRate });
    this.trainer = new FlowTrainer({ model: policy.model, optimizer: this.optimizer });
    this.targets = new Float32Array(batch * CHUNK_WIDTH);
    this.noisy = new Float32Array(CHUNK_WIDTH);
    this.conditioning = new Float32Array(batch * policy.observationSize);
    this.steps = 0;
  }

  gaussian() {
    return Math.sqrt(-2 * Math.log(1 - this.random())) * Math.cos(2 * Math.PI * this.random());
  }

  step(dataset) {
    const { policy, batch, targets, conditioning } = this;
    const input = policy.model.inputBuffer();
    const width = dataset.observationSize;

    for (let sample = 0; sample < batch; sample++) {
      const row = Math.floor(this.random() * dataset.count);
      const time = sampleTime(this.random);
      const chunkOffset = row * CHUNK_WIDTH;
      const observationOffset = row * width;
      const noisyObservationOffset = sample * width;
      for (let index = 0; index < width; index++) {
        const value = dataset.observations[observationOffset + index] + this.gaussian() * this.stateNoise;
        conditioning[noisyObservationOffset + index] = Math.max(-1, Math.min(1, value));
      }
      for (let index = 0; index < CHUNK_WIDTH; index++) {
        const source = this.gaussian();
        const channel = index % ACTION_DIM;
        const clean = dataset.chunks[chunkOffset + index];
        const step = Math.floor(index / ACTION_DIM);
        const target = channel < 2
          ? clean + this.gaussian() * this.actionNoise / dataset.scales[step]
          : clean;
        this.noisy[index] = source * (1 - time) + target * time;
        targets[sample * CHUNK_WIDTH + index] = target - source;
      }
      policy.encode(this.noisy, 0, conditioning, noisyObservationOffset, time, input, sample * policy.encoder.size);
    }

    this.steps += 1;
    return this.trainer.step(targets, batch);
  }
}

// A sampler that exposes the integration one Euler step at a time, so the
// transport from noise to a trajectory can be animated. Several candidates are
// carried at once: at t=0 they are independent noise, and by t=1 they have
// collapsed onto the chunks the policy considers plausible here, which is the
// distribution the whole method exists to represent.
export function createFlowSampler(policy, observation, { count = 12, steps = 10, random = Math.random, scales } = {}) {
  const pusherX = observation[0];
  const pusherY = observation[1];
  const cos = observation[4];
  const sin = observation[5];
  const conditioning = normalizeObservation(observation);
  const gaussian = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());

  const states = new Float32Array(count * CHUNK_WIDTH);
  for (let index = 0; index < states.length; index++) states[index] = gaussian();

  let step = 0;
  const delta = 1 / steps;
  const input = policy.model.inputBuffer();

  function advance() {
    if (step >= steps) return true;
    for (let sample = 0; sample < count; sample++) {
      policy.encode(states, sample * CHUNK_WIDTH, conditioning, 0, step * delta, input, sample * policy.encoder.size);
    }
    policy.model.forward(count);
    const velocity = policy.model.outputs(count);
    for (let index = 0; index < count * CHUNK_WIDTH; index++) states[index] += velocity[index] * delta;
    step += 1;
    return step >= steps;
  }

  // One candidate as a world-space polyline, valid at any point during the
  // integration, so partially-transported noise can be drawn too.
  function polyline(sample, out) {
    const points = out ?? new Float32Array(CHUNK * 2);
    let targetX = pusherX;
    let targetY = pusherY;
    for (let k = 0; k < CHUNK; k++) {
      const base = sample * CHUNK_WIDTH + k * ACTION_DIM;
      const scale = scales ? scales[k] : DELTA_SCALE;
      const [dx, dy] = outOfBlockFrame(states[base] * scale, states[base + 1] * scale, cos, sin);
      targetX += dx;
      targetY += dy;
      points[k * 2] = targetX;
      points[k * 2 + 1] = targetY;
    }
    return points;
  }

  // The finished chunk for one candidate, in the form step() consumes.
  function chunk(sample) {
    const out = new Float32Array(CHUNK_WIDTH);
    let targetX = pusherX;
    let targetY = pusherY;
    for (let k = 0; k < CHUNK; k++) {
      const base = sample * CHUNK_WIDTH + k * ACTION_DIM;
      const scale = scales ? scales[k] : DELTA_SCALE;
      const [dx, dy] = outOfBlockFrame(states[base] * scale, states[base + 1] * scale, cos, sin);
      targetX += dx;
      targetY += dy;
      out[k * ACTION_DIM] = targetX;
      out[k * ACTION_DIM + 1] = targetY;
      out[k * ACTION_DIM + 2] = states[base + 2];
    }
    return out;
  }

  return { advance, polyline, chunk, count, get step() { return step; }, steps };
}

// Integrates the velocity field from noise to an action chunk, in world
// coordinates. `model` here is a single-sample copy so inference does not
// disturb a training batch in flight.
export function sampleChunk(policy, observation, { steps = 10, random = Math.random, out, scales } = {}) {
  const pusherX = observation[0];
  const pusherY = observation[1];
  const cos = observation[4];
  const sin = observation[5];
  const normalized = normalizeObservation(observation);
  const state = out ?? new Float32Array(CHUNK_WIDTH);
  const gaussian = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
  for (let index = 0; index < CHUNK_WIDTH; index++) state[index] = gaussian();

  const input = policy.model.inputBuffer();
  const delta = 1 / steps;
  for (let step = 0; step < steps; step++) {
    policy.encode(state, 0, normalized, 0, step * delta, input, 0);
    const velocity = policy.model.forward(1);
    for (let index = 0; index < CHUNK_WIDTH; index++) state[index] += velocity[index] * delta;
  }
  let targetX = pusherX;
  let targetY = pusherY;
  for (let step = 0; step < CHUNK; step++) {
    const scale = scales ? scales[step] : DELTA_SCALE;
    const base = step * ACTION_DIM;
    const [dx, dy] = outOfBlockFrame(state[base] * scale, state[base + 1] * scale, cos, sin);
    targetX += dx;
    targetY += dy;
    state[base] = targetX;
    state[base + 1] = targetY;
  }
  return state;
}
