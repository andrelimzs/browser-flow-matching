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

export const CHUNK = 8;
export const ACTION_DIM = 2;
export const CHUNK_WIDTH = CHUNK * ACTION_DIM;

// Actions are predicted relative to the pusher's current position, not as
// absolute arena coordinates. The expert's action sits a mean of 0.053 from the
// pusher (p99 0.19, max 0.31) against a coordinate range of 1.0, so predicting
// the absolute position spends almost all of the model's output variance
// re-encoding where the pusher already is, and the residual that actually
// steers is lost in it. Dividing by the p99-ish scale puts the target back in
// the same range as the noise it is being transported from.
export const DELTA_SCALE = 0.2;

// Observation layout from PushWorld.writeObservation: pusher xy, block xy,
// block cos/sin, goal xy, goal cos/sin, then (x, y, r) per obstacle. Positions
// live in [0, 1] and are centred; the trig terms are already in [-1, 1]; radii
// are small and get their own scaling.
const POSITION_INDICES = new Set([0, 1, 2, 3, 6, 7]);

export function normalizeObservation(raw, out = new Float32Array(raw.length)) {
  for (let index = 0; index < 10; index++) {
    out[index] = POSITION_INDICES.has(index) ? raw[index] * 2 - 1 : raw[index];
  }
  for (let index = 10; index < raw.length; index += 3) {
    out[index] = raw[index] * 2 - 1;
    out[index + 1] = raw[index + 1] * 2 - 1;
    out[index + 2] = (raw[index + 2] - 0.05) * 20;
  }
  return out;
}

const toDelta = (action, origin) => (action - origin) / DELTA_SCALE;
const fromDelta = (value, origin) => origin + value * DELTA_SCALE;

// Input is [chunk, observation, time]; only time gets a Fourier lift. The
// planner needed bands on its spatial inputs because it was fitting a thin
// curve; this is a much lower-frequency function of position.
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
export function buildDataset(episodes, { observationSize } = {}) {
  const usable = episodes.filter(
    (episode) =>
      episode.observations?.length &&
      (observationSize === undefined || (episode.observationSize ?? episode.observations[0].length) === observationSize),
  );
  if (!usable.length) return null;
  const width = observationSize ?? usable[0].observationSize ?? usable[0].observations[0].length;
  const count = usable.reduce((total, episode) => total + episode.observations.length, 0);

  const observations = new Float32Array(count * width);
  const chunks = new Float32Array(count * CHUNK_WIDTH);
  const scratch = new Float32Array(width);
  let cursor = 0;

  for (const episode of usable) {
    const length = episode.observations.length;
    for (let index = 0; index < length; index++) {
      normalizeObservation(episode.observations[index], scratch);
      observations.set(scratch, cursor * width);
      // Deltas are taken against the pusher position in this observation, which
      // is what the policy also has available at inference time.
      const pusherX = episode.observations[index][0];
      const pusherY = episode.observations[index][1];
      for (let step = 0; step < CHUNK; step++) {
        const source = episode.actions[Math.min(length - 1, index + step)];
        chunks[cursor * CHUNK_WIDTH + step * ACTION_DIM] = toDelta(source[0], pusherX);
        chunks[cursor * CHUNK_WIDTH + step * ACTION_DIM + 1] = toDelta(source[1], pusherY);
      }
      cursor += 1;
    }
  }

  return { observations, chunks, count, observationSize: width, episodes: usable.length };
}

export class PolicyTrainer {
  constructor({ policy, batch = 256, learningRate = 0.002, random = Math.random }) {
    this.policy = policy;
    this.batch = batch;
    this.random = random;
    this.optimizer = new Adam(policy.model.params, { learningRate });
    this.trainer = new FlowTrainer({ model: policy.model, optimizer: this.optimizer });
    this.targets = new Float32Array(batch * CHUNK_WIDTH);
    this.noisy = new Float32Array(CHUNK_WIDTH);
    this.steps = 0;
  }

  gaussian() {
    return Math.sqrt(-2 * Math.log(1 - this.random())) * Math.cos(2 * Math.PI * this.random());
  }

  step(dataset) {
    const { policy, batch, targets } = this;
    const input = policy.model.inputBuffer();
    const width = dataset.observationSize;

    for (let sample = 0; sample < batch; sample++) {
      const row = Math.floor(this.random() * dataset.count);
      const time = sampleTime(this.random);
      const chunkOffset = row * CHUNK_WIDTH;
      for (let index = 0; index < CHUNK_WIDTH; index++) {
        const source = this.gaussian();
        const target = dataset.chunks[chunkOffset + index];
        this.noisy[index] = source * (1 - time) + target * time;
        targets[sample * CHUNK_WIDTH + index] = target - source;
      }
      policy.encode(this.noisy, 0, dataset.observations, row * width, time, input, sample * policy.encoder.size);
    }

    this.steps += 1;
    return this.trainer.step(targets, batch);
  }
}

// Integrates the velocity field from noise to an action chunk, in world
// coordinates. `model` here is a single-sample copy so inference does not
// disturb a training batch in flight.
export function sampleChunk(policy, observation, { steps = 10, random = Math.random, out } = {}) {
  const pusherX = observation[0];
  const pusherY = observation[1];
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
  for (let index = 0; index < CHUNK_WIDTH; index += ACTION_DIM) {
    state[index] = fromDelta(state[index], pusherX);
    state[index + 1] = fromDelta(state[index + 1], pusherY);
  }
  return state;
}
