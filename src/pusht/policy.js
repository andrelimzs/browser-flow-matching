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
// x, y, lift. Lift cannot be derived from the target: commanding a point past
// the block surface is what pushing *is*, and it is geometrically identical to
// flying over the block. Deriving it made the pusher lift whenever it should
// have pushed, and closed-loop coverage fell from 0.219 to 0.002.
export const ACTION_DIM = 3;
export const CHUNK_WIDTH = CHUNK * ACTION_DIM;

// Actions are predicted relative to the pusher's current position, not as
// absolute arena coordinates. The expert's action sits a mean of 0.053 from the
// pusher against a coordinate range of 1.0, so predicting the absolute position
// spends almost all of the model's output variance re-encoding where the pusher
// already is, and the residual that actually steers is lost in it.
//
// The scale is the standard deviation of that delta, not its extreme: flow
// matching transports a unit Gaussian onto this distribution, and dividing by
// the p99 instead left the target at std 0.29 against a source of std 1.0, so
// the model saw mostly noise. Tails past 1 are fine and expected.
//
// One scalar is not enough once the chunk is long: a step 32 ahead moves three
// times as far from the current pusher as the next one does (rms 0.64 to 2.02
// across the chunk), so each position is normalized by its own statistic and
// the scales travel with the dataset. This is the fallback when none are given.
export const DELTA_SCALE = 0.06;

// The observation is expressed in the block's frame, not the arena's.
//
// Pushing a block toward a goal is the same problem wherever it happens in the
// arena and whatever its global orientation, but absolute coordinates hide that
// from the model, which then has to learn the invariance from data. Measured on
// solved demonstrations, the egocentric encoding explains 53.3% of the action
// variance against 33.9% for the absolute one, from four fewer inputs. The
// actions have to be rotated into the same frame: leaving them in world
// coordinates gives back half the gain (42.3%).
//
// Input from PushWorld.writeObservation is absolute: pusher xy, block xy, block
// cos/sin, goal xy, goal cos/sin, then (x, y, r) per obstacle.
const POSITION_SCALE = 2;

// Raw layout is 11 fixed values (pusher xy, block xy, block cos/sin, goal xy,
// goal cos/sin, lifted) plus 3 per obstacle.
export function egocentricSize(rawSize) {
  return 7 + (rawSize - 11);
}

export function egocentricObservation(raw, out = new Float32Array(egocentricSize(raw.length))) {
  const pusherX = raw[0], pusherY = raw[1];
  const blockX = raw[2], blockY = raw[3];
  const cos = raw[4], sin = raw[5];

  // World offset into the block frame.
  const toBlock = (x, y) => [(x * cos + y * sin) * POSITION_SCALE, (-x * sin + y * cos) * POSITION_SCALE];

  const [pusherRelX, pusherRelY] = toBlock(pusherX - blockX, pusherY - blockY);
  const [goalRelX, goalRelY] = toBlock(raw[6] - blockX, raw[7] - blockY);
  // Goal orientation relative to the block, as cos/sin of the difference.
  const goalCos = raw[8], goalSin = raw[9];
  out[0] = pusherRelX;
  out[1] = pusherRelY;
  out[2] = goalRelX;
  out[3] = goalRelY;
  out[4] = goalCos * cos + goalSin * sin;
  out[5] = goalSin * cos - goalCos * sin;
  out[6] = raw[10];   // whether the pusher is currently lifted

  let cursor = 7;
  for (let index = 11; index < raw.length; index += 3) {
    const [x, y] = toBlock(raw[index] - blockX, raw[index + 1] - blockY);
    out[cursor++] = x;
    out[cursor++] = y;
    out[cursor++] = (raw[index + 2] - 0.05) * 20;
  }
  return out;
}

// Rotates a world-frame offset into the block frame, and back.
export function intoBlockFrame(dx, dy, cos, sin) {
  return [dx * cos + dy * sin, -dx * sin + dy * cos];
}

export function outOfBlockFrame(dx, dy, cos, sin) {
  return [dx * cos - dy * sin, dx * sin + dy * cos];
}

const toDelta = (action, origin, scale) => (action - origin) / scale;
const fromDelta = (value, origin, scale) => origin + value * scale;

// Input is [chunk, observation, time]; only time gets a Fourier lift. The
// planner needed bands on its spatial inputs because it was fitting a thin
// curve; this is a much lower-frequency function of position.
// observationSize here is the egocentric width, i.e. egocentricSize(rawWidth).
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
  const width = egocentricSize(rawWidth);
  const count = usable.reduce((total, episode) => total + episode.observations.length, 0);

  const observations = new Float32Array(count * width);
  const chunks = new Float32Array(count * CHUNK_WIDTH);
  const scratch = new Float32Array(width);
  let cursor = 0;

  // Raw deltas first; the per-position scale is measured from them.
  for (const episode of usable) {
    const length = episode.observations.length;
    for (let index = 0; index < length; index++) {
      const raw = episode.observations[index];
      egocentricObservation(raw, scratch);
      observations.set(scratch, cursor * width);
      // Deltas are taken against the pusher position in this observation, then
      // rotated into the block frame so the target is equivariant too.
      const pusherX = raw[0];
      const pusherY = raw[1];
      const cos = raw[4];
      const sin = raw[5];
      for (let step = 0; step < CHUNK; step++) {
        const source = episode.actions[Math.min(length - 1, index + step)];
        const [dx, dy] = intoBlockFrame(source[0] - pusherX, source[1] - pusherY, cos, sin);
        chunks[cursor * CHUNK_WIDTH + step * ACTION_DIM] = dx;
        chunks[cursor * CHUNK_WIDTH + step * ACTION_DIM + 1] = dy;
        // Lift as +-1, in the same range as the rescaled xy channels.
        chunks[cursor * CHUNK_WIDTH + step * ACTION_DIM + 2] = ((source[2] ?? 0) > 0.5 ? 1 : -1);
      }
      cursor += 1;
    }
  }

  // Only the xy channels are rescaled; lift is already a +-1 flag.
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
  const conditioning = egocentricObservation(observation);
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
    for (let k = 0; k < CHUNK; k++) {
      const base = sample * CHUNK_WIDTH + k * ACTION_DIM;
      const scale = scales ? scales[k] : DELTA_SCALE;
      const [dx, dy] = outOfBlockFrame(states[base] * scale, states[base + 1] * scale, cos, sin);
      points[k * 2] = pusherX + dx;
      points[k * 2 + 1] = pusherY + dy;
    }
    return points;
  }

  // The finished chunk for one candidate, in the form step() consumes.
  function chunk(sample) {
    const out = new Float32Array(CHUNK_WIDTH);
    for (let k = 0; k < CHUNK; k++) {
      const base = sample * CHUNK_WIDTH + k * ACTION_DIM;
      const scale = scales ? scales[k] : DELTA_SCALE;
      const [dx, dy] = outOfBlockFrame(states[base] * scale, states[base + 1] * scale, cos, sin);
      out[k * ACTION_DIM] = pusherX + dx;
      out[k * ACTION_DIM + 1] = pusherY + dy;
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
  const normalized = egocentricObservation(observation);
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
  // Back out of the block frame, then off the pusher position.
  for (let step = 0; step < CHUNK; step++) {
    const scale = scales ? scales[step] : DELTA_SCALE;
    const base = step * ACTION_DIM;
    const [dx, dy] = outOfBlockFrame(state[base] * scale, state[base + 1] * scale, cos, sin);
    state[base] = pusherX + dx;
    state[base + 1] = pusherY + dy;
    // Left as the raw signed value; execution applies hysteresis to it.
  }
  return state;
}
