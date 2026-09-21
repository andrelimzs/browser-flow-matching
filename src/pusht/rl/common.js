import { MLP } from "../../flow/mlp.js";

export const LOG_2PI = Math.log(2 * Math.PI);
export const LOG_STD_MIN = -5;
export const LOG_STD_MAX = 1;

export function gaussian(random) {
  return Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
}

export function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

export function makeActor(observationSize, width, maxBatch, random) {
  return new MLP({ sizes: [observationSize, width, width, 4], maxBatch, random });
}

export function makeValue(observationSize, width, maxBatch, random) {
  return new MLP({ sizes: [observationSize, width, width, 1], maxBatch, random });
}

export function makeCritic(observationSize, width, maxBatch, random) {
  return new MLP({ sizes: [observationSize + 2, width, width, 1], maxBatch, random });
}

export function actorSample(outputs, offset, random, deterministic = false) {
  const meanX = outputs[offset];
  const meanY = outputs[offset + 1];
  const rawLogStdX = outputs[offset + 2];
  const rawLogStdY = outputs[offset + 3];
  const logStdX = clamp(rawLogStdX, LOG_STD_MIN, LOG_STD_MAX);
  const logStdY = clamp(rawLogStdY, LOG_STD_MIN, LOG_STD_MAX);
  const epsilonX = deterministic ? 0 : gaussian(random);
  const epsilonY = deterministic ? 0 : gaussian(random);
  const zX = meanX + Math.exp(logStdX) * epsilonX;
  const zY = meanY + Math.exp(logStdY) * epsilonY;
  const actionX = Math.tanh(zX);
  const actionY = Math.tanh(zY);
  const logProbability = deterministic ? 0 :
    -0.5 * (epsilonX * epsilonX + epsilonY * epsilonY + 2 * LOG_2PI) - logStdX - logStdY
    - Math.log(1 - actionX * actionX + 1e-6)
    - Math.log(1 - actionY * actionY + 1e-6);
  return {
    actionX,
    actionY,
    zX,
    zY,
    epsilonX,
    epsilonY,
    meanX,
    meanY,
    logStdX,
    logStdY,
    rawLogStdX,
    rawLogStdY,
    logProbability,
  };
}

export function copyParameters(target, source) {
  for (let index = 0; index < target.params.length; index++) target.params[index].set(source.params[index]);
}

export function softUpdate(target, source, amount) {
  for (let group = 0; group < target.params.length; group++) {
    const to = target.params[group];
    const from = source.params[group];
    for (let index = 0; index < to.length; index++) to[index] += amount * (from[index] - to[index]);
  }
}

export function clipGradients(model, maximum = 1) {
  let squared = 0;
  for (const gradient of model.grads) {
    for (let index = 0; index < gradient.length; index++) squared += gradient[index] * gradient[index];
  }
  const norm = Math.sqrt(squared);
  if (norm > maximum) {
    const scale = maximum / norm;
    for (const gradient of model.grads) {
      for (let index = 0; index < gradient.length; index++) gradient[index] *= scale;
    }
  }
  return norm;
}

export function fillRows(buffer, rows, width, indices, source) {
  for (let row = 0; row < rows; row++) {
    const sourceRow = indices ? indices[row] : row;
    buffer.set(source.subarray(sourceRow * width, (sourceRow + 1) * width), row * width);
  }
}

export function shuffle(values, random) {
  for (let index = values.length - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1));
    const value = values[index];
    values[index] = values[other];
    values[other] = value;
  }
}

export function evaluatePolicy(actor, env, episodes = 5) {
  let successes = 0;
  let coverage = 0;
  const action = new Float32Array(2);
  for (let episode = 0; episode < episodes; episode++) {
    let observation = env.reset();
    for (;;) {
      actor.inputBuffer().set(observation, 0);
      const output = actor.forward(1);
      const sample = actorSample(output, 0, Math.random, true);
      action[0] = sample.actionX;
      action[1] = sample.actionY;
      const transition = env.step(action);
      observation = transition.observation;
      if (transition.done) {
        if (transition.success) successes += 1;
        coverage += transition.coverage;
        break;
      }
    }
  }
  return { successes, episodes, meanCoverage: coverage / episodes };
}

export function recordPolicyRollout(actor, env) {
  const valuesPerFrame = 6;
  const frames = new Float32Array((env.horizon + 1) * valuesPerFrame);
  const action = new Float32Array(2);
  let observation = env.reset();
  let count = 0;
  let totalReturn = 0;
  let success = false;
  let coverage = env.world.coverage();

  const recordFrame = () => {
    const offset = count * valuesPerFrame;
    frames[offset] = env.world.pusher.x;
    frames[offset + 1] = env.world.pusher.y;
    frames[offset + 2] = env.world.block.x;
    frames[offset + 3] = env.world.block.y;
    frames[offset + 4] = env.world.block.angle;
    frames[offset + 5] = coverage;
    count += 1;
  };

  recordFrame();
  for (;;) {
    actor.inputBuffer().set(observation, 0);
    const output = actor.forward(1);
    const sample = actorSample(output, 0, Math.random, true);
    action[0] = sample.actionX;
    action[1] = sample.actionY;
    const transition = env.step(action);
    observation = transition.observation;
    totalReturn += transition.reward;
    coverage = transition.coverage;
    success = transition.success;
    recordFrame();
    if (transition.done) break;
  }

  return {
    frames: frames.slice(0, count * valuesPerFrame),
    count,
    return: totalReturn,
    success,
    coverage,
  };
}
