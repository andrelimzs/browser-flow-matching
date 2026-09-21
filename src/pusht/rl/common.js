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

function orthogonal(values, fanIn, fanOut, gain, random) {
  values.fill(0);
  const previousNormSquared = gain * gain;
  if (fanOut <= fanIn) {
    for (let row = 0; row < fanOut; row++) {
      const offset = row * fanIn;
      for (let column = 0; column < fanIn; column++) values[offset + column] = gaussian(random);
      for (let previous = 0; previous < row; previous++) {
        const previousOffset = previous * fanIn;
        let projection = 0;
        for (let column = 0; column < fanIn; column++) {
          projection += values[offset + column] * values[previousOffset + column];
        }
        projection /= previousNormSquared;
        for (let column = 0; column < fanIn; column++) {
          values[offset + column] -= projection * values[previousOffset + column];
        }
      }
      let norm = 0;
      for (let column = 0; column < fanIn; column++) norm += values[offset + column] ** 2;
      const scale = gain / Math.max(1e-12, Math.sqrt(norm));
      for (let column = 0; column < fanIn; column++) values[offset + column] *= scale;
    }
    return values;
  }

  for (let column = 0; column < fanIn; column++) {
    for (let row = 0; row < fanOut; row++) values[row * fanIn + column] = gaussian(random);
    for (let previous = 0; previous < column; previous++) {
      let projection = 0;
      for (let row = 0; row < fanOut; row++) {
        projection += values[row * fanIn + column] * values[row * fanIn + previous];
      }
      projection /= previousNormSquared;
      for (let row = 0; row < fanOut; row++) {
        values[row * fanIn + column] -= projection * values[row * fanIn + previous];
      }
    }
    let norm = 0;
    for (let row = 0; row < fanOut; row++) norm += values[row * fanIn + column] ** 2;
    const scale = gain / Math.max(1e-12, Math.sqrt(norm));
    for (let row = 0; row < fanOut; row++) values[row * fanIn + column] *= scale;
  }
  return values;
}

function initializeCleanRLPPO(model, outputGain, random) {
  for (let layer = 0; layer < model.layers; layer++) {
    const gain = layer === model.layers - 1 ? outputGain : Math.sqrt(2);
    orthogonal(model.weights[layer], model.sizes[layer], model.sizes[layer + 1], gain, random);
    model.biases[layer].fill(0);
  }
  return model;
}

export function makeActor(observationSize, width, maxBatch, random) {
  const actor = new MLP({ sizes: [observationSize, width, width, 4], maxBatch, random });
  actor.actionSquash = "tanh";
  return actor;
}

export function makePPOActor(observationSize, width, maxBatch, random) {
  const mean = initializeCleanRLPPO(
    new MLP({ sizes: [observationSize, width, width, 2], maxBatch, random }),
    0.01,
    random,
  );
  const logStd = new Float32Array(2);
  const logStdGradient = new Float32Array(2);
  const output = new Float32Array(maxBatch * 4);
  return {
    actionSquash: "clip",
    mean,
    logStd,
    logStdGradient,
    params: [...mean.params, logStd],
    grads: [...mean.grads, logStdGradient],
    inputBuffer: () => mean.inputBuffer(),
    forward(batch) {
      const means = mean.forward(batch);
      for (let row = 0; row < batch; row++) {
        output[row * 4] = means[row * 2];
        output[row * 4 + 1] = means[row * 2 + 1];
        output[row * 4 + 2] = logStd[0];
        output[row * 4 + 3] = logStd[1];
      }
      return output;
    },
    zeroGrad() {
      mean.zeroGrad();
      logStdGradient.fill(0);
    },
    backward(gradient, batch) {
      mean.backward(gradient, batch);
    },
    toJSON() {
      return {
        type: "ppo-normal",
        mean: mean.toJSON(),
        logStd: Array.from(logStd),
      };
    },
  };
}

export function makeValue(observationSize, width, maxBatch, random) {
  return initializeCleanRLPPO(
    new MLP({ sizes: [observationSize, width, width, 1], maxBatch, random }),
    1,
    random,
  );
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

export function ppoActorSample(outputs, offset, random, deterministic = false) {
  const meanX = outputs[offset];
  const meanY = outputs[offset + 1];
  const logStdX = outputs[offset + 2];
  const logStdY = outputs[offset + 3];
  const epsilonX = deterministic ? 0 : gaussian(random);
  const epsilonY = deterministic ? 0 : gaussian(random);
  const actionX = meanX + Math.exp(logStdX) * epsilonX;
  const actionY = meanY + Math.exp(logStdY) * epsilonY;
  const logProbability = deterministic ? 0 :
    -0.5 * (epsilonX * epsilonX + epsilonY * epsilonY + 2 * LOG_2PI) - logStdX - logStdY;
  return {
    actionX,
    actionY,
    zX: actionX,
    zY: actionY,
    epsilonX,
    epsilonY,
    meanX,
    meanY,
    logStdX,
    logStdY,
    rawLogStdX: logStdX,
    rawLogStdY: logStdY,
    logProbability,
  };
}

export function sampleActor(actor, outputs, offset, random, deterministic = false) {
  return actor.actionSquash === "clip"
    ? ppoActorSample(outputs, offset, random, deterministic)
    : actorSample(outputs, offset, random, deterministic);
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
      const sample = sampleActor(actor, output, 0, Math.random, true);
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

export function recordPolicyRollout(actor, env, {
  random = Math.random,
  deterministic = true,
  estimateValue = () => 0,
} = {}) {
  const valuesPerFrame = 12;
  const frames = new Float32Array((env.horizon + 1) * valuesPerFrame);
  const action = new Float32Array(2);
  let observation = env.reset();
  let count = 0;
  let totalReturn = 0;
  let success = false;
  let wallContact = false;
  let stalled = false;
  let coverage = env.world.coverage();

  const recordFrame = (sample, value) => {
    const offset = count * valuesPerFrame;
    frames[offset] = env.world.pusher.x;
    frames[offset + 1] = env.world.pusher.y;
    frames[offset + 2] = env.world.block.x;
    frames[offset + 3] = env.world.block.y;
    frames[offset + 4] = env.world.block.angle;
    frames[offset + 5] = coverage;
    frames[offset + 6] = sample.meanX;
    frames[offset + 7] = sample.meanY;
    frames[offset + 8] = sample.logStdX;
    frames[offset + 9] = sample.logStdY;
    frames[offset + 10] = value;
    frames[offset + 11] = 0;
    count += 1;
  };

  for (;;) {
    actor.inputBuffer().set(observation, 0);
    const output = actor.forward(1);
    const sample = sampleActor(actor, output, 0, random, deterministic);
    recordFrame(sample, estimateValue(observation, sample));
    action[0] = sample.actionX;
    action[1] = sample.actionY;
    const transition = env.step(action);
    frames[(count - 1) * valuesPerFrame + 11] = transition.reward;
    observation = transition.observation;
    totalReturn += transition.reward;
    coverage = transition.coverage;
    success = transition.success;
    wallContact = transition.wallContact;
    stalled = transition.stalled;
    if (transition.done) {
      actor.inputBuffer().set(observation, 0);
      const finalSample = sampleActor(actor, actor.forward(1), 0, random, deterministic);
      recordFrame(finalSample, estimateValue(observation, finalSample));
      break;
    }
  }

  return {
    frames: frames.slice(0, count * valuesPerFrame),
    count,
    return: totalReturn,
    success,
    wallContact,
    stalled,
    coverage,
    squashed: actor.actionSquash !== "clip",
  };
}
