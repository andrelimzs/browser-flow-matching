// Proximal Policy Optimization for the single-frame Push-T RL environment.

import { Adam } from "../../flow/adam.js";
import {
  LOG_2PI,
  LOG_STD_MIN,
  LOG_STD_MAX,
  clamp,
  makeActor,
  makeValue,
  actorSample,
  clipGradients,
  fillRows,
  shuffle,
} from "./common.js";
import { RL_ACTION_SIZE, RL_OBSERVATION_SIZE } from "./env.js";

export function trainPPO({
  env,
  random,
  totalSteps = 200_000,
  rolloutSteps = 2048,
  epochs = 10,
  batchSize = 64,
  width = 128,
  gamma = 0.99,
  gaeLambda = 0.95,
  clipRatio = 0.2,
  entropyCoefficient = 0.02,
  actorLearningRate = 3e-4,
  criticLearningRate = 1e-3,
  onProgress = () => {},
} = {}) {
  const actor = makeActor(RL_OBSERVATION_SIZE, width, batchSize, random);
  const critic = makeValue(RL_OBSERVATION_SIZE, width, batchSize, random);
  const actorOptimizer = new Adam(actor.params, { learningRate: actorLearningRate });
  const criticOptimizer = new Adam(critic.params, { learningRate: criticLearningRate });

  const observations = new Float32Array(rolloutSteps * RL_OBSERVATION_SIZE);
  const actions = new Float32Array(rolloutSteps * RL_ACTION_SIZE);
  const preSquash = new Float32Array(rolloutSteps * RL_ACTION_SIZE);
  const oldLogProbabilities = new Float32Array(rolloutSteps);
  const rewards = new Float32Array(rolloutSteps);
  const dones = new Uint8Array(rolloutSteps);
  const values = new Float32Array(rolloutSteps);
  const advantages = new Float32Array(rolloutSteps);
  const returns = new Float32Array(rolloutSteps);
  const order = new Int32Array(rolloutSteps);
  const actorGradient = new Float32Array(batchSize * 4);
  const criticGradient = new Float32Array(batchSize);
  const action = new Float32Array(2);

  let observation = Float32Array.from(env.reset());
  let steps = 0;
  let episodes = 0;
  let successes = 0;

  while (steps < totalSteps) {
    const count = Math.min(rolloutSteps, totalSteps - steps);
    let rolloutSuccesses = 0;
    for (let index = 0; index < count; index++) {
      observations.set(observation, index * RL_OBSERVATION_SIZE);

      actor.inputBuffer().set(observation, 0);
      const actorOutput = actor.forward(1);
      const sample = actorSample(actorOutput, 0, random);
      action[0] = sample.actionX;
      action[1] = sample.actionY;
      actions[index * 2] = action[0];
      actions[index * 2 + 1] = action[1];
      preSquash[index * 2] = sample.zX;
      preSquash[index * 2 + 1] = sample.zY;
      oldLogProbabilities[index] = sample.logProbability;

      critic.inputBuffer().set(observation, 0);
      values[index] = critic.forward(1)[0];

      const transition = env.step(action);
      rewards[index] = transition.reward;
      dones[index] = transition.done ? 1 : 0;
      observation = Float32Array.from(transition.observation);
      if (transition.done) {
        episodes += 1;
        if (transition.success) { successes += 1; rolloutSuccesses += 1; }
        observation = Float32Array.from(env.reset());
      }
    }

    let bootstrap = 0;
    if (!dones[count - 1]) {
      critic.inputBuffer().set(observation, 0);
      bootstrap = critic.forward(1)[0];
    }
    let gae = 0;
    for (let index = count - 1; index >= 0; index--) {
      const nonterminal = dones[index] ? 0 : 1;
      const nextValue = index === count - 1 ? bootstrap : values[index + 1];
      const delta = rewards[index] + gamma * nextValue * nonterminal - values[index];
      gae = delta + gamma * gaeLambda * nonterminal * gae;
      advantages[index] = gae;
      returns[index] = gae + values[index];
    }

    let advantageMean = 0;
    for (let index = 0; index < count; index++) advantageMean += advantages[index];
    advantageMean /= count;
    let advantageVariance = 0;
    for (let index = 0; index < count; index++) {
      const difference = advantages[index] - advantageMean;
      advantageVariance += difference * difference;
    }
    const advantageScale = 1 / Math.sqrt(advantageVariance / count + 1e-8);
    for (let index = 0; index < count; index++) advantages[index] = (advantages[index] - advantageMean) * advantageScale;

    for (let index = 0; index < count; index++) order[index] = index;
    let policyLoss = 0;
    let valueLoss = 0;
    let updates = 0;
    for (let epoch = 0; epoch < epochs; epoch++) {
      shuffle(order.subarray(0, count), random);
      for (let start = 0; start < count; start += batchSize) {
        const batch = Math.min(batchSize, count - start);
        const indices = order.subarray(start, start + batch);

        fillRows(actor.inputBuffer(), batch, RL_OBSERVATION_SIZE, indices, observations);
        const output = actor.forward(batch);
        actorGradient.fill(0, 0, batch * 4);
        let batchPolicyLoss = 0;
        for (let row = 0; row < batch; row++) {
          const index = indices[row];
          const offset = row * 4;
          const meanX = output[offset];
          const meanY = output[offset + 1];
          const rawLogStdX = output[offset + 2];
          const rawLogStdY = output[offset + 3];
          const logStdX = clamp(rawLogStdX, LOG_STD_MIN, LOG_STD_MAX);
          const logStdY = clamp(rawLogStdY, LOG_STD_MIN, LOG_STD_MAX);
          const zX = preSquash[index * 2];
          const zY = preSquash[index * 2 + 1];
          const inverseVarianceX = Math.exp(-2 * logStdX);
          const inverseVarianceY = Math.exp(-2 * logStdY);
          const differenceX = zX - meanX;
          const differenceY = zY - meanY;
          const normalizedSquaredX = differenceX * differenceX * inverseVarianceX;
          const normalizedSquaredY = differenceY * differenceY * inverseVarianceY;
          const actionX = actions[index * 2];
          const actionY = actions[index * 2 + 1];
          const logProbability =
            -0.5 * (normalizedSquaredX + normalizedSquaredY + 2 * LOG_2PI) - logStdX - logStdY
            - Math.log(1 - actionX * actionX + 1e-6)
            - Math.log(1 - actionY * actionY + 1e-6);
          const ratio = Math.exp(clamp(logProbability - oldLogProbabilities[index], -20, 20));
          const advantage = advantages[index];
          const unclipped = ratio * advantage;
          const clipped = clamp(ratio, 1 - clipRatio, 1 + clipRatio) * advantage;
          batchPolicyLoss -= Math.min(unclipped, clipped) / batch;
          const coefficient = unclipped <= clipped ? (-advantage * ratio) / batch : 0;
          actorGradient[offset] = coefficient * differenceX * inverseVarianceX;
          actorGradient[offset + 1] = coefficient * differenceY * inverseVarianceY;
          if (rawLogStdX > LOG_STD_MIN && rawLogStdX < LOG_STD_MAX) {
            actorGradient[offset + 2] = coefficient * (-1 + normalizedSquaredX) - entropyCoefficient / batch;
          }
          if (rawLogStdY > LOG_STD_MIN && rawLogStdY < LOG_STD_MAX) {
            actorGradient[offset + 3] = coefficient * (-1 + normalizedSquaredY) - entropyCoefficient / batch;
          }
        }
        actor.zeroGrad();
        actor.backward(actorGradient, batch);
        clipGradients(actor, 0.5);
        actorOptimizer.step(actor.grads);

        fillRows(critic.inputBuffer(), batch, RL_OBSERVATION_SIZE, indices, observations);
        const predicted = critic.forward(batch);
        let batchValueLoss = 0;
        for (let row = 0; row < batch; row++) {
          const error = predicted[row] - returns[indices[row]];
          batchValueLoss += 0.5 * error * error / batch;
          criticGradient[row] = error / batch;
        }
        critic.zeroGrad();
        critic.backward(criticGradient, batch);
        clipGradients(critic, 0.5);
        criticOptimizer.step(critic.grads);
        policyLoss += batchPolicyLoss;
        valueLoss += batchValueLoss;
        updates += 1;
      }
    }

    steps += count;
    onProgress({
      algorithm: "ppo",
      steps,
      episodes,
      successes,
      rolloutSuccesses,
      policyLoss: policyLoss / Math.max(1, updates),
      valueLoss: valueLoss / Math.max(1, updates),
    }, { actor, critic });
  }

  return { actor, critic, steps, episodes, successes };
}
