// Proximal Policy Optimization for the single-frame Push-T RL environment.

import { Adam } from "../../flow/adam.js";
import {
  LOG_2PI,
  clamp,
  makePPOActor,
  makeValue,
  ppoActorSample,
  clipGradients,
  fillRows,
  shuffle,
} from "./common.js";
import { RL_ACTION_SIZE, RL_OBSERVATION_SIZE } from "./env.js";

// Generalized advantage estimation over transitions interleaved by environment.
// A terminal transition cuts both the value bootstrap and the recursive trace;
// otherwise the per-environment scratch arrays point at that environment's next
// transition (or the critic bootstrap at the end of the rollout).
export function computeGAE({
  rewards,
  dones,
  values,
  environmentIndices,
  bootstrapValues,
  count = rewards.length,
  gamma = 0.99,
  gaeLambda = 0.95,
  advantages = new Float32Array(count),
  returns = new Float32Array(count),
  nextValues = new Float32Array(bootstrapValues.length),
  nextAdvantages = new Float32Array(bootstrapValues.length),
}) {
  nextValues.set(bootstrapValues);
  nextAdvantages.fill(0);
  for (let index = count - 1; index >= 0; index--) {
    const environment = environmentIndices[index];
    const nonterminal = dones[index] ? 0 : 1;
    const delta = rewards[index] + gamma * nextValues[environment] * nonterminal - values[index];
    nextAdvantages[environment] = delta +
      gamma * gaeLambda * nonterminal * nextAdvantages[environment];
    advantages[index] = nextAdvantages[environment];
    returns[index] = advantages[index] + values[index];
    nextValues[environment] = values[index];
  }
  return { advantages, returns };
}

export function trainPPO({
  env,
  envs,
  random,
  totalSteps = 100_000,
  rolloutSteps = 2048,
  epochs = 10,
  batchSize = 64,
  width = 64,
  gamma = 0.99,
  gaeLambda = 0.95,
  clipRatio = 0.2,
  entropyCoefficient = 0,
  valueCoefficient = 0.5,
  annealLearningRate = true,
  actorLearningRate = 3e-4,
  criticLearningRate = 3e-4,
  progressEvery = 0,
  onCheckpoint = () => {},
  onProgress = () => {},
} = {}) {
  const environments = envs ?? (env ? [env] : []);
  if (!environments.length) throw new Error("PPO needs at least one environment");
  if (environments.length > batchSize) throw new Error("PPO environment count cannot exceed batch size");
  const environmentCount = environments.length;
  const actor = makePPOActor(RL_OBSERVATION_SIZE, width, batchSize, random);
  const critic = makeValue(RL_OBSERVATION_SIZE, width, batchSize, random);
  const actorOptimizer = new Adam(actor.params, { learningRate: actorLearningRate, epsilon: 1e-5 });
  const criticOptimizer = new Adam(critic.params, { learningRate: criticLearningRate, epsilon: 1e-5 });

  const observations = new Float32Array(rolloutSteps * RL_OBSERVATION_SIZE);
  const actions = new Float32Array(rolloutSteps * RL_ACTION_SIZE);
  const oldLogProbabilities = new Float32Array(rolloutSteps);
  const rewards = new Float32Array(rolloutSteps);
  const dones = new Uint8Array(rolloutSteps);
  const values = new Float32Array(rolloutSteps);
  const advantages = new Float32Array(rolloutSteps);
  const returns = new Float32Array(rolloutSteps);
  const order = new Int32Array(rolloutSteps);
  const environmentIndices = new Uint16Array(rolloutSteps);
  const actorGradient = new Float32Array(batchSize * 2);
  const criticGradient = new Float32Array(batchSize);
  const actionsByEnvironment = Array.from({ length: environmentCount }, () => new Float32Array(2));
  const currentObservations = new Float32Array(environmentCount * RL_OBSERVATION_SIZE);
  const bootstrapValues = new Float32Array(environmentCount);
  const nextValues = new Float32Array(environmentCount);
  const nextAdvantages = new Float32Array(environmentCount);

  for (let environment = 0; environment < environmentCount; environment++) {
    environments[environment].setTrainingProgress?.(0);
    currentObservations.set(environments[environment].reset(), environment * RL_OBSERVATION_SIZE);
  }
  let steps = 0;
  let episodes = 0;
  let successes = 0;
  let nextCheckpoint = progressEvery > 0 ? progressEvery : Infinity;

  while (steps < totalSteps) {
    const learningRateFraction = annealLearningRate ? 1 - steps / totalSteps : 1;
    actorOptimizer.learningRate = actorLearningRate * learningRateFraction;
    criticOptimizer.learningRate = criticLearningRate * learningRateFraction;
    const count = Math.min(rolloutSteps, totalSteps - steps);
    let rolloutSuccesses = 0;
    let cursor = 0;
    while (cursor < count) {
      const active = Math.min(environmentCount, count - cursor);
      for (let environment = 0; environment < active; environment++) {
        const observationOffset = environment * RL_OBSERVATION_SIZE;
        actor.inputBuffer().set(
          currentObservations.subarray(observationOffset, observationOffset + RL_OBSERVATION_SIZE),
          observationOffset,
        );
        critic.inputBuffer().set(
          currentObservations.subarray(observationOffset, observationOffset + RL_OBSERVATION_SIZE),
          observationOffset,
        );
      }
      const actorOutput = actor.forward(active);
      const criticOutput = critic.forward(active);
      for (let environment = 0; environment < active; environment++) {
        const index = cursor + environment;
        const observationOffset = environment * RL_OBSERVATION_SIZE;
        observations.set(
          currentObservations.subarray(observationOffset, observationOffset + RL_OBSERVATION_SIZE),
          index * RL_OBSERVATION_SIZE,
        );
        environmentIndices[index] = environment;
        const sample = ppoActorSample(actorOutput, environment * 4, random);
        const action = actionsByEnvironment[environment];
        action[0] = sample.actionX;
        action[1] = sample.actionY;
        actions[index * 2] = action[0];
        actions[index * 2 + 1] = action[1];
        oldLogProbabilities[index] = sample.logProbability;
        values[index] = criticOutput[environment];

        const transition = environments[environment].step(action);
        rewards[index] = transition.reward;
        dones[index] = transition.done ? 1 : 0;
        let nextObservation = transition.observation;
        if (transition.done) {
          environments[environment].setTrainingProgress?.((steps + index + 1) / totalSteps);
          nextObservation = environments[environment].reset();
        }
        currentObservations.set(nextObservation, observationOffset);
        if (transition.done) {
          episodes += 1;
          if (transition.success) { successes += 1; rolloutSuccesses += 1; }
        }
      }
      cursor += active;
      const collectedSteps = steps + cursor;
      while (collectedSteps >= nextCheckpoint) {
        onCheckpoint({
          algorithm: "ppo",
          steps: nextCheckpoint,
          episodes,
          successes,
          rolloutSuccesses,
        }, { actor, critic });
        nextCheckpoint += progressEvery;
      }
    }

    for (let environment = 0; environment < environmentCount; environment++) {
      const observationOffset = environment * RL_OBSERVATION_SIZE;
      critic.inputBuffer().set(
        currentObservations.subarray(observationOffset, observationOffset + RL_OBSERVATION_SIZE),
        observationOffset,
      );
    }
    bootstrapValues.set(critic.forward(environmentCount).subarray(0, environmentCount));
    computeGAE({
      rewards,
      dones,
      values,
      environmentIndices,
      bootstrapValues,
      count,
      gamma,
      gaeLambda,
      advantages,
      returns,
      nextValues,
      nextAdvantages,
    });

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
        actor.zeroGrad();
        actorGradient.fill(0, 0, batch * 2);
        let batchPolicyLoss = 0;
        for (let row = 0; row < batch; row++) {
          const index = indices[row];
          const offset = row * 4;
          const meanX = output[offset];
          const meanY = output[offset + 1];
          const logStdX = output[offset + 2];
          const logStdY = output[offset + 3];
          const actionX = actions[index * 2];
          const actionY = actions[index * 2 + 1];
          const inverseVarianceX = Math.exp(-2 * logStdX);
          const inverseVarianceY = Math.exp(-2 * logStdY);
          const differenceX = actionX - meanX;
          const differenceY = actionY - meanY;
          const normalizedSquaredX = differenceX * differenceX * inverseVarianceX;
          const normalizedSquaredY = differenceY * differenceY * inverseVarianceY;
          const logProbability =
            -0.5 * (normalizedSquaredX + normalizedSquaredY + 2 * LOG_2PI) - logStdX - logStdY;
          const ratio = Math.exp(clamp(logProbability - oldLogProbabilities[index], -20, 20));
          const advantage = advantages[index];
          const unclipped = ratio * advantage;
          const clipped = clamp(ratio, 1 - clipRatio, 1 + clipRatio) * advantage;
          batchPolicyLoss -= Math.min(unclipped, clipped) / batch;
          const coefficient = unclipped <= clipped ? (-advantage * ratio) / batch : 0;
          actorGradient[row * 2] = coefficient * differenceX * inverseVarianceX;
          actorGradient[row * 2 + 1] = coefficient * differenceY * inverseVarianceY;
          actor.logStdGradient[0] += coefficient * (-1 + normalizedSquaredX) - entropyCoefficient / batch;
          actor.logStdGradient[1] += coefficient * (-1 + normalizedSquaredY) - entropyCoefficient / batch;
        }
        actor.backward(actorGradient, batch);

        fillRows(critic.inputBuffer(), batch, RL_OBSERVATION_SIZE, indices, observations);
        const predicted = critic.forward(batch);
        let batchValueLoss = 0;
        for (let row = 0; row < batch; row++) {
          const index = indices[row];
          const target = returns[index];
          const error = predicted[row] - target;
          const valueDelta = predicted[row] - values[index];
          const clippedValue = values[index] + clamp(valueDelta, -clipRatio, clipRatio);
          const clippedError = clippedValue - target;
          const useUnclipped = error * error >= clippedError * clippedError;
          batchValueLoss += 0.5 * Math.max(error * error, clippedError * clippedError) / batch;
          const clipDerivative = Math.abs(valueDelta) <= clipRatio ? 1 : 0;
          criticGradient[row] = valueCoefficient *
            (useUnclipped ? error : clippedError * clipDerivative) / batch;
        }
        critic.zeroGrad();
        critic.backward(criticGradient, batch);
        clipGradients({ grads: [...actor.grads, ...critic.grads] }, 0.5);
        actorOptimizer.step(actor.grads);
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
