// Group Relative Policy Optimization for the single-frame Push-T environment.
// Each update compares complete stochastic rollouts from the same initial
// condition and assigns every transition the normalized return of its rollout.

import { Adam } from "../../flow/adam.js";
import {
  LOG_2PI,
  clamp,
  makePPOActor,
  ppoActorSample,
  clipGradients,
  fillRows,
  shuffle,
} from "./common.js";
import { RL_ACTION_SIZE, RL_OBSERVATION_SIZE } from "./env.js";

export function normalizeGroupAdvantages(returns, out = new Float32Array(returns.length)) {
  let mean = 0;
  for (const value of returns) mean += value;
  mean /= Math.max(1, returns.length);
  let variance = 0;
  for (const value of returns) variance += (value - mean) ** 2;
  const standardDeviation = Math.sqrt(variance / Math.max(1, returns.length));
  const scale = standardDeviation > 1e-8 ? 1 / standardDeviation : 0;
  for (let index = 0; index < returns.length; index++) {
    out[index] = (returns[index] - mean) * scale;
  }
  return { advantages: out, mean, standardDeviation };
}

export function trainGRPO({
  env,
  envs,
  random,
  totalSteps = 100_000,
  groupSize = 8,
  epochs = 4,
  batchSize = 64,
  width = 64,
  clipRatio = 0.2,
  entropyCoefficient = 0.01,
  annealLearningRate = true,
  actorLearningRate = 3e-4,
  progressEvery = 0,
  onCheckpoint = () => {},
  onProgress = () => {},
} = {}) {
  const environments = envs ?? (env ? [env] : []);
  if (environments.length < 2) throw new Error("GRPO needs at least two matched environments");
  if (environments.length !== groupSize) {
    throw new Error(`GRPO expected ${groupSize} environments, got ${environments.length}`);
  }
  if (groupSize > batchSize) throw new Error("GRPO group size cannot exceed batch size");

  const maximumHorizon = Math.max(...environments.map((environment) => environment.horizon));
  const capacity = groupSize * maximumHorizon;
  const actor = makePPOActor(RL_OBSERVATION_SIZE, width, batchSize, random);
  const optimizer = new Adam(actor.params, { learningRate: actorLearningRate, epsilon: 1e-5 });
  const observations = new Float32Array(capacity * RL_OBSERVATION_SIZE);
  const actions = new Float32Array(capacity * RL_ACTION_SIZE);
  const oldLogProbabilities = new Float32Array(capacity);
  const trajectoryIndices = new Uint16Array(capacity);
  const advantages = new Float32Array(capacity);
  const order = new Int32Array(capacity);
  const actorGradient = new Float32Array(batchSize * RL_ACTION_SIZE);
  const groupReturns = new Float32Array(groupSize);
  const groupAdvantages = new Float32Array(groupSize);
  const currentObservations = new Float32Array(groupSize * RL_OBSERVATION_SIZE);
  const active = new Uint8Array(groupSize);
  const actionsByEnvironment = Array.from({ length: groupSize }, () => new Float32Array(2));

  let steps = 0;
  let episodes = 0;
  let successes = 0;
  let nextCheckpoint = progressEvery > 0 ? progressEvery : Infinity;

  while (steps < totalSteps) {
    const groupCount = Math.min(groupSize, totalSteps - steps);
    const groupPaths = Array.from({ length: groupCount }, () => []);
    const learningRateFraction = annealLearningRate ? 1 - steps / totalSteps : 1;
    optimizer.learningRate = actorLearningRate * learningRateFraction;
    groupReturns.fill(0);
    active.fill(0);
    active.fill(1, 0, groupCount);
    for (let trajectory = 0; trajectory < groupCount; trajectory++) {
      environments[trajectory].setTrainingProgress?.(steps / totalSteps);
      currentObservations.set(environments[trajectory].reset(), trajectory * RL_OBSERVATION_SIZE);
      groupPaths[trajectory].push(
        environments[trajectory].world.pusher.x,
        environments[trajectory].world.pusher.y,
      );
    }

    let count = 0;
    let activeCount = groupCount;
    let groupSuccesses = 0;
    while (activeCount > 0 && count < capacity && steps + count < totalSteps) {
      const activeTrajectories = [];
      for (let trajectory = 0; trajectory < groupCount; trajectory++) {
        if (!active[trajectory]) continue;
        const sourceOffset = trajectory * RL_OBSERVATION_SIZE;
        const row = activeTrajectories.length;
        actor.inputBuffer().set(
          currentObservations.subarray(sourceOffset, sourceOffset + RL_OBSERVATION_SIZE),
          row * RL_OBSERVATION_SIZE,
        );
        activeTrajectories.push(trajectory);
      }
      const output = actor.forward(activeTrajectories.length);
      for (let row = 0; row < activeTrajectories.length && steps + count < totalSteps; row++) {
        const trajectory = activeTrajectories[row];
        const observationOffset = trajectory * RL_OBSERVATION_SIZE;
        observations.set(
          currentObservations.subarray(observationOffset, observationOffset + RL_OBSERVATION_SIZE),
          count * RL_OBSERVATION_SIZE,
        );
        const sample = ppoActorSample(output, row * 4, random);
        actions[count * 2] = sample.actionX;
        actions[count * 2 + 1] = sample.actionY;
        oldLogProbabilities[count] = sample.logProbability;
        trajectoryIndices[count] = trajectory;

        const action = actionsByEnvironment[trajectory];
        action[0] = sample.actionX;
        action[1] = sample.actionY;
        const transition = environments[trajectory].step(action);
        groupReturns[trajectory] += transition.reward;
        groupPaths[trajectory].push(
          environments[trajectory].world.pusher.x,
          environments[trajectory].world.pusher.y,
        );
        currentObservations.set(transition.observation, observationOffset);
        count += 1;
        if (transition.done) {
          active[trajectory] = 0;
          activeCount -= 1;
          episodes += 1;
          if (transition.success) {
            successes += 1;
            groupSuccesses += 1;
          }
        }
      }
    }

    const normalized = normalizeGroupAdvantages(
      groupReturns.subarray(0, groupCount),
      groupAdvantages.subarray(0, groupCount),
    );
    for (let index = 0; index < count; index++) {
      advantages[index] = groupAdvantages[trajectoryIndices[index]];
      order[index] = index;
    }

    let policyLoss = 0;
    let updates = 0;
    for (let epoch = 0; epoch < epochs; epoch++) {
      shuffle(order.subarray(0, count), random);
      for (let start = 0; start < count; start += batchSize) {
        const batch = Math.min(batchSize, count - start);
        const indices = order.subarray(start, start + batch);
        fillRows(actor.inputBuffer(), batch, RL_OBSERVATION_SIZE, indices, observations);
        const output = actor.forward(batch);
        actor.zeroGrad();
        actorGradient.fill(0, 0, batch * RL_ACTION_SIZE);
        let batchPolicyLoss = 0;
        for (let row = 0; row < batch; row++) {
          const index = indices[row];
          const offset = row * 4;
          const meanX = output[offset];
          const meanY = output[offset + 1];
          const logStdX = output[offset + 2];
          const logStdY = output[offset + 3];
          const differenceX = actions[index * 2] - meanX;
          const differenceY = actions[index * 2 + 1] - meanY;
          const inverseVarianceX = Math.exp(-2 * logStdX);
          const inverseVarianceY = Math.exp(-2 * logStdY);
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
        clipGradients(actor, 0.5);
        optimizer.step(actor.grads);
        policyLoss += batchPolicyLoss;
        updates += 1;
      }
    }

    steps += count;
    const progress = {
      algorithm: "grpo",
      steps,
      episodes,
      successes,
      groupSuccesses,
      groupReturnMean: normalized.mean,
      groupReturnStd: normalized.standardDeviation,
      policyLoss: policyLoss / Math.max(1, updates),
    };
    while (steps >= nextCheckpoint) {
      onCheckpoint({ ...progress, steps: nextCheckpoint }, {
        actor,
        groupPaths,
        groupAdvantages: Array.from(groupAdvantages.subarray(0, groupCount)),
      });
      nextCheckpoint += progressEvery;
    }
    onProgress(progress, { actor });
  }

  return { actor, steps, episodes, successes };
}
