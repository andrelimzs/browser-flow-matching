// Soft Actor-Critic with twin Q-functions and a uniform replay buffer.

import { Adam } from "../../flow/adam.js";
import {
  LOG_STD_MIN,
  LOG_STD_MAX,
  actorSample,
  clipGradients,
  copyParameters,
  makeActor,
  makeCritic,
  softUpdate,
} from "./common.js";
import { RL_ACTION_SIZE, RL_OBSERVATION_SIZE } from "./env.js";

class ReplayBuffer {
  constructor(capacity) {
    this.capacity = capacity;
    this.observations = new Float32Array(capacity * RL_OBSERVATION_SIZE);
    this.nextObservations = new Float32Array(capacity * RL_OBSERVATION_SIZE);
    this.actions = new Float32Array(capacity * RL_ACTION_SIZE);
    this.rewards = new Float32Array(capacity);
    this.dones = new Uint8Array(capacity);
    this.size = 0;
    this.cursor = 0;
  }

  add(observation, action, reward, nextObservation, done) {
    const index = this.cursor;
    this.observations.set(observation, index * RL_OBSERVATION_SIZE);
    this.nextObservations.set(nextObservation, index * RL_OBSERVATION_SIZE);
    this.actions.set(action, index * RL_ACTION_SIZE);
    this.rewards[index] = reward;
    this.dones[index] = done ? 1 : 0;
    this.cursor = (index + 1) % this.capacity;
    this.size = Math.min(this.capacity, this.size + 1);
  }

  sample(batch, random, indices) {
    for (let row = 0; row < batch; row++) indices[row] = Math.floor(random() * this.size);
  }
}

function fillCriticInput(buffer, observations, actions, indices, batch) {
  const width = RL_OBSERVATION_SIZE + RL_ACTION_SIZE;
  for (let row = 0; row < batch; row++) {
    const index = indices[row];
    const offset = row * width;
    buffer.set(
      observations.subarray(index * RL_OBSERVATION_SIZE, (index + 1) * RL_OBSERVATION_SIZE),
      offset,
    );
    buffer[offset + RL_OBSERVATION_SIZE] = actions[row * 2];
    buffer[offset + RL_OBSERVATION_SIZE + 1] = actions[row * 2 + 1];
  }
}

export function trainSAC({
  env,
  random,
  totalSteps = 200_000,
  width = 128,
  batchSize = 128,
  replayCapacity = 100_000,
  warmupSteps = 2_000,
  gamma = 0.99,
  tau = 0.005,
  alpha = 0.2,
  actorLearningRate = 3e-4,
  criticLearningRate = 3e-4,
  progressEvery = 5000,
  onProgress = () => {},
} = {}) {
  const actor = makeActor(RL_OBSERVATION_SIZE, width, batchSize, random);
  const critic1 = makeCritic(RL_OBSERVATION_SIZE, width, batchSize, random);
  const critic2 = makeCritic(RL_OBSERVATION_SIZE, width, batchSize, random);
  const target1 = makeCritic(RL_OBSERVATION_SIZE, width, batchSize, random);
  const target2 = makeCritic(RL_OBSERVATION_SIZE, width, batchSize, random);
  copyParameters(target1, critic1);
  copyParameters(target2, critic2);
  const actorOptimizer = new Adam(actor.params, { learningRate: actorLearningRate });
  const critic1Optimizer = new Adam(critic1.params, { learningRate: criticLearningRate });
  const critic2Optimizer = new Adam(critic2.params, { learningRate: criticLearningRate });
  const replay = new ReplayBuffer(replayCapacity);

  const indices = new Int32Array(batchSize);
  const batchActions = new Float32Array(batchSize * 2);
  const nextActions = new Float32Array(batchSize * 2);
  const targets = new Float32Array(batchSize);
  const critic1Gradient = new Float32Array(batchSize);
  const critic2Gradient = new Float32Array(batchSize);
  const actorGradient = new Float32Array(batchSize * 4);
  const selector1 = new Float32Array(batchSize);
  const selector2 = new Float32Array(batchSize);
  const inputGradient1 = new Float32Array(batchSize * (RL_OBSERVATION_SIZE + RL_ACTION_SIZE));
  const inputGradient2 = new Float32Array(batchSize * (RL_OBSERVATION_SIZE + RL_ACTION_SIZE));
  const epsilons = new Float32Array(batchSize * 2);
  const logStds = new Float32Array(batchSize * 2);
  const rawLogStds = new Float32Array(batchSize * 2);
  const action = new Float32Array(2);

  let observation = Float32Array.from(env.reset());
  let episodes = 0;
  let successes = 0;
  let intervalSuccesses = 0;
  let criticLoss = 0;
  let actorLoss = 0;
  let updates = 0;

  for (let step = 1; step <= totalSteps; step++) {
    if (step <= warmupSteps) {
      action[0] = random() * 2 - 1;
      action[1] = random() * 2 - 1;
    } else {
      actor.inputBuffer().set(observation, 0);
      const sample = actorSample(actor.forward(1), 0, random);
      action[0] = sample.actionX;
      action[1] = sample.actionY;
    }
    const transition = env.step(action);
    const nextObservation = Float32Array.from(transition.observation);
    replay.add(observation, action, transition.reward, nextObservation, transition.done);
    observation = nextObservation;
    if (transition.done) {
      episodes += 1;
      if (transition.success) { successes += 1; intervalSuccesses += 1; }
      observation = Float32Array.from(env.reset());
    }

    if (replay.size >= batchSize && step > warmupSteps) {
      replay.sample(batchSize, random, indices);

      // Bellman targets use fresh actions from the current actor and slowly
      // moving target critics.
      const actorInput = actor.inputBuffer();
      for (let row = 0; row < batchSize; row++) {
        const index = indices[row];
        actorInput.set(
          replay.nextObservations.subarray(index * RL_OBSERVATION_SIZE, (index + 1) * RL_OBSERVATION_SIZE),
          row * RL_OBSERVATION_SIZE,
        );
      }
      const nextActorOutput = actor.forward(batchSize);
      const nextLogProbabilities = new Float32Array(batchSize);
      for (let row = 0; row < batchSize; row++) {
        const sample = actorSample(nextActorOutput, row * 4, random);
        nextActions[row * 2] = sample.actionX;
        nextActions[row * 2 + 1] = sample.actionY;
        nextLogProbabilities[row] = sample.logProbability;
      }
      fillCriticInput(target1.inputBuffer(), replay.nextObservations, nextActions, indices, batchSize);
      fillCriticInput(target2.inputBuffer(), replay.nextObservations, nextActions, indices, batchSize);
      const nextQ1 = target1.forward(batchSize);
      const nextQ2 = target2.forward(batchSize);
      for (let row = 0; row < batchSize; row++) {
        const index = indices[row];
        const continuation = replay.dones[index] ? 0 : 1;
        targets[row] = replay.rewards[index] +
          gamma * continuation * (Math.min(nextQ1[row], nextQ2[row]) - alpha * nextLogProbabilities[row]);
      }

      for (let row = 0; row < batchSize; row++) {
        const index = indices[row];
        batchActions[row * 2] = replay.actions[index * 2];
        batchActions[row * 2 + 1] = replay.actions[index * 2 + 1];
      }
      fillCriticInput(critic1.inputBuffer(), replay.observations, batchActions, indices, batchSize);
      fillCriticInput(critic2.inputBuffer(), replay.observations, batchActions, indices, batchSize);
      const currentQ1 = critic1.forward(batchSize);
      const currentQ2 = critic2.forward(batchSize);
      let batchCriticLoss = 0;
      for (let row = 0; row < batchSize; row++) {
        const error1 = currentQ1[row] - targets[row];
        const error2 = currentQ2[row] - targets[row];
        critic1Gradient[row] = error1 / batchSize;
        critic2Gradient[row] = error2 / batchSize;
        batchCriticLoss += 0.5 * (error1 * error1 + error2 * error2) / batchSize;
      }
      critic1.zeroGrad();
      critic1.backward(critic1Gradient, batchSize);
      clipGradients(critic1, 1);
      critic1Optimizer.step(critic1.grads);
      critic2.zeroGrad();
      critic2.backward(critic2Gradient, batchSize);
      clipGradients(critic2, 1);
      critic2Optimizer.step(critic2.grads);

      // Reparameterized actor update: alpha*log pi(a|s) - min(Q1,Q2).
      for (let row = 0; row < batchSize; row++) {
        const index = indices[row];
        actorInput.set(
          replay.observations.subarray(index * RL_OBSERVATION_SIZE, (index + 1) * RL_OBSERVATION_SIZE),
          row * RL_OBSERVATION_SIZE,
        );
      }
      const actorOutput = actor.forward(batchSize);
      let batchActorLoss = 0;
      for (let row = 0; row < batchSize; row++) {
        const sample = actorSample(actorOutput, row * 4, random);
        batchActions[row * 2] = sample.actionX;
        batchActions[row * 2 + 1] = sample.actionY;
        epsilons[row * 2] = sample.epsilonX;
        epsilons[row * 2 + 1] = sample.epsilonY;
        logStds[row * 2] = sample.logStdX;
        logStds[row * 2 + 1] = sample.logStdY;
        rawLogStds[row * 2] = sample.rawLogStdX;
        rawLogStds[row * 2 + 1] = sample.rawLogStdY;
      }
      fillCriticInput(critic1.inputBuffer(), replay.observations, batchActions, indices, batchSize);
      fillCriticInput(critic2.inputBuffer(), replay.observations, batchActions, indices, batchSize);
      const actorQ1 = critic1.forward(batchSize);
      const actorQ2 = critic2.forward(batchSize);
      selector1.fill(0);
      selector2.fill(0);
      for (let row = 0; row < batchSize; row++) {
        const useFirst = actorQ1[row] <= actorQ2[row];
        selector1[row] = useFirst ? 1 : 0;
        selector2[row] = useFirst ? 0 : 1;
        const actionX = batchActions[row * 2];
        const actionY = batchActions[row * 2 + 1];
        const epsilonX = epsilons[row * 2];
        const epsilonY = epsilons[row * 2 + 1];
        const logProbability =
          -0.5 * (epsilonX * epsilonX + epsilonY * epsilonY + 2 * Math.log(2 * Math.PI))
          - logStds[row * 2] - logStds[row * 2 + 1]
          - Math.log(1 - actionX * actionX + 1e-6)
          - Math.log(1 - actionY * actionY + 1e-6);
        batchActorLoss += (alpha * logProbability - Math.min(actorQ1[row], actorQ2[row])) / batchSize;
      }
      critic1.zeroGrad();
      critic1.backward(selector1, batchSize);
      critic1.inputGradients(batchSize, inputGradient1);
      critic2.zeroGrad();
      critic2.backward(selector2, batchSize);
      critic2.inputGradients(batchSize, inputGradient2);

      actorGradient.fill(0);
      for (let row = 0; row < batchSize; row++) {
        const actorOffset = row * 4;
        const criticOffset = row * (RL_OBSERVATION_SIZE + RL_ACTION_SIZE) + RL_OBSERVATION_SIZE;
        const actionX = batchActions[row * 2];
        const actionY = batchActions[row * 2 + 1];
        const qGradientX = inputGradient1[criticOffset] + inputGradient2[criticOffset];
        const qGradientY = inputGradient1[criticOffset + 1] + inputGradient2[criticOffset + 1];
        const dzX = alpha * 2 * actionX - qGradientX * (1 - actionX * actionX);
        const dzY = alpha * 2 * actionY - qGradientY * (1 - actionY * actionY);
        actorGradient[actorOffset] = dzX / batchSize;
        actorGradient[actorOffset + 1] = dzY / batchSize;
        if (rawLogStds[row * 2] > LOG_STD_MIN && rawLogStds[row * 2] < LOG_STD_MAX) {
          actorGradient[actorOffset + 2] =
            (-alpha + dzX * Math.exp(logStds[row * 2]) * epsilons[row * 2]) / batchSize;
        }
        if (rawLogStds[row * 2 + 1] > LOG_STD_MIN && rawLogStds[row * 2 + 1] < LOG_STD_MAX) {
          actorGradient[actorOffset + 3] =
            (-alpha + dzY * Math.exp(logStds[row * 2 + 1]) * epsilons[row * 2 + 1]) / batchSize;
        }
      }
      actor.zeroGrad();
      actor.backward(actorGradient, batchSize);
      clipGradients(actor, 1);
      actorOptimizer.step(actor.grads);
      softUpdate(target1, critic1, tau);
      softUpdate(target2, critic2, tau);
      criticLoss += batchCriticLoss;
      actorLoss += batchActorLoss;
      updates += 1;
    }

    if (step % progressEvery === 0 || step === totalSteps) {
      onProgress({
        algorithm: "sac",
        steps: step,
        episodes,
        successes,
        intervalSuccesses,
        actorLoss: actorLoss / Math.max(1, updates),
        criticLoss: criticLoss / Math.max(1, updates),
        replaySize: replay.size,
      });
      intervalSuccesses = 0;
      criticLoss = 0;
      actorLoss = 0;
      updates = 0;
    }
  }

  return { actor, critic1, critic2, steps: totalSteps, episodes, successes, replay };
}
