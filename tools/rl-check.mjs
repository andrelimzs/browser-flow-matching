import { MLP } from "../src/flow/mlp.js";
import { createRandom, MAX_PUSHER_SPEED } from "../src/pusht/sim.js";
import { ScriptedExpert } from "../src/pusht/expert.js";
import { PushTRLEnv, RL_ACTION_SIZE, RL_OBSERVATION_SIZE } from "../src/pusht/rl/env.js";
import { trainPPO } from "../src/pusht/rl/ppo.js";
import { trainSAC } from "../src/pusht/rl/sac.js";

const finiteModel = (model) => model.params.every((buffer) => buffer.every(Number.isFinite));

// Environment contract and sparse reward: every nonterminal reward is zero and
// the expert receives exactly one reward on the completing transition.
const env = new PushTRLEnv({ seed: 10, horizon: 2200 });
let observation = env.reset();
if (observation.length !== RL_OBSERVATION_SIZE || RL_ACTION_SIZE !== 2) throw new Error("wrong RL shape");
const expert = new ScriptedExpert({ random: createRandom(11), tieBreak: "cw" });
const action = new Float32Array(2);
let completionRewards = 0;
let shapedRewards = 0;
let solved = false;
for (let step = 0; step < 2200; step++) {
  const [targetX, targetY] = expert.act(env.world);
  action[0] = (targetX - env.world.pusher.x) / MAX_PUSHER_SPEED;
  action[1] = (targetY - env.world.pusher.y) / MAX_PUSHER_SPEED;
  const transition = env.step(action);
  observation = transition.observation;
  if (transition.reward === 1) completionRewards += 1;
  else if (transition.reward !== 0) shapedRewards += 1;
  if (transition.done) { solved = transition.success; break; }
}
console.log(`environment: obs ${observation.length}, action ${RL_ACTION_SIZE}, solved ${solved}, completion rewards ${completionRewards}, other rewards ${shapedRewards}`);
if (!solved || completionRewards !== 1 || shapedRewards !== 0) throw new Error("reward is not completion-only");

// Validate the critic input-gradient path against finite differences.
const gradientRandom = createRandom(12);
const gradientModel = new MLP({ sizes: [3, 5, 1], maxBatch: 1, random: gradientRandom });
const input = gradientModel.inputBuffer();
input.set([0.2, -0.3, 0.4]);
gradientModel.forward(1);
gradientModel.zeroGrad();
gradientModel.backward(Float32Array.of(1), 1);
const analytic = gradientModel.inputGradients(1);
const epsilon = 1e-3;
let worst = 0;
for (let index = 0; index < 3; index++) {
  const original = input[index];
  input[index] = original + epsilon;
  const plus = gradientModel.forward(1)[0];
  input[index] = original - epsilon;
  const minus = gradientModel.forward(1)[0];
  input[index] = original;
  worst = Math.max(worst, Math.abs(analytic[index] - (plus - minus) / (2 * epsilon)));
}
console.log(`critic input-gradient worst error ${worst.toExponential(3)}`);
if (worst > 2e-4) throw new Error(`critic input-gradient mismatch ${worst}`);

// Short algorithm smoke runs catch non-finite losses, buffer mistakes and SAC's
// actor-through-critic gradient wiring. They are not expected to solve a sparse
// completion task in a few hundred interactions.
const ppo = trainPPO({
  env: new PushTRLEnv({ seed: 20, horizon: 80 }),
  random: createRandom(21),
  totalSteps: 256,
  rolloutSteps: 128,
  epochs: 1,
  batchSize: 32,
  width: 16,
});
console.log(`PPO smoke: ${ppo.steps} steps, finite ${finiteModel(ppo.actor) && finiteModel(ppo.critic)}`);
if (!finiteModel(ppo.actor) || !finiteModel(ppo.critic)) throw new Error("PPO produced non-finite parameters");

const sac = trainSAC({
  env: new PushTRLEnv({ seed: 30, horizon: 80 }),
  random: createRandom(31),
  totalSteps: 256,
  warmupSteps: 64,
  batchSize: 32,
  replayCapacity: 512,
  progressEvery: 256,
  width: 16,
});
const sacFinite = finiteModel(sac.actor) && finiteModel(sac.critic1) && finiteModel(sac.critic2);
console.log(`SAC smoke: ${sac.steps} steps, replay ${sac.replay.size}, finite ${sacFinite}`);
if (!sacFinite) throw new Error("SAC produced non-finite parameters");
