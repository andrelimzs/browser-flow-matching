import { MLP } from "../src/flow/mlp.js";
import { createRandom, MAX_PUSHER_SPEED } from "../src/pusht/sim.js";
import { ScriptedExpert } from "../src/pusht/expert.js";
import { PushTRLEnv, RL_ACTION_SIZE, RL_OBSERVATION_SIZE } from "../src/pusht/rl/env.js";
import { recordPolicyRollout } from "../src/pusht/rl/common.js";
import { trainPPO } from "../src/pusht/rl/ppo.js";
import { trainSAC } from "../src/pusht/rl/sac.js";

const finiteModel = (model) => model.params.every((buffer) => buffer.every(Number.isFinite));

// Environment contract: position and orientation progress telescope to their
// net reductions, with exactly one additional reward on completion.
const env = new PushTRLEnv({ seed: 10, horizon: 2200 });
let observation = env.reset();
const initialDistance = env.distance;
const initialOrientationError = env.orientationError;
if (observation.length !== RL_OBSERVATION_SIZE || RL_ACTION_SIZE !== 2) throw new Error("wrong RL shape");
const expert = new ScriptedExpert({ random: createRandom(11), tieBreak: "cw" });
const action = new Float32Array(2);
let completionRewards = 0;
let distanceReward = 0;
let orientationReward = 0;
let finalClosenessReward = 0;
let totalReward = 0;
let solved = false;
for (let step = 0; step < 2200; step++) {
  const [targetX, targetY] = expert.act(env.world);
  action[0] = (targetX - env.world.pusher.x) / MAX_PUSHER_SPEED;
  action[1] = (targetY - env.world.pusher.y) / MAX_PUSHER_SPEED;
  const transition = env.step(action);
  observation = transition.observation;
  completionRewards += transition.completionReward;
  distanceReward += transition.distanceProgress;
  orientationReward += transition.orientationProgress;
  finalClosenessReward += transition.finalClosenessReward;
  totalReward += transition.reward;
  if (transition.done) { solved = transition.success; break; }
}
const expectedDistanceReward = initialDistance - env.distance;
const expectedOrientationReward = initialOrientationError - env.orientationError;
console.log(
  `environment: obs ${observation.length}, action ${RL_ACTION_SIZE}, solved ${solved}, ` +
  `completion ${completionRewards}, distance reward ${distanceReward.toFixed(4)}, ` +
  `orientation reward ${orientationReward.toFixed(4)}, closeness ${finalClosenessReward.toFixed(4)}, ` +
  `total ${totalReward.toFixed(4)}`,
);
if (!solved || completionRewards !== 1) throw new Error("completion reward is wrong");
if (Math.abs(distanceReward - expectedDistanceReward) > 1e-6) throw new Error("distance shaping does not telescope");
if (Math.abs(orientationReward - expectedOrientationReward) > 1e-6) throw new Error("orientation shaping does not telescope");
if (Math.abs(totalReward - (1 + finalClosenessReward + expectedDistanceReward + expectedOrientationReward)) > 1e-6) {
  throw new Error("combined reward is wrong");
}

const orientationEnv = new PushTRLEnv({ seed: 14, horizon: 100 });
orientationEnv.reset();
orientationEnv.world.block.angle = Math.PI;
orientationEnv.orientationError = orientationEnv.blockGoalOrientationError();
orientationEnv.world.block.angle = Math.PI / 2;
const orientationTransition = orientationEnv.step(Float32Array.of(0, 0));
console.log(`orientation progress: ${orientationTransition.orientationProgress.toFixed(4)}`);
if (Math.abs(orientationTransition.orientationProgress - 0.5) > 1e-6) {
  throw new Error("orientation shaping is not normalized angular progress");
}

const closenessEnv = new PushTRLEnv({ seed: 16, horizon: 1 });
closenessEnv.reset();
closenessEnv.world.block.x = closenessEnv.world.goal.x + 0.015;
closenessEnv.world.block.y = closenessEnv.world.goal.y;
closenessEnv.world.block.angle = closenessEnv.world.goal.angle;
closenessEnv.distance = closenessEnv.blockGoalDistance();
closenessEnv.orientationError = closenessEnv.blockGoalOrientationError();
const closenessTransition = closenessEnv.step(Float32Array.of(0, 0));
console.log(
  `terminal closeness: coverage ${closenessTransition.coverage.toFixed(4)}, reward ${closenessTransition.reward.toFixed(4)}`,
);
if (!closenessTransition.truncated || closenessTransition.success || closenessTransition.wallContact) {
  throw new Error("closeness test did not terminate by horizon");
}
if (Math.abs(closenessTransition.finalClosenessReward - closenessTransition.coverage) > 1e-6) {
  throw new Error("terminal closeness reward does not equal final coverage");
}

// Touching any outer wall is an immediate terminal failure with an exact -1
// reward, regardless of the ordinary distance-shaping term.
const wallEnv = new PushTRLEnv({ seed: 15, horizon: 100 });
wallEnv.reset();
let wallTransition = null;
for (let step = 0; step < 20 && !wallTransition?.done; step++) {
  wallTransition = wallEnv.step(Float32Array.of(-1, 0));
}
console.log(
  `wall contact: done ${wallTransition?.done}, contact ${wallTransition?.wallContact}, reward ${wallTransition?.reward}`,
);
if (!wallTransition?.done || !wallTransition.wallContact || wallTransition.success || wallTransition.truncated) {
  throw new Error("wall contact did not terminate as a failure");
}
if (wallTransition.reward !== -1 || wallTransition.wallPenalty !== -1) {
  throw new Error("wall contact reward is not exactly -1");
}

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
// actor-through-critic gradient wiring. They are not expected to solve the task
// in a few hundred interactions.
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
const recordedRollout = recordPolicyRollout(ppo.actor, new PushTRLEnv({ seed: 22, horizon: 80 }));
console.log(`recorded rollout: ${recordedRollout.count} frames, ${recordedRollout.frames.length} values`);
if (recordedRollout.frames.length !== recordedRollout.count * 10 || !recordedRollout.frames.every(Number.isFinite)) {
  throw new Error("recorded rollout action distribution has the wrong shape or non-finite values");
}
const sampledRollout = recordPolicyRollout(ppo.actor, new PushTRLEnv({ seed: 23, horizon: 80 }), {
  random: createRandom(24),
  deterministic: false,
});
let sampledTravel = 0;
for (let frame = 1; frame < sampledRollout.count; frame++) {
  const before = (frame - 1) * 10;
  const after = frame * 10;
  sampledTravel += Math.hypot(
    sampledRollout.frames[after] - sampledRollout.frames[before],
    sampledRollout.frames[after + 1] - sampledRollout.frames[before + 1],
  );
}
console.log(`sampled rollout: ${sampledRollout.count} frames, pusher travel ${sampledTravel.toFixed(4)}`);
if (sampledTravel <= 0) throw new Error("sampled rollout did not execute policy noise");

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
