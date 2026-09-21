import { MLP } from "../src/flow/mlp.js";
import {
  createRandom,
  FIXED_BLOCK_START,
  FIXED_PUSHER_START,
  MAX_PUSHER_SPEED,
} from "../src/pusht/sim.js";
import { ScriptedExpert } from "../src/pusht/expert.js";
import {
  CURRICULUM_INITIAL_DISTANCE_FRACTION,
  PushTRLEnv,
  RL_ACTION_SIZE,
  RL_OBSERVATION_SIZE,
} from "../src/pusht/rl/env.js";
import { recordPolicyRollout } from "../src/pusht/rl/common.js";
import { trainPPO } from "../src/pusht/rl/ppo.js";
import { trainSAC } from "../src/pusht/rl/sac.js";

const finiteModel = (model) => model.params.every((buffer) => buffer.every(Number.isFinite));
const scaledRewardEnv = new PushTRLEnv({ distanceRewardScale: 3 });
if (Math.abs(scaledRewardEnv.pusherDistanceRewardScale - 0.3) > 1e-12) {
  throw new Error("pusher shaping is not 0.1x the block-distance weight");
}
const noShapingEnv = new PushTRLEnv({
  seed: 8,
  curriculum: false,
  rewardShaping: {
    blockDistance: false,
    pusherDistance: false,
    orientation: false,
    closeness: false,
  },
});
noShapingEnv.reset();
const towardGoalX = noShapingEnv.world.goal.x - noShapingEnv.world.pusher.x;
const towardGoalY = noShapingEnv.world.goal.y - noShapingEnv.world.pusher.y;
const towardGoalLength = Math.hypot(towardGoalX, towardGoalY);
const noShapingTransition = noShapingEnv.step(Float32Array.of(
  towardGoalX / towardGoalLength,
  towardGoalY / towardGoalLength,
));
if (noShapingTransition.pusherDistanceProgress <= 0 || noShapingTransition.reward !== 0) {
  throw new Error("disabled reward shaping still changed the reward");
}

// The block starts near the goal, moves to the final task radius by 70%, and
// gets a randomized valid bearing while holding the scheduled radius exact.
const curriculumEnv = new PushTRLEnv({ seed: 9 });
const finalDistance = Math.hypot(
  FIXED_BLOCK_START.x - curriculumEnv.world.goal.x,
  FIXED_BLOCK_START.y - curriculumEnv.world.goal.y,
);
const initialPusherDistance = Math.hypot(
  FIXED_PUSHER_START.x - FIXED_BLOCK_START.x,
  FIXED_PUSHER_START.y - FIXED_BLOCK_START.y,
);
const curriculumPositions = [];
const checkCurriculumDistance = (progress, expectedFraction) => {
  curriculumEnv.setTrainingProgress(progress);
  curriculumEnv.reset();
  curriculumPositions.push([curriculumEnv.world.block.x, curriculumEnv.world.block.y]);
  if (Math.abs(curriculumEnv.blockGoalDistance() - finalDistance * expectedFraction) > 1e-9) {
    throw new Error(`wrong curriculum distance at progress ${progress}`);
  }
  const pusherDistance = Math.hypot(
    curriculumEnv.world.pusher.x - curriculumEnv.world.block.x,
    curriculumEnv.world.pusher.y - curriculumEnv.world.block.y,
  );
  if (Math.abs(pusherDistance - initialPusherDistance) > 1e-9) {
    throw new Error(`wrong pusher distance at progress ${progress}`);
  }
};
checkCurriculumDistance(0, CURRICULUM_INITIAL_DISTANCE_FRACTION);
checkCurriculumDistance(0.35, (1 + CURRICULUM_INITIAL_DISTANCE_FRACTION) / 2);
checkCurriculumDistance(0.7, 1);
checkCurriculumDistance(1, 1);
const distinctBearings = new Set(curriculumPositions.map(([x, y]) =>
  Math.atan2(y - curriculumEnv.world.goal.y, x - curriculumEnv.world.goal.x).toFixed(6)));
if (distinctBearings.size < 2) throw new Error("curriculum did not randomize block bearing");
const noCurriculumEnv = new PushTRLEnv({ seed: 9, curriculum: false });
noCurriculumEnv.setTrainingProgress(0);
noCurriculumEnv.reset();
if (noCurriculumEnv.world.block.x !== FIXED_BLOCK_START.x ||
    noCurriculumEnv.world.block.y !== FIXED_BLOCK_START.y) {
  throw new Error("disabled curriculum did not use the final block start");
}
console.log("curriculum: randomized bearing and matched pusher gap, near radius at 0%, final radius at 70%");

// Environment contract: position and orientation progress telescope to their
// net reductions, with exactly one additional reward on completion.
const env = new PushTRLEnv({ seed: 10, horizon: 2200 });
env.setTrainingProgress(1);
let observation = env.reset();
const initialDistance = env.distance;
const initialPusherDistanceToGoal = env.pusherDistance;
const initialOrientationError = env.orientationError;
if (observation.length !== RL_OBSERVATION_SIZE || RL_ACTION_SIZE !== 2) throw new Error("wrong RL shape");
const expert = new ScriptedExpert({ random: createRandom(11), tieBreak: "cw" });
const action = new Float32Array(2);
let completionRewards = 0;
let distanceReward = 0;
let pusherDistanceReward = 0;
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
  pusherDistanceReward += transition.pusherDistanceProgress;
  orientationReward += transition.orientationProgress;
  finalClosenessReward += transition.finalClosenessReward;
  totalReward += transition.reward;
  if (transition.done) { solved = transition.success; break; }
}
const expectedDistanceReward = initialDistance - env.distance;
const expectedPusherDistanceReward = initialPusherDistanceToGoal - env.pusherDistance;
const expectedOrientationReward = initialOrientationError - env.orientationError;
console.log(
  `environment: obs ${observation.length}, action ${RL_ACTION_SIZE}, solved ${solved}, ` +
  `completion ${completionRewards}, distance reward ${distanceReward.toFixed(4)}, ` +
  `pusher reward ${(0.1 * pusherDistanceReward).toFixed(4)}, ` +
  `orientation reward ${orientationReward.toFixed(4)}, closeness ${finalClosenessReward.toFixed(4)}, ` +
  `total ${totalReward.toFixed(4)}`,
);
if (!solved || completionRewards !== 1) throw new Error("completion reward is wrong");
if (Math.abs(distanceReward - expectedDistanceReward) > 1e-6) throw new Error("distance shaping does not telescope");
if (Math.abs(pusherDistanceReward - expectedPusherDistanceReward) > 1e-6) {
  throw new Error("pusher distance shaping does not telescope");
}
if (Math.abs(orientationReward - expectedOrientationReward) > 1e-6) throw new Error("orientation shaping does not telescope");
if (Math.abs(totalReward - (1 + finalClosenessReward + expectedDistanceReward +
    0.1 * expectedPusherDistanceReward + expectedOrientationReward)) > 1e-6) {
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

// Touching any outer wall is an immediate terminal failure whose -1 penalty is
// summed with the ordinary shaping terms from that final transition.
const wallEnv = new PushTRLEnv({ seed: 15, horizon: 100, curriculum: false });
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
const expectedWallReward = wallTransition.wallPenalty + wallTransition.distanceProgress +
  0.1 * wallTransition.pusherDistanceProgress + wallTransition.orientationProgress;
if (wallTransition.wallPenalty !== -1 || Math.abs(wallTransition.reward - expectedWallReward) > 1e-9) {
  throw new Error("wall contact penalty was not summed with shaping");
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
let ppoCheckpoints = 0;
const ppo = trainPPO({
  env: new PushTRLEnv({ seed: 20, horizon: 80 }),
  random: createRandom(21),
  totalSteps: 256,
  rolloutSteps: 128,
  epochs: 1,
  batchSize: 32,
  width: 16,
  progressEvery: 200,
  onCheckpoint(progress) {
    if (progress.steps !== 200) throw new Error(`unexpected PPO checkpoint ${progress.steps}`);
    ppoCheckpoints += 1;
  },
});
console.log(`PPO smoke: ${ppo.steps} steps, finite ${finiteModel(ppo.actor) && finiteModel(ppo.critic)}`);
if (!finiteModel(ppo.actor) || !finiteModel(ppo.critic)) throw new Error("PPO produced non-finite parameters");
if (ppoCheckpoints !== 1) throw new Error(`expected one PPO checkpoint, got ${ppoCheckpoints}`);
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
