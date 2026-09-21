import { MLP } from "../src/flow/mlp.js";
import {
  createRandom,
  FIXED_BLOCK_START,
  FIXED_PUSHER_START,
  MAX_PUSHER_SPEED,
  WALL_THICKNESS,
} from "../src/pusht/sim.js";
import { ScriptedExpert } from "../src/pusht/expert.js";
import {
  CURRICULUM_INITIAL_DISTANCE_FRACTION,
  cartesianActionToDelta,
  MAX_STATIONARY_STEPS,
  PushTRLEnv,
  RL_ACTION_SIZE,
  RL_OBSERVATION_SIZE,
} from "../src/pusht/rl/env.js";
import { makePPOActor, recordPolicyRollout } from "../src/pusht/rl/common.js";
import { normalizeGroupAdvantages, trainGRPO } from "../src/pusht/rl/grpo.js";
import { computeGAE, trainPPO } from "../src/pusht/rl/ppo.js";

const finiteModel = (model) => model.params.every((buffer) => buffer.every(Number.isFinite));
const initializedActor = makePPOActor(RL_OBSERVATION_SIZE, 16, 1, createRandom(19));
const outputWeights = initializedActor.mean.weights.at(-1);
const outputWidth = initializedActor.mean.sizes.at(-2);
let outputDot = 0;
let outputNorm0 = 0;
let outputNorm1 = 0;
for (let index = 0; index < outputWidth; index++) {
  outputDot += outputWeights[index] * outputWeights[outputWidth + index];
  outputNorm0 += outputWeights[index] ** 2;
  outputNorm1 += outputWeights[outputWidth + index] ** 2;
}
if (Math.abs(outputDot) > 1e-7 || Math.abs(Math.sqrt(outputNorm0) - 0.01) > 1e-6 ||
    Math.abs(Math.sqrt(outputNorm1) - 0.01) > 1e-6) {
  throw new Error("PPO actor output layer is not orthogonal with gain 0.01");
}
const normalizedAction = (dx, dy) => Float32Array.of(dx / MAX_PUSHER_SPEED, dy / MAX_PUSHER_SPEED);
const decodedAction = cartesianActionToDelta(Float32Array.of(1, 1));
const diagonal = MAX_PUSHER_SPEED / Math.sqrt(2);
if (Math.abs(decodedAction[0] - diagonal) > 1e-7 || Math.abs(decodedAction[1] - diagonal) > 1e-7) {
  throw new Error("diagonal action was not projected onto the unit disk");
}
const scaledRewardEnv = new PushTRLEnv({ distanceRewardScale: 3 });
if (Math.abs(scaledRewardEnv.pusherDistanceRewardScale * MAX_PUSHER_SPEED - 0.01) > 1e-12) {
  throw new Error("a full-speed pusher step is not scaled to 0.01");
}
if (scaledRewardEnv.rewardShaping.pusherWall || scaledRewardEnv.rewardShaping.blockWall) {
  throw new Error("wall reward components do not default to off");
}
if (!scaledRewardEnv.rewardShaping.stepPenalty) throw new Error("step penalty does not default to on");
if (scaledRewardEnv.rewardWeights.blockDistance !== 3 ||
    Math.abs(scaledRewardEnv.rewardWeights.pusherDistance - 0.01 / MAX_PUSHER_SPEED) > 1e-12 ||
    scaledRewardEnv.rewardWeights.orientation !== 1 || scaledRewardEnv.rewardWeights.completion !== 1 ||
    scaledRewardEnv.rewardWeights.closeness !== 1 || scaledRewardEnv.rewardWeights.pusherWall !== -1 ||
    scaledRewardEnv.rewardWeights.blockWall !== -10 || scaledRewardEnv.rewardWeights.inactivity !== -1 ||
    scaledRewardEnv.rewardWeights.stepPenalty !== -0.01) {
  throw new Error("reward weights do not preserve the previous defaults");
}
const noShapingEnv = new PushTRLEnv({
  seed: 8,
  curriculum: false,
  rewardShaping: {
    blockDistance: false,
    pusherDistance: false,
    orientation: false,
    closeness: false,
    stepPenalty: false,
  },
});
noShapingEnv.reset();
const towardGoalX = noShapingEnv.world.goal.x - noShapingEnv.world.pusher.x;
const towardGoalY = noShapingEnv.world.goal.y - noShapingEnv.world.pusher.y;
const towardGoalLength = Math.hypot(towardGoalX, towardGoalY);
const noShapingTransition = noShapingEnv.step(normalizedAction(
  towardGoalX / towardGoalLength * MAX_PUSHER_SPEED,
  towardGoalY / towardGoalLength * MAX_PUSHER_SPEED,
));
if (noShapingTransition.pusherDistanceProgress <= 0 || noShapingTransition.reward !== 0) {
  throw new Error("disabled reward shaping still changed the reward");
}
const weightedStepEnv = new PushTRLEnv({
  seed: 8,
  curriculum: false,
  rewardShaping: {
    blockDistance: false,
    pusherDistance: false,
    orientation: false,
    closeness: false,
  },
  rewardWeights: { stepPenalty: -0.25 },
});
weightedStepEnv.reset();
const weightedStepTransition = weightedStepEnv.step(Float32Array.of(1, 0));
if (weightedStepTransition.stepPenalty !== -0.25 || weightedStepTransition.reward !== -0.25) {
  throw new Error("custom reward weight did not scale the enabled component");
}
const stationaryShapingEnv = new PushTRLEnv({ seed: 8, curriculum: false });
stationaryShapingEnv.reset();
const stationaryTransition = stationaryShapingEnv.step(Float32Array.of(0, 0));
if (stationaryTransition.distanceShapingReward !== 0 ||
    stationaryTransition.pusherDistanceShapingReward !== 0 ||
    stationaryTransition.orientationShapingReward !== 0 || stationaryTransition.stepPenalty !== -0.01 ||
    stationaryTransition.reward !== -0.01) {
  throw new Error("stationary action earned shaping reward");
}
const onlyInactivity = {
  completion: false,
  blockDistance: false,
  pusherDistance: false,
  orientation: false,
  closeness: false,
  pusherWall: false,
  blockWall: false,
  inactivity: true,
  stepPenalty: false,
};
const inactivityEnv = new PushTRLEnv({
  seed: 18,
  horizon: 100,
  curriculum: false,
  rewardShaping: onlyInactivity,
});
inactivityEnv.reset();
let inactivityTransition;
for (let step = 1; step <= MAX_STATIONARY_STEPS; step++) {
  inactivityTransition = inactivityEnv.step(Float32Array.of(0, 0));
  if (step < MAX_STATIONARY_STEPS && inactivityTransition.done) {
    throw new Error(`inactivity terminated after only ${step} steps`);
  }
}
if (!inactivityTransition.done || !inactivityTransition.stalled ||
    inactivityTransition.inactivityPenalty !== -1 || inactivityTransition.reward !== -1) {
  throw new Error("five stationary steps did not terminate with exactly -1 reward");
}
const resetInactivityEnv = new PushTRLEnv({
  seed: 18,
  horizon: 100,
  curriculum: false,
  rewardShaping: onlyInactivity,
});
resetInactivityEnv.reset();
for (let step = 0; step < MAX_STATIONARY_STEPS - 1; step++) {
  resetInactivityEnv.step(Float32Array.of(0, 0));
}
const movementTransition = resetInactivityEnv.step(Float32Array.of(1, 0));
if (movementTransition.done || movementTransition.stationarySteps !== 0 ||
    movementTransition.pusherDisplacement <= 0) {
  throw new Error("actual pusher movement did not reset the inactivity counter");
}
for (let step = 1; step < MAX_STATIONARY_STEPS; step++) {
  const transition = resetInactivityEnv.step(Float32Array.of(0, 0));
  if (transition.done) throw new Error("inactivity counter did not restart after movement");
}
const disabledInactivityEnv = new PushTRLEnv({
  seed: 18,
  horizon: 100,
  curriculum: false,
  rewardShaping: { ...onlyInactivity, inactivity: false },
});
disabledInactivityEnv.reset();
let disabledInactivityTransition;
for (let step = 0; step < MAX_STATIONARY_STEPS; step++) {
  disabledInactivityTransition = disabledInactivityEnv.step(Float32Array.of(0, 0));
}
if (!disabledInactivityTransition.stalled || !disabledInactivityTransition.done ||
    disabledInactivityTransition.inactivityPenalty !== 0 || disabledInactivityTransition.reward !== 0) {
  throw new Error("disabling the inactivity reward changed its termination semantics");
}
const pusherShapingEnv = new PushTRLEnv({
  seed: 8,
  curriculum: false,
  rewardShaping: { blockDistance: false, orientation: false, closeness: false, stepPenalty: false },
});
pusherShapingEnv.reset();
const pusherDistanceBefore = pusherShapingEnv.pusherDistance;
const pusherDirectionX = pusherShapingEnv.world.goal.x - pusherShapingEnv.world.pusher.x;
const pusherDirectionY = pusherShapingEnv.world.goal.y - pusherShapingEnv.world.pusher.y;
const pusherDirectionLength = Math.hypot(pusherDirectionX, pusherDirectionY);
const pusherShapingTransition = pusherShapingEnv.step(normalizedAction(
  pusherDirectionX / pusherDirectionLength * MAX_PUSHER_SPEED,
  pusherDirectionY / pusherDirectionLength * MAX_PUSHER_SPEED,
));
const expectedPusherShaping = pusherShapingEnv.pusherDistanceRewardScale *
  (pusherDistanceBefore - pusherShapingTransition.pusherDistance);
if (Math.abs(pusherShapingTransition.reward - expectedPusherShaping) > 1e-6 ||
    Math.abs(pusherShapingTransition.pusherDistanceProgress *
      pusherShapingEnv.pusherDistanceRewardScale - 0.01) > 1e-6) {
  throw new Error("pusher reward does not match the 0.01 full-step progress coefficient");
}
const awayShapingEnv = new PushTRLEnv({
  seed: 8,
  curriculum: false,
  rewardShaping: { blockDistance: false, orientation: false, closeness: false, stepPenalty: false },
});
awayShapingEnv.reset();
const awayTransition = awayShapingEnv.step(normalizedAction(
  -pusherDirectionX / pusherDirectionLength * MAX_PUSHER_SPEED,
  -pusherDirectionY / pusherDirectionLength * MAX_PUSHER_SPEED,
));
if (Math.abs(awayTransition.reward + 0.01) > 1e-6) {
  throw new Error(`full-speed movement away from the goal earned ${awayTransition.reward}`);
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

// Environment contract: every shaping term is signed transition progress, and
// all enabled reward components sum exactly.
const env = new PushTRLEnv({ seed: 10, horizon: 2200, maxStationarySteps: Infinity });
env.setTrainingProgress(1);
let observation = env.reset();
if (observation.length !== RL_OBSERVATION_SIZE || RL_ACTION_SIZE !== 2) throw new Error("wrong RL shape");
const expert = new ScriptedExpert({ random: createRandom(11), tieBreak: "cw" });
const action = new Float32Array(2);
let completionRewards = 0;
let distanceReward = 0;
let pusherDistanceReward = 0;
let orientationReward = 0;
let finalClosenessReward = 0;
let stepPenaltyReward = 0;
let totalReward = 0;
let solved = false;
for (let step = 0; step < 2200; step++) {
  const [targetX, targetY] = expert.act(env.world);
  action.set(normalizedAction(targetX - env.world.pusher.x, targetY - env.world.pusher.y));
  const transition = env.step(action);
  observation = transition.observation;
  completionRewards += transition.completionReward;
  distanceReward += transition.distanceShapingReward;
  pusherDistanceReward += transition.pusherDistanceShapingReward;
  orientationReward += transition.orientationShapingReward;
  finalClosenessReward += transition.finalClosenessReward;
  stepPenaltyReward += transition.stepPenalty;
  totalReward += transition.reward;
  if (transition.done) { solved = transition.success; break; }
}
console.log(
  `environment: obs ${observation.length}, action ${RL_ACTION_SIZE}, solved ${solved}, ` +
  `completion ${completionRewards}, distance reward ${distanceReward.toFixed(4)}, ` +
  `pusher reward ${pusherDistanceReward.toFixed(4)}, ` +
  `orientation reward ${orientationReward.toFixed(4)}, closeness ${finalClosenessReward.toFixed(4)}, ` +
  `step cost ${stepPenaltyReward.toFixed(4)}, ` +
  `total ${totalReward.toFixed(4)}`,
);
if (!solved || completionRewards !== 1) throw new Error("completion reward is wrong");
if (Math.abs(totalReward - (1 + finalClosenessReward + distanceReward +
    pusherDistanceReward + orientationReward + stepPenaltyReward)) > 1e-5) {
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
if (Math.abs(orientationTransition.orientationShapingReward - 0.5) > 1e-6) {
  throw new Error("orientation shaping does not equal signed angular progress");
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
  throw new Error("terminal closeness reward does not match final shape-overlap coverage");
}

// The pusher-wall toggle links its penalty and termination behavior.
const wallEnv = new PushTRLEnv({
  seed: 15,
  horizon: 100,
  curriculum: false,
  maxStationarySteps: Infinity,
});
wallEnv.reset();
let wallTransition = null;
for (let step = 0; step < 20 && !wallTransition?.wallContact; step++) {
  wallTransition = wallEnv.step(Float32Array.of(-1, 0));
}
console.log(
  `wall contact: done ${wallTransition?.done}, contact ${wallTransition?.wallContact}, reward ${wallTransition?.reward}`,
);
if (!wallTransition?.wallContact || wallTransition.wallTermination || wallTransition.done) {
  throw new Error("disabled pusher-wall component terminated the episode");
}
const expectedWallReward = wallTransition.distanceShapingReward +
  wallTransition.pusherDistanceShapingReward + wallTransition.orientationShapingReward +
  wallTransition.stepPenalty;
if (wallTransition.wallPenalty !== 0 || Math.abs(wallTransition.reward - expectedWallReward) > 1e-9) {
  throw new Error("disabled pusher-wall penalty changed the reward");
}

const terminatingWallEnv = new PushTRLEnv({
  seed: 15,
  horizon: 100,
  curriculum: false,
  maxStationarySteps: Infinity,
  rewardShaping: { pusherWall: true },
});
terminatingWallEnv.reset();
let terminatingWallTransition = null;
for (let step = 0; step < 20 && !terminatingWallTransition?.done; step++) {
  terminatingWallTransition = terminatingWallEnv.step(Float32Array.of(-1, 0));
}
if (!terminatingWallTransition?.wallContact || !terminatingWallTransition.wallTermination ||
    !terminatingWallTransition.done || terminatingWallTransition.wallPenalty !== -1 ||
    terminatingWallTransition.success || terminatingWallTransition.truncated) {
  throw new Error("enabled pusher-wall component did not penalize and terminate");
}

// The block-wall toggle likewise links its -10 penalty and termination behavior.
const blockWallEnv = new PushTRLEnv({
  seed: 17,
  horizon: 100,
  curriculum: false,
  rewardShaping: {
    blockDistance: false,
    pusherDistance: false,
    orientation: false,
    closeness: false,
    blockWall: true,
    stepPenalty: false,
  },
});
blockWallEnv.reset();
const minimumLocalX = Math.min(...blockWallEnv.world.blockCorners().flat().map(([x]) => x - blockWallEnv.world.block.x));
blockWallEnv.world.block.x = WALL_THICKNESS - minimumLocalX;
blockWallEnv.distance = blockWallEnv.blockGoalDistance();
blockWallEnv.orientationError = blockWallEnv.blockGoalOrientationError();
const blockWallTransition = blockWallEnv.step(Float32Array.of(0, 0));
console.log(
  `block wall: contact ${blockWallTransition.blockWallContact}, penalty ${blockWallTransition.blockWallPenalty}, ` +
  `done ${blockWallTransition.done}`,
);
if (!blockWallTransition.blockWallContact || !blockWallTransition.blockWallTermination ||
    blockWallTransition.blockWallPenalty !== -10 || blockWallTransition.reward !== -10 ||
    !blockWallTransition.done || blockWallTransition.success || blockWallTransition.truncated) {
  throw new Error("enabled block-wall component did not penalize and terminate");
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

// Reference GAE calculation for two interleaved environments. Env 0 terminates
// on its second transition, so its trace must not leak into the reset episode;
// env 1 remains live and bootstraps from its final critic value.
const gaeRewards = Float32Array.of(1, 10, 2, 20, 3, 30);
const gaeDones = Uint8Array.of(0, 0, 1, 0, 0, 0);
const gaeValues = Float32Array.of(0.5, 5, 0.6, 6, 0.7, 7);
const gaeEnvironments = Uint16Array.of(0, 1, 0, 1, 0, 1);
const gae = computeGAE({
  rewards: gaeRewards,
  dones: gaeDones,
  values: gaeValues,
  environmentIndices: gaeEnvironments,
  bootstrapValues: Float32Array.of(0.8, 8),
  gamma: 0.9,
  gaeLambda: 0.5,
});
const expectedGAE = [
  1 + 0.9 * 0.6 - 0.5 + 0.9 * 0.5 * (2 - 0.6),
  10 + 0.9 * 6 - 5 + 0.9 * 0.5 * (20 + 0.9 * 7 - 6 + 0.9 * 0.5 * (30 + 0.9 * 8 - 7)),
  2 - 0.6,
  20 + 0.9 * 7 - 6 + 0.9 * 0.5 * (30 + 0.9 * 8 - 7),
  3 + 0.9 * 0.8 - 0.7,
  30 + 0.9 * 8 - 7,
];
for (let index = 0; index < expectedGAE.length; index++) {
  if (Math.abs(gae.advantages[index] - expectedGAE[index]) > 1e-5 ||
      Math.abs(gae.returns[index] - (expectedGAE[index] + gaeValues[index])) > 1e-5) {
    throw new Error(`GAE mismatch at interleaved transition ${index}`);
  }
}
console.log("GAE: interleaved environments, terminal trace cut, and final bootstrap match reference");

// Short algorithm smoke runs catch non-finite losses and buffer mistakes. They
// are not expected to solve the task in a few hundred interactions.
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
if (ppo.actor.actionSquash !== "clip" || ppo.actor.logStd.length !== 2) {
  throw new Error("PPO actor is not an unsquashed Normal with global log standard deviation");
}
const recordedRollout = recordPolicyRollout(ppo.actor, new PushTRLEnv({ seed: 22, horizon: 80 }), {
  estimateValue(observation) {
    ppo.critic.inputBuffer().set(observation, 0);
    return ppo.critic.forward(1)[0];
  },
});
console.log(`recorded rollout: ${recordedRollout.count} frames, ${recordedRollout.frames.length} values`);
if (recordedRollout.frames.length !== recordedRollout.count * 12 || !recordedRollout.frames.every(Number.isFinite)) {
  throw new Error("recorded rollout action distribution has the wrong shape or non-finite values");
}
if (recordedRollout.squashed) throw new Error("PPO rollout was marked as tanh-squashed");
let recordedReward = 0;
for (let frame = 0; frame < recordedRollout.count; frame++) {
  recordedReward += recordedRollout.frames[frame * 12 + 11];
}
if (Math.abs(recordedReward - recordedRollout.return) > 1e-6) {
  throw new Error("per-frame rewards do not sum to rollout return");
}
const sampledRollout = recordPolicyRollout(ppo.actor, new PushTRLEnv({ seed: 23, horizon: 80 }), {
  random: createRandom(24),
  deterministic: false,
});
let sampledTravel = 0;
for (let frame = 1; frame < sampledRollout.count; frame++) {
  const before = (frame - 1) * 12;
  const after = frame * 12;
  sampledTravel += Math.hypot(
    sampledRollout.frames[after] - sampledRollout.frames[before],
    sampledRollout.frames[after + 1] - sampledRollout.frames[before + 1],
  );
}
console.log(`sampled rollout: ${sampledRollout.count} frames, pusher travel ${sampledTravel.toFixed(4)}`);
if (sampledTravel <= 0) throw new Error("sampled rollout did not execute policy noise");

const relative = normalizeGroupAdvantages(Float32Array.of(1, 2, 3));
const relativeMean = relative.advantages.reduce((sum, value) => sum + value, 0) / 3;
const relativeVariance = relative.advantages.reduce((sum, value) => sum + value * value, 0) / 3;
if (Math.abs(relativeMean) > 1e-6 || Math.abs(relativeVariance - 1) > 1e-6 ||
    relative.mean !== 2 || Math.abs(relative.standardDeviation - Math.sqrt(2 / 3)) > 1e-6) {
  throw new Error("GRPO group-return normalization is wrong");
}
const tied = normalizeGroupAdvantages(Float32Array.of(4, 4));
if (tied.advantages[0] !== 0 || tied.advantages[1] !== 0) {
  throw new Error("GRPO tied returns should have zero relative advantage");
}

let grpoCheckpoints = 0;
let grpoGroupPaths = 0;
let grpoGroupAdvantages = 0;
const grpo = trainGRPO({
  envs: Array.from({ length: 4 }, () => new PushTRLEnv({ seed: 30, horizon: 80 })),
  random: createRandom(31),
  totalSteps: 256,
  batchSize: 32,
  groupSize: 4,
  epochs: 1,
  progressEvery: 200,
  width: 16,
  onCheckpoint(progress, models) {
    if (progress.steps !== 200) throw new Error(`unexpected GRPO checkpoint ${progress.steps}`);
    if (models.groupPaths.length !== 4 || models.groupPaths.some((path) => path.length < 4)) {
      throw new Error("GRPO checkpoint did not include the rollout group paths");
    }
    if (models.groupAdvantages.length !== 4 || models.groupAdvantages.some((value) => !Number.isFinite(value))) {
      throw new Error("GRPO checkpoint did not include finite group advantages");
    }
    grpoGroupPaths = models.groupPaths.length;
    grpoGroupAdvantages = models.groupAdvantages.length;
    grpoCheckpoints += 1;
  },
});
console.log(`GRPO smoke: ${grpo.steps} steps, finite ${finiteModel(grpo.actor)}`);
if (!finiteModel(grpo.actor)) throw new Error("GRPO produced non-finite parameters");
if (grpoCheckpoints !== 1) throw new Error(`expected one GRPO checkpoint, got ${grpoCheckpoints}`);
if (grpoGroupPaths !== 4) throw new Error("GRPO rollout group was not exposed to the viewer");
if (grpoGroupAdvantages !== 4) throw new Error("GRPO rollout advantages were not exposed to the viewer");
if (grpo.actor.actionSquash !== "clip" || grpo.actor.logStd.length !== 2) {
  throw new Error("GRPO actor is not an unsquashed Normal with global log standard deviation");
}
