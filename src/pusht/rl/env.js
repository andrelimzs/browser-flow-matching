// Minimal single-frame RL interface over PushWorld.
//
// Observation: the current normalized simulator observation (no frame stack).
// Action: dx/dy projected onto the unit disk, then scaled by MAX_PUSHER_SPEED.
// Reward: signed progress in block distance, pusher distance, and orientation;
// optional terminal/closeness and wall components.

import {
  FIXED_BLOCK_START,
  FIXED_GOAL,
  FIXED_PUSHER_START,
  PushWorld,
  TEE,
  createRandom,
  MAX_PUSHER_SPEED,
  PUSHER_RADIUS,
  SUCCESS_COVERAGE,
  WALL_THICKNESS,
} from "../sim.js";
import { normalizeObservation } from "../policy.js";
import { rectCorners, wrapAngle } from "../geometry.js";

export const RL_OBSERVATION_SIZE = 11;
export const RL_ACTION_SIZE = 2;
export const CURRICULUM_FINAL_PROGRESS = 0.7;
export const CURRICULUM_INITIAL_DISTANCE_FRACTION = 0.25;

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

export function cartesianActionToDelta(action, out = new Float32Array(2)) {
  const rawX = Number.isFinite(action[0]) ? action[0] : 0;
  const rawY = Number.isFinite(action[1]) ? action[1] : 0;
  const magnitude = Math.hypot(rawX, rawY);
  const scale = MAX_PUSHER_SPEED / Math.max(1, magnitude);
  out[0] = rawX * scale;
  out[1] = rawY * scale;
  return out;
}
const PUSHER_WALL_MIN = WALL_THICKNESS + PUSHER_RADIUS;
const PUSHER_WALL_MAX = 1 - PUSHER_WALL_MIN;
const WALL_EPSILON = 1e-9;
const PUSHER_MOVEMENT_EPSILON = 1e-6;
export const MAX_STATIONARY_STEPS = 5;
const BLOCK_WALL_EPSILON = 1e-6;
const BLOCK_LOCAL_CORNERS = TEE.parts.flatMap((part) => rectCorners(part));
const BLOCK_MIN_X = WALL_THICKNESS - Math.min(...TEE.parts.map((part) => part.minX));
const BLOCK_MAX_X = 1 - WALL_THICKNESS - Math.max(...TEE.parts.map((part) => part.maxX));
const BLOCK_MIN_Y = WALL_THICKNESS - Math.min(...TEE.parts.map((part) => part.minY));
const BLOCK_MAX_Y = 1 - WALL_THICKNESS - Math.max(...TEE.parts.map((part) => part.maxY));
const FINAL_BLOCK_GOAL_DISTANCE = Math.hypot(
  FIXED_BLOCK_START.x - FIXED_GOAL.x,
  FIXED_BLOCK_START.y - FIXED_GOAL.y,
);
const PUSHER_BLOCK_START_DISTANCE = Math.hypot(
  FIXED_PUSHER_START.x - FIXED_BLOCK_START.x,
  FIXED_PUSHER_START.y - FIXED_BLOCK_START.y,
);

export class PushTRLEnv {
  constructor({
    seed = 1,
    horizon = 200,
    maxStationarySteps = MAX_STATIONARY_STEPS,
    distanceRewardScale = 1,
    pusherDistanceRewardScale = 0.01 / MAX_PUSHER_SPEED,
    orientationRewardScale = 1,
    rewardShaping = {},
    curriculum = true,
  } = {}) {
    this.random = createRandom(seed);
    this.world = new PushWorld({ random: this.random, obstacleCount: 0 });
    this.horizon = horizon;
    this.maxStationarySteps = maxStationarySteps;
    this.distanceRewardScale = distanceRewardScale;
    this.pusherDistanceRewardScale = pusherDistanceRewardScale;
    this.orientationRewardScale = orientationRewardScale;
    this.rewardShaping = {
      completion: rewardShaping.completion ?? true,
      blockDistance: rewardShaping.blockDistance ?? true,
      pusherDistance: rewardShaping.pusherDistance ?? true,
      orientation: rewardShaping.orientation ?? true,
      closeness: rewardShaping.closeness ?? true,
      pusherWall: rewardShaping.pusherWall ?? false,
      blockWall: rewardShaping.blockWall ?? false,
      inactivity: rewardShaping.inactivity ?? true,
    };
    this.curriculum = curriculum;
    this.trainingProgress = 0;
    this.observation = new Float32Array(RL_OBSERVATION_SIZE);
    this.episodeReturn = 0;
    this.episodeSteps = 0;
    this.distance = 0;
    this.pusherDistance = 0;
    this.orientationError = 0;
    this.stationarySteps = 0;
    this.actionDelta = new Float32Array(2);
  }

  setTrainingProgress(progress) {
    this.trainingProgress = clamp(progress, 0, 1);
  }

  curriculumDistanceFraction() {
    if (!this.curriculum) return 1;
    const stage = clamp(this.trainingProgress / CURRICULUM_FINAL_PROGRESS, 0, 1);
    return CURRICULUM_INITIAL_DISTANCE_FRACTION +
      (1 - CURRICULUM_INITIAL_DISTANCE_FRACTION) * stage;
  }

  randomizeBlockPosition(distance) {
    for (let attempt = 0; attempt < 256; attempt++) {
      const bearing = this.random() * Math.PI * 2;
      const directionX = Math.cos(bearing);
      const directionY = Math.sin(bearing);
      const x = this.world.goal.x + directionX * distance;
      const y = this.world.goal.y + directionY * distance;
      const pusherX = x + directionX * PUSHER_BLOCK_START_DISTANCE;
      const pusherY = y + directionY * PUSHER_BLOCK_START_DISTANCE;
      if (x < BLOCK_MIN_X || x > BLOCK_MAX_X || y < BLOCK_MIN_Y || y > BLOCK_MAX_Y) continue;
      if (pusherX < PUSHER_WALL_MIN || pusherX > PUSHER_WALL_MAX ||
          pusherY < PUSHER_WALL_MIN || pusherY > PUSHER_WALL_MAX) continue;
      this.world.block.x = x;
      this.world.block.y = y;
      this.world.pusher.x = pusherX;
      this.world.pusher.y = pusherY;
      this.world.command.x = pusherX;
      this.world.command.y = pusherY;
      return;
    }

    // The original diagonal is valid at every curriculum radius and provides
    // a deterministic fallback if rejection sampling ever misses the arc.
    const finalDistance = FINAL_BLOCK_GOAL_DISTANCE || 1;
    this.world.block.x = this.world.goal.x +
      (FIXED_BLOCK_START.x - this.world.goal.x) * distance / finalDistance;
    this.world.block.y = this.world.goal.y +
      (FIXED_BLOCK_START.y - this.world.goal.y) * distance / finalDistance;
    const directionX = (FIXED_BLOCK_START.x - this.world.goal.x) / finalDistance;
    const directionY = (FIXED_BLOCK_START.y - this.world.goal.y) / finalDistance;
    this.world.pusher.x = this.world.block.x + directionX * PUSHER_BLOCK_START_DISTANCE;
    this.world.pusher.y = this.world.block.y + directionY * PUSHER_BLOCK_START_DISTANCE;
    this.world.command.x = this.world.pusher.x;
    this.world.command.y = this.world.pusher.y;
  }

  blockGoalDistance() {
    return Math.hypot(this.world.goal.x - this.world.block.x, this.world.goal.y - this.world.block.y);
  }

  pusherGoalDistance() {
    return Math.hypot(this.world.goal.x - this.world.pusher.x, this.world.goal.y - this.world.pusher.y);
  }

  blockGoalOrientationError() {
    return Math.abs(wrapAngle(this.world.goal.angle - this.world.block.angle)) / Math.PI;
  }

  pusherTouchesWall() {
    const { x, y } = this.world.pusher;
    return x <= PUSHER_WALL_MIN + WALL_EPSILON || x >= PUSHER_WALL_MAX - WALL_EPSILON ||
      y <= PUSHER_WALL_MIN + WALL_EPSILON || y >= PUSHER_WALL_MAX - WALL_EPSILON;
  }

  blockTouchesWall() {
    const { x, y, angle } = this.world.block;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    for (const [localX, localY] of BLOCK_LOCAL_CORNERS) {
      const worldX = x + localX * cos - localY * sin;
      const worldY = y + localX * sin + localY * cos;
      if (worldX <= WALL_THICKNESS + BLOCK_WALL_EPSILON ||
          worldX >= 1 - WALL_THICKNESS - BLOCK_WALL_EPSILON ||
          worldY <= WALL_THICKNESS + BLOCK_WALL_EPSILON ||
          worldY >= 1 - WALL_THICKNESS - BLOCK_WALL_EPSILON) return true;
    }
    return false;
  }

  observe() {
    normalizeObservation(this.world.writeObservation(), this.observation);
    return this.observation;
  }

  reset() {
    this.world.reset({ obstacleCount: 0 });
    const distanceFraction = this.curriculumDistanceFraction();
    if (this.curriculum) this.randomizeBlockPosition(FINAL_BLOCK_GOAL_DISTANCE * distanceFraction);
    this.world.previous.x = this.world.block.x;
    this.world.previous.y = this.world.block.y;
    this.world.previous.angle = this.world.block.angle;
    this.episodeReturn = 0;
    this.episodeSteps = 0;
    this.distance = this.blockGoalDistance();
    this.pusherDistance = this.pusherGoalDistance();
    this.orientationError = this.blockGoalOrientationError();
    this.stationarySteps = 0;
    return this.observe();
  }

  step(action) {
    const [dx, dy] = cartesianActionToDelta(action, this.actionDelta);
    const previousPusherX = this.world.pusher.x;
    const previousPusherY = this.world.pusher.y;
    this.world.step(this.world.pusher.x + dx, this.world.pusher.y + dy, 0);
    this.episodeSteps += 1;
    const pusherDisplacement = Math.hypot(
      this.world.pusher.x - previousPusherX,
      this.world.pusher.y - previousPusherY,
    );
    this.stationarySteps = pusherDisplacement <= PUSHER_MOVEMENT_EPSILON
      ? this.stationarySteps + 1
      : 0;
    const previousDistance = this.distance;
    const distance = this.blockGoalDistance();
    const distanceProgress = previousDistance - distance;
    const distanceShapingReward = this.distanceRewardScale * distanceProgress;
    this.distance = distance;
    const previousPusherDistance = this.pusherDistance;
    const pusherDistance = this.pusherGoalDistance();
    const pusherDistanceProgress = previousPusherDistance - pusherDistance;
    const pusherDistanceShapingReward = this.pusherDistanceRewardScale * pusherDistanceProgress;
    this.pusherDistance = pusherDistance;
    const previousOrientationError = this.orientationError;
    const orientationError = this.blockGoalOrientationError();
    const orientationProgress = previousOrientationError - orientationError;
    const orientationShapingReward = this.orientationRewardScale * orientationProgress;
    this.orientationError = orientationError;
    const coverage = this.world.coverage();
    const success = coverage >= SUCCESS_COVERAGE;
    const wallContact = !success && this.pusherTouchesWall();
    const blockWallContact = this.blockTouchesWall();
    const stalled = !success && this.stationarySteps >= this.maxStationarySteps;
    const completionReward = success && this.rewardShaping.completion ? 1 : 0;
    const wallPenalty = wallContact && this.rewardShaping.pusherWall ? -1 : 0;
    const blockWallPenalty = blockWallContact && this.rewardShaping.blockWall ? -10 : 0;
    const inactivityPenalty = stalled && this.rewardShaping.inactivity ? -1 : 0;
    const truncated = !success && !wallContact && !stalled && this.episodeSteps >= this.horizon;
    const done = success || wallContact || stalled || truncated;
    const finalClosenessReward = done && !wallContact && !stalled && this.rewardShaping.closeness
      ? Math.exp(-distance) + Math.exp(-orientationError)
      : 0;
    const reward = completionReward + finalClosenessReward + wallPenalty + blockWallPenalty +
      inactivityPenalty +
      (this.rewardShaping.blockDistance ? distanceShapingReward : 0) +
      (this.rewardShaping.pusherDistance ? pusherDistanceShapingReward : 0) +
      (this.rewardShaping.orientation ? orientationShapingReward : 0);
    this.episodeReturn += reward;
    return {
      observation: this.observe(),
      reward,
      completionReward,
      finalClosenessReward,
      wallPenalty,
      blockWallPenalty,
      inactivityPenalty,
      distanceProgress,
      distanceShapingReward,
      distance,
      pusherDistanceProgress,
      pusherDistanceShapingReward,
      pusherDistance,
      orientationProgress,
      orientationShapingReward,
      orientationError,
      pusherDisplacement,
      stationarySteps: this.stationarySteps,
      done,
      success,
      wallContact,
      blockWallContact,
      stalled,
      truncated,
      coverage,
    };
  }
}
