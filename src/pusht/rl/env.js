// Minimal single-frame RL interface over PushWorld.
//
// Observation: the current normalized simulator observation (no frame stack).
// Action: normalized dx/dy in [-1, 1], scaled to one maximum pusher step.
// Reward: progress in block position and orientation, 0.1x progress in pusher
// distance to the goal, final overlap closeness, +1 on completion, -1 when
// the pusher hits a wall, and -10 when the block touches a wall.

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
const PUSHER_WALL_MIN = WALL_THICKNESS + PUSHER_RADIUS;
const PUSHER_WALL_MAX = 1 - PUSHER_WALL_MIN;
const WALL_EPSILON = 1e-9;
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
    distanceRewardScale = 1,
    pusherDistanceRewardScale = distanceRewardScale * 0.1,
    orientationRewardScale = 1,
    rewardShaping = {},
    curriculum = true,
  } = {}) {
    this.random = createRandom(seed);
    this.world = new PushWorld({ random: this.random, obstacleCount: 0 });
    this.horizon = horizon;
    this.distanceRewardScale = distanceRewardScale;
    this.pusherDistanceRewardScale = pusherDistanceRewardScale;
    this.orientationRewardScale = orientationRewardScale;
    this.rewardShaping = {
      blockDistance: rewardShaping.blockDistance ?? true,
      pusherDistance: rewardShaping.pusherDistance ?? true,
      orientation: rewardShaping.orientation ?? true,
      closeness: rewardShaping.closeness ?? true,
    };
    this.curriculum = curriculum;
    this.trainingProgress = 0;
    this.observation = new Float32Array(RL_OBSERVATION_SIZE);
    this.episodeReturn = 0;
    this.episodeSteps = 0;
    this.distance = 0;
    this.pusherDistance = 0;
    this.orientationError = 0;
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
    return this.observe();
  }

  step(action) {
    const dx = clamp(action[0], -1, 1) * MAX_PUSHER_SPEED;
    const dy = clamp(action[1], -1, 1) * MAX_PUSHER_SPEED;
    this.world.step(this.world.pusher.x + dx, this.world.pusher.y + dy, 0);
    this.episodeSteps += 1;
    const distance = this.blockGoalDistance();
    const distanceProgress = this.distance - distance;
    this.distance = distance;
    const pusherDistance = this.pusherGoalDistance();
    const pusherDistanceProgress = this.pusherDistance - pusherDistance;
    this.pusherDistance = pusherDistance;
    const orientationError = this.blockGoalOrientationError();
    const orientationProgress = this.orientationError - orientationError;
    this.orientationError = orientationError;
    const coverage = this.world.coverage();
    const success = coverage >= SUCCESS_COVERAGE;
    const wallContact = !success && this.pusherTouchesWall();
    const blockWallContact = this.blockTouchesWall();
    const completionReward = success ? 1 : 0;
    const wallPenalty = wallContact ? -1 : 0;
    const blockWallPenalty = blockWallContact ? -10 : 0;
    const truncated = !success && !wallContact && this.episodeSteps >= this.horizon;
    const done = success || wallContact || truncated;
    const finalClosenessReward = done && !wallContact && this.rewardShaping.closeness ? coverage : 0;
    const reward = completionReward + finalClosenessReward + wallPenalty + blockWallPenalty +
      (this.rewardShaping.blockDistance ? this.distanceRewardScale * distanceProgress : 0) +
      (this.rewardShaping.pusherDistance ? this.pusherDistanceRewardScale * pusherDistanceProgress : 0) +
      (this.rewardShaping.orientation ? this.orientationRewardScale * orientationProgress : 0);
    this.episodeReturn += reward;
    return {
      observation: this.observe(),
      reward,
      completionReward,
      finalClosenessReward,
      wallPenalty,
      blockWallPenalty,
      distanceProgress,
      distance,
      pusherDistanceProgress,
      pusherDistance,
      orientationProgress,
      orientationError,
      done,
      success,
      wallContact,
      blockWallContact,
      truncated,
      coverage,
    };
  }
}
