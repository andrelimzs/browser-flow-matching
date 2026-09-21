// Minimal single-frame RL interface over PushWorld.
//
// Observation: the current normalized simulator observation (no frame stack).
// Action: normalized dx/dy in [-1, 1], scaled to one maximum pusher step.
// Reward: progress in block position and orientation, final overlap closeness,
// +1 on completion, or -1 on wall contact.

import {
  PushWorld,
  createRandom,
  MAX_PUSHER_SPEED,
  PUSHER_RADIUS,
  SUCCESS_COVERAGE,
  WALL_THICKNESS,
} from "../sim.js";
import { normalizeObservation } from "../policy.js";
import { wrapAngle } from "../geometry.js";

export const RL_OBSERVATION_SIZE = 11;
export const RL_ACTION_SIZE = 2;

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const PUSHER_WALL_MIN = WALL_THICKNESS + PUSHER_RADIUS;
const PUSHER_WALL_MAX = 1 - PUSHER_WALL_MIN;
const WALL_EPSILON = 1e-9;

export class PushTRLEnv {
  constructor({ seed = 1, horizon = 600, distanceRewardScale = 1, orientationRewardScale = 1 } = {}) {
    this.random = createRandom(seed);
    this.world = new PushWorld({ random: this.random, obstacleCount: 0 });
    this.horizon = horizon;
    this.distanceRewardScale = distanceRewardScale;
    this.orientationRewardScale = orientationRewardScale;
    this.observation = new Float32Array(RL_OBSERVATION_SIZE);
    this.episodeReturn = 0;
    this.episodeSteps = 0;
    this.distance = 0;
    this.orientationError = 0;
  }

  blockGoalDistance() {
    return Math.hypot(this.world.goal.x - this.world.block.x, this.world.goal.y - this.world.block.y);
  }

  blockGoalOrientationError() {
    return Math.abs(wrapAngle(this.world.goal.angle - this.world.block.angle)) / Math.PI;
  }

  pusherTouchesWall() {
    const { x, y } = this.world.pusher;
    return x <= PUSHER_WALL_MIN + WALL_EPSILON || x >= PUSHER_WALL_MAX - WALL_EPSILON ||
      y <= PUSHER_WALL_MIN + WALL_EPSILON || y >= PUSHER_WALL_MAX - WALL_EPSILON;
  }

  observe() {
    normalizeObservation(this.world.writeObservation(), this.observation);
    return this.observation;
  }

  reset() {
    this.world.reset({ obstacleCount: 0 });
    this.episodeReturn = 0;
    this.episodeSteps = 0;
    this.distance = this.blockGoalDistance();
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
    const orientationError = this.blockGoalOrientationError();
    const orientationProgress = this.orientationError - orientationError;
    this.orientationError = orientationError;
    const coverage = this.world.coverage();
    const success = coverage >= SUCCESS_COVERAGE;
    const wallContact = !success && this.pusherTouchesWall();
    const completionReward = success ? 1 : 0;
    const wallPenalty = wallContact ? -1 : 0;
    const truncated = !success && !wallContact && this.episodeSteps >= this.horizon;
    const done = success || wallContact || truncated;
    const finalClosenessReward = done && !wallContact ? coverage : 0;
    const reward = wallContact ? wallPenalty : completionReward + finalClosenessReward +
      this.distanceRewardScale * distanceProgress +
      this.orientationRewardScale * orientationProgress;
    this.episodeReturn += reward;
    return {
      observation: this.observe(),
      reward,
      completionReward,
      finalClosenessReward,
      wallPenalty,
      distanceProgress,
      distance,
      orientationProgress,
      orientationError,
      done,
      success,
      wallContact,
      truncated,
      coverage,
    };
  }
}
