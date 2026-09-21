// Minimal single-frame RL interface over PushWorld.
//
// Observation: the current normalized simulator observation (no frame stack).
// Action: normalized dx/dy in [-1, 1], scaled to one maximum pusher step.
// Reward: 1 only on task completion, otherwise 0.

import { PushWorld, createRandom, MAX_PUSHER_SPEED, SUCCESS_COVERAGE } from "../sim.js";
import { normalizeObservation } from "../policy.js";

export const RL_OBSERVATION_SIZE = 11;
export const RL_ACTION_SIZE = 2;

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

export class PushTRLEnv {
  constructor({ seed = 1, horizon = 600 } = {}) {
    this.random = createRandom(seed);
    this.world = new PushWorld({ random: this.random, obstacleCount: 0 });
    this.horizon = horizon;
    this.observation = new Float32Array(RL_OBSERVATION_SIZE);
    this.episodeReturn = 0;
    this.episodeSteps = 0;
  }

  observe() {
    normalizeObservation(this.world.writeObservation(), this.observation);
    return this.observation;
  }

  reset() {
    this.world.reset({ obstacleCount: 0 });
    this.episodeReturn = 0;
    this.episodeSteps = 0;
    return this.observe();
  }

  step(action) {
    const dx = clamp(action[0], -1, 1) * MAX_PUSHER_SPEED;
    const dy = clamp(action[1], -1, 1) * MAX_PUSHER_SPEED;
    this.world.step(this.world.pusher.x + dx, this.world.pusher.y + dy, 0);
    this.episodeSteps += 1;
    const success = this.world.coverage() >= SUCCESS_COVERAGE;
    const reward = success ? 1 : 0;
    const truncated = !success && this.episodeSteps >= this.horizon;
    const done = success || truncated;
    this.episodeReturn += reward;
    return {
      observation: this.observe(),
      reward,
      done,
      success,
      truncated,
      coverage: this.world.coverage(),
    };
  }
}
