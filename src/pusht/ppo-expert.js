// Deterministic adapter from a saved PPO actor to the absolute pusher targets
// consumed by PushWorld and the Flow BC demonstration recorder.

import { MLP } from "../flow/mlp.js";
import { normalizeObservation } from "./policy.js";
import { cartesianActionToDelta, RL_OBSERVATION_SIZE } from "./rl/env.js";

export class SavedPPOExpert {
  constructor(checkpoint) {
    const actor = checkpoint?.actor;
    if (checkpoint?.algorithm !== "ppo" || actor?.type !== "ppo-normal") {
      throw new Error("Flow BC expert checkpoint must contain a PPO normal actor");
    }
    if (actor.mean?.sizes?.[0] !== RL_OBSERVATION_SIZE || actor.mean?.sizes?.at(-1) !== 2) {
      throw new Error(`Flow BC expert needs an ${RL_OBSERVATION_SIZE}-input, 2-output actor`);
    }

    this.model = new MLP({ sizes: actor.mean.sizes, maxBatch: 1, random: () => 0.5 });
    this.model.loadJSON(actor.mean);
    this.observation = new Float32Array(RL_OBSERVATION_SIZE);
    this.delta = new Float32Array(2);
  }

  reset() {}

  describe() {
    return "PPO policy";
  }

  act(world) {
    normalizeObservation(world.writeObservation(), this.observation);
    this.model.inputBuffer().set(this.observation);
    const mean = this.model.forward(1);
    cartesianActionToDelta(mean, this.delta);
    return [world.pusher.x + this.delta[0], world.pusher.y + this.delta[1], 0];
  }
}
