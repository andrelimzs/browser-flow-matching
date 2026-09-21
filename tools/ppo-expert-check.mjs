// Verifies that the committed PPO checkpoint can drive the exact fixed task
// used by Flow BC through the absolute-target expert adapter.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SavedPPOExpert } from "../src/pusht/ppo-expert.js";
import { MAX_PUSHER_SPEED, PushWorld, createRandom } from "../src/pusht/sim.js";

const checkpoint = JSON.parse(readFileSync(new URL("../src/pusht/ppo-expert.json", import.meta.url)));
assert.equal(checkpoint.algorithm, "ppo");
assert.equal(checkpoint.totalSteps, 1_000_000);

let successes = 0;
let totalSteps = 0;
for (let episode = 0; episode < 10; episode++) {
  const world = new PushWorld({ random: createRandom(1000 + episode), obstacleCount: 0 });
  const expert = new SavedPPOExpert(checkpoint);
  expert.reset();
  for (let step = 0; step < checkpoint.horizon && !world.succeeded(); step++) {
    const beforeX = world.pusher.x;
    const beforeY = world.pusher.y;
    const [targetX, targetY, lift] = expert.act(world);
    assert.equal(lift, 0);
    assert.ok(Math.hypot(targetX - beforeX, targetY - beforeY) <= MAX_PUSHER_SPEED + 1e-7);
    world.step(targetX, targetY, lift);
  }
  totalSteps += world.steps;
  if (world.succeeded()) successes += 1;
}

assert.equal(successes, 10, `saved PPO expert solved only ${successes}/10 fixed episodes`);
console.log(`PPO Flow BC expert: ${successes}/10 solved, ${Math.round(totalSteps / 10)} mean steps`);
