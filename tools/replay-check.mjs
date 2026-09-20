// Restoring a snapshot and replaying the same actions must reproduce the state
// exactly, or "Reset" lies and recorded episodes are not reproducible.
import { PushWorld, createRandom } from "../src/pusht/sim.js";
import { ScriptedExpert } from "../src/pusht/expert.js";

const random = createRandom(31337);
const world = new PushWorld({ random, obstacleCount: 3 });
let worstPosition = 0;
let worstAngle = 0;

for (let trial = 0; trial < 40; trial++) {
  world.reset();
  const start = world.snapshot();

  const expert = new ScriptedExpert({ random, tieBreak: "cw" });
  const actions = [];
  for (let step = 0; step < 220; step++) {
    const [x, y] = expert.act(world);
    actions.push([x, y]);
    world.step(x, y);
  }
  const first = { ...world.block, pusherX: world.pusher.x, pusherY: world.pusher.y };

  world.restore(start);
  for (const [x, y] of actions) world.step(x, y);
  const second = { ...world.block, pusherX: world.pusher.x, pusherY: world.pusher.y };

  worstPosition = Math.max(
    worstPosition,
    Math.hypot(first.x - second.x, first.y - second.y),
    Math.hypot(first.pusherX - second.pusherX, first.pusherY - second.pusherY),
  );
  worstAngle = Math.max(worstAngle, Math.abs(first.angle - second.angle));
}

console.log("40 replays of 220 steps each");
console.log("worst position divergence", worstPosition.toExponential(3));
console.log("worst angle divergence   ", worstAngle.toExponential(3));
console.log(worstPosition === 0 && worstAngle === 0 ? "bit-exact replay" : "DIVERGENT");
