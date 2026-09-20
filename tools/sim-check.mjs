// Sanity check: random pushing must stay finite, stay in bounds, and actually
// move the block. Run with `node tools/sim-check.mjs`.
import { PushWorld, createRandom, TEE, BLOCK_UNIT } from "../src/pusht/sim.js";

const random = createRandom(7);
const world = new PushWorld({ random, obstacleCount: 2 });

console.log("tee: unit", BLOCK_UNIT, "area", TEE.area.toFixed(5), "radius", TEE.radius.toFixed(4));
console.log("characteristic c^2", TEE.characteristicSquared.toFixed(6), "c", Math.sqrt(TEE.characteristicSquared).toFixed(4));

let worst = { finite: true, minX: 1, maxX: 0, minY: 1, maxY: 0 };
let moved = 0;
const episodes = 200;
const horizon = 200;
const startTime = process.hrtime.bigint();
let steps = 0;

for (let episode = 0; episode < episodes; episode++) {
  world.reset();
  const startBlock = { ...world.block };
  for (let step = 0; step < horizon; step++) {
    // Random walk of the pusher target across the arena.
    const target = [random(), random()];
    world.step(target[0], target[1]);
    steps += 1;
    const { x, y, angle } = world.block;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(angle)) worst.finite = false;
    worst.minX = Math.min(worst.minX, x); worst.maxX = Math.max(worst.maxX, x);
    worst.minY = Math.min(worst.minY, y); worst.maxY = Math.max(worst.maxY, y);
  }
  if (Math.hypot(world.block.x - startBlock.x, world.block.y - startBlock.y) > 0.02) moved += 1;
}

const elapsed = Number(process.hrtime.bigint() - startTime) / 1e9;
console.log("finite:", worst.finite);
console.log("block centre range x", worst.minX.toFixed(3), worst.maxX.toFixed(3), "y", worst.minY.toFixed(3), worst.maxY.toFixed(3));
console.log("episodes where block moved >0.02:", moved, "/", episodes);
console.log("steps", steps, "in", elapsed.toFixed(3), "s =", Math.round(steps / elapsed), "steps/s");
