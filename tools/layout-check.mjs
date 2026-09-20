// Are obstacles actually in the way? Projects each obstacle onto the
// start->goal segment: `along` in [0,1] means between the two poses, and
// `lateral` is how far off the straight line it sits.
import { PushWorld, createRandom, TEE } from "../src/pusht/sim.js";

const random = createRandom(Number(process.env.SEED ?? 77));

for (const count of [1, 2, 3, 4]) {
  const world = new PushWorld({ random, obstacleCount: count });
  const alongs = [];
  const laterals = [];
  let fallbacks = 0;
  let separations = 0;
  const trials = 400;

  for (let trial = 0; trial < trials; trial++) {
    world.reset({ obstacleCount: count });
    if (world.obstacles.length < count) fallbacks += 1;
    if (world.obstacles.length === 0) continue;
    const dx = world.goal.x - world.block.x;
    const dy = world.goal.y - world.block.y;
    const span = Math.hypot(dx, dy);
    separations += span;
    const ux = dx / span;
    const uy = dy / span;
    for (const obstacle of world.obstacles) {
      const ox = obstacle.x - world.block.x;
      const oy = obstacle.y - world.block.y;
      alongs.push((ox * ux + oy * uy) / span);
      laterals.push(Math.abs(-oy * ux + ox * uy));
    }
  }

  const kept = trials;
  const between = alongs.filter((a) => a > 0 && a < 1).length;
  // "Obstructing" = on the segment and close enough to the line that the block,
  // which is 0.2 across, cannot simply sail past it.
  const blocking = alongs.filter((a, i) => a > 0 && a < 1 && laterals[i] < TEE.radius).length;
  const mean = (xs) => xs.reduce((s, v) => s + v, 0) / xs.length;

  console.log(
    `obstacles=${count}  full layouts ${trials - fallbacks}/${trials}  short ${fallbacks}  obstacles placed ${alongs.length}/${trials * count}\n` +
    `   along:   mean ${mean(alongs).toFixed(2)}  between start & goal ${(100 * between / alongs.length).toFixed(0)}%\n` +
    `   lateral: mean ${mean(laterals).toFixed(3)}  within a block-radius of the line ${(100 * blocking / alongs.length).toFixed(0)}%\n` +
    `   mean start-goal separation ${(separations / kept).toFixed(3)}`,
  );
}
