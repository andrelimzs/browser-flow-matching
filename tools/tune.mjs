import { PushWorld, createRandom } from "../src/pusht/sim.js";
import { ScriptedExpert, rollout } from "../src/pusht/expert.js";

function evaluate(options, episodes, obstacleCount, horizon = 1500) {
  const random = createRandom(11);
  const world = new PushWorld({ random, obstacleCount });
  const expert = new ScriptedExpert({ random, ...options });
  let successes = 0;
  let lengthSum = 0;
  for (let episode = 0; episode < episodes; episode++) {
    world.reset();
    const result = rollout(world, expert, { horizon });
    if (result.success) { successes += 1; lengthSum += result.steps; }
  }
  return { rate: successes / episodes, length: successes ? lengthSum / successes : NaN };
}

const grid = [];
for (const travelWeight of [0.35, 0.5, 0.7, 1.0]) {
  for (const rotationWeight of [1.5, 2, 2.5]) grid.push({ travelWeight, rotationWeight });
}
const scored = grid.map((options) => {
  const a = evaluate(options, 100, 1);
  const b = evaluate(options, 100, 2);
  const c = evaluate(options, 100, 4);
  const rate = (a.rate + b.rate + c.rate) / 3;
  return { options, rate, a, b, c };
});
scored.sort((x, y) => y.rate - x.rate);
for (const row of scored) {
  console.log(
    `travel ${String(row.options.travelWeight).padStart(4)} rot ${String(row.options.rotationWeight).padStart(4)}` +
    `  mean ${(row.rate * 100).toFixed(1)}%  [o1 ${(row.a.rate * 100).toFixed(0)}% o2 ${(row.b.rate * 100).toFixed(0)}% o4 ${(row.c.rate * 100).toFixed(0)}%]` +
    `  len ${Number.isNaN(row.b.length) ? "-" : row.b.length.toFixed(0)}`,
  );
}
