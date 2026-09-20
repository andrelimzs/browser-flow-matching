// Measures scripted-expert success rate and throughput.
// Usage: node tools/expert-bench.mjs [episodes] [obstacleCount]
import { PushWorld, createRandom, SUCCESS_COVERAGE } from "../src/pusht/sim.js";
import { ScriptedExpert, rollout } from "../src/pusht/expert.js";

const episodes = Number(process.argv[2] ?? 100);
const obstacleCount = Number(process.argv[3] ?? 1);
const horizon = Number(process.argv[4] ?? 2200);

const random = createRandom(Number(process.env.SEED ?? 11));
const world = new PushWorld({ random, obstacleCount });
const expert = new ScriptedExpert({ random });

let successes = 0;
let totalSteps = 0;
let coverageSum = 0;
let bestSum = 0;
const lengths = [];
const start = process.hrtime.bigint();

for (let episode = 0; episode < episodes; episode++) {
  world.reset();
  const result = rollout(world, expert, { horizon });
  if (result.success) { successes += 1; lengths.push(result.steps); }
  totalSteps += result.steps;
  coverageSum += result.coverage;
  bestSum += result.best;
}

const elapsed = Number(process.hrtime.bigint() - start) / 1e9;
lengths.sort((a, b) => a - b);
console.log(`obstacles=${obstacleCount} horizon=${horizon} episodes=${episodes}`);
console.log(`success ${successes}/${episodes} = ${(100 * successes / episodes).toFixed(1)}%  (threshold coverage ${SUCCESS_COVERAGE})`);
console.log(`mean final coverage ${(coverageSum / episodes).toFixed(3)}  mean best coverage ${(bestSum / episodes).toFixed(3)}`);
if (lengths.length) console.log(`success length median ${lengths[lengths.length >> 1]} p90 ${lengths[Math.floor(lengths.length * 0.9)]}`);
console.log(`${totalSteps} steps in ${elapsed.toFixed(2)}s = ${Math.round(totalSteps / elapsed)} steps/s`);
