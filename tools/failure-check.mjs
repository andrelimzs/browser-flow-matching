// Where do failures actually come from? Splits them by whether the block got
// near the goal position at all, and whether it was orientation that was left.
import { PushWorld, createRandom, SUCCESS_COVERAGE, TEE } from "../src/pusht/sim.js";
import { ScriptedExpert } from "../src/pusht/expert.js";
import { planPath } from "../src/pusht/plan.js";

const obstacleCount = Number(process.argv[2] ?? 2);
const random = createRandom(101);
const world = new PushWorld({ random, obstacleCount });
const expert = new ScriptedExpert({ random });

let noRoute = 0, solved = 0;
const buckets = { stranded: 0, positionOnly: 0, angleOnly: 0, both: 0 };
const episodes = 150;

for (let episode = 0; episode < episodes; episode++) {
  world.reset({ obstacleCount });
  expert.reset();
  if (!planPath([world.block.x, world.block.y], [world.goal.x, world.goal.y], world.obstacles, TEE.radius * 0.78)) noRoute += 1;

  for (let step = 0; step < 900; step++) {
    if (world.coverage() >= SUCCESS_COVERAGE) break;
    const [x, y, lift] = expert.act(world);
    world.step(x, y, lift);
  }

  if (world.coverage() >= SUCCESS_COVERAGE) { solved += 1; continue; }
  const error = world.poseError();
  const nearGoal = error.position < 0.06;
  const aligned = error.angle < 0.35;
  if (error.position > 0.2) buckets.stranded += 1;
  else if (nearGoal && !aligned) buckets.angleOnly += 1;
  else if (!nearGoal && aligned) buckets.positionOnly += 1;
  else buckets.both += 1;
}

const failures = episodes - solved;
console.log(`obstacles=${obstacleCount}  solved ${solved}/${episodes}  unplannable layouts ${noRoute}`);
console.log(`failures (${failures}):`);
console.log(`  stranded far from goal (>0.20)   ${buckets.stranded}`);
console.log(`  near goal, orientation left      ${buckets.angleOnly}`);
console.log(`  aligned, position left           ${buckets.positionOnly}`);
console.log(`  both still off                   ${buckets.both}`);
