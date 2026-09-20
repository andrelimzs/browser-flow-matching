// Flow-matching policy on the simplest setting: no obstacles, lifting enabled.
// Evaluates from the training start poses first — if it cannot reproduce those,
// generalization numbers mean nothing.
// Usage: node tools/policy-flow.mjs [episodes] [steps] [width]
import { PushWorld, createRandom, SUCCESS_COVERAGE } from "../src/pusht/sim.js";
import { ScriptedExpert } from "../src/pusht/expert.js";
import { buildDataset, makePolicy, PolicyTrainer, sampleChunk, CHUNK, ACTION_DIM } from "../src/pusht/policy.js";

const EPISODES = Number(process.argv[2] ?? 150);
const STEPS = Number(process.argv[3] ?? 12000);
const WIDTH = Number(process.argv[4] ?? 256);
const OBSTACLES = Number(process.env.OBSTACLES ?? 0);
const EXECUTE = Number(process.env.EXECUTE ?? 16);
const EULER = Number(process.env.EULER ?? 10);
const CAP = 900;
const LIFT_ON = Number(process.env.LIFT_ON ?? 0.4);
const LIFT_OFF = Number(process.env.LIFT_OFF ?? -0.4);

const random = createRandom(1717);
const world = new PushWorld({ random, obstacleCount: OBSTACLES });
const expert = new ScriptedExpert({ random });

function collect(n) {
  const eps = [];
  for (let e = 0; e < n; e++) {
    world.reset({ obstacleCount: OBSTACLES });
    const initial = world.snapshot();
    expert.reset();
    const observations = [], actions = [];
    for (let s = 0; s < CAP; s++) {
      if (world.coverage() >= SUCCESS_COVERAGE) break;
      observations.push(Array.from(world.writeObservation()));
      const [x, y, lift] = expert.act(world);
      actions.push([x, y, lift]);
      world.step(x, y, lift);
    }
    eps.push({ initial, observations, actions, success: world.coverage() >= SUCCESS_COVERAGE,
               observationSize: observations[0].length });
  }
  return eps;
}

const train = collect(EPISODES).filter((e) => e.success);
const dataset = buildDataset(train);
console.log(`${train.length} solved episodes, ${dataset.count.toLocaleString()} transitions, obs ${dataset.observationSize} egocentric dims`);

const policy = makePolicy({ observationSize: dataset.observationSize, width: WIDTH, maxBatch: 256, random });
const trainer = new PolicyTrainer({ policy, batch: 256, learningRate: 0.002, random });
console.log(`policy ${policy.model.sizes.join("->")}, ${policy.model.params.reduce((s, a) => s + a.length, 0).toLocaleString()} params`);

const started = process.hrtime.bigint();
let loss = 0;
for (let step = 0; step < STEPS; step++) {
  loss = trainer.step(dataset);
  if (step % Math.max(1, Math.floor(STEPS / 8)) === 0) console.log(`  step ${String(step).padStart(6)}  loss ${loss.toFixed(3)}`);
}
console.log(`trained ${STEPS} steps in ${(Number(process.hrtime.bigint() - started) / 1e9).toFixed(0)}s, loss ${loss.toFixed(3)}`);

function evaluate(episodes, label) {
  let wins = 0, coverage = 0;
  for (const ep of episodes) {
    world.restore(ep.initial);
    let chunk = null, cursor = 1e9, lift = 0;
    for (let s = 0; s < CAP; s++) {
      if (world.coverage() >= SUCCESS_COVERAGE) break;
      if (cursor >= EXECUTE) {
        chunk = sampleChunk(policy, world.writeObservation(), { steps: EULER, random, scales: dataset.scales });
        cursor = 0;
      }
      // Hysteresis on lift: the raw per-step sign is wrong ~16% of the time,
      // and an isolated flip is catastrophic (the pusher passes through the
      // block instead of pushing it). Switching only on a confident value and
      // holding otherwise turns those isolated errors into no-ops.
      const base = cursor * ACTION_DIM;
      const raw = chunk[base + 2];
      if (raw > LIFT_ON) lift = 1;
      else if (raw < LIFT_OFF) lift = 0;
      world.step(chunk[base], chunk[base + 1], lift);
      cursor += 1;
    }
    coverage += world.coverage();
    if (world.coverage() >= SUCCESS_COVERAGE) wins += 1;
  }
  console.log(`${label.padEnd(36)} ${wins}/${episodes.length} = ${(100 * wins / episodes.length).toFixed(0)}%  mean coverage ${(coverage / episodes.length).toFixed(3)}`);
}

console.log("");
evaluate(train.slice(0, 40), "flow, training start poses");
evaluate(collect(40).filter((e) => e.success), "flow, fresh start poses");
