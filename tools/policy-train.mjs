// Collects scripted demonstrations, trains the flow-matching policy on them,
// and evaluates it closed-loop in the simulator.
// Usage: node tools/policy-train.mjs [episodes] [trainSteps] [evalEpisodes]
import { PushWorld, createRandom, SUCCESS_COVERAGE } from "../src/pusht/sim.js";
import { ScriptedExpert } from "../src/pusht/expert.js";
import { makePolicy, buildDataset, PolicyTrainer, sampleChunk, CHUNK } from "../src/pusht/policy.js";

const DEMOS = Number(process.argv[2] ?? 60);
const TRAIN_STEPS = Number(process.argv[3] ?? 3000);
const EVAL = Number(process.argv[4] ?? 40);
const EXECUTE = Number(process.env.EXECUTE ?? 16);
const WIDTH = Number(process.env.WIDTH ?? 128);
const CAP = 2200;
const OBSTACLES = Number(process.env.OBSTACLES ?? 1);

const random = createRandom(2718);
const world = new PushWorld({ random, obstacleCount: OBSTACLES });
const expert = new ScriptedExpert({ random });

// ---- collect ----
let collected = 0, expertWins = 0;
const episodes = [];
const collectStart = process.hrtime.bigint();
for (let episode = 0; episode < DEMOS; episode++) {
  world.reset({ obstacleCount: OBSTACLES });
  expert.reset();
  const observations = [], actions = [];
  for (let step = 0; step < CAP; step++) {
    if (world.coverage() >= SUCCESS_COVERAGE) break;
    observations.push(Array.from(world.writeObservation()));
    const [x, y, lift] = expert.act(world);
    actions.push([x, y, lift]);
    world.step(x, y, lift);
  }
  const success = world.coverage() >= SUCCESS_COVERAGE;
  if (success) expertWins += 1;
  episodes.push({ observations, actions, success, observationSize: observations[0].length });
  collected += observations.length;
}
const collectSeconds = Number(process.hrtime.bigint() - collectStart) / 1e9;
console.log(`collected ${DEMOS} episodes (${collected.toLocaleString()} transitions) in ${collectSeconds.toFixed(1)}s, expert success ${(100*expertWins/DEMOS).toFixed(0)}%`);

// ---- train ----
const dataset = buildDataset(episodes);
console.log(`training on ${dataset.episodes}/${episodes.length} solved episodes, ${dataset.count.toLocaleString()} of ${collected.toLocaleString()} transitions (failures dropped)`);
const policy = makePolicy({ observationSize: dataset.observationSize, width: WIDTH, maxBatch: 256, random });
console.log(`observation: ${dataset.observationSize} egocentric dims (from ${episodes[0].observationSize} absolute)`);
const trainer = new PolicyTrainer({ policy, batch: 256, learningRate: 0.002, random });
console.log(`policy ${policy.model.sizes.join("->")}, ${policy.model.params.reduce((s,a)=>s+a.length,0).toLocaleString()} params`);

const trainStart = process.hrtime.bigint();
let loss = 0;
for (let step = 0; step < TRAIN_STEPS; step++) {
  loss = trainer.step(dataset);
  if (step % Math.max(1, Math.floor(TRAIN_STEPS / 6)) === 0) {
    console.log(`  step ${String(step).padStart(5)}  loss ${loss.toFixed(4)}`);
  }
}
const trainSeconds = Number(process.hrtime.bigint() - trainStart) / 1e9;
console.log(`trained ${TRAIN_STEPS} steps in ${trainSeconds.toFixed(1)}s (${(trainSeconds/TRAIN_STEPS*1000).toFixed(1)} ms/step), final loss ${loss.toFixed(4)}`);

// ---- evaluate closed loop ----
let wins = 0, coverageSum = 0;
const evalStart = process.hrtime.bigint();
for (let episode = 0; episode < EVAL; episode++) {
  world.reset({ obstacleCount: OBSTACLES });
  let chunk = null, cursor = CHUNK;
  for (let step = 0; step < CAP; step++) {
    if (world.coverage() >= SUCCESS_COVERAGE) break;
    if (cursor >= EXECUTE || !chunk) {
      chunk = sampleChunk(policy, world.writeObservation(), { steps: 10, random, scales: dataset.scales });
      cursor = 0;
    }
    world.step(chunk[cursor * 2], chunk[cursor * 2 + 1]);
    cursor += 1;
  }
  const coverage = world.coverage();
  coverageSum += coverage;
  if (coverage >= SUCCESS_COVERAGE) wins += 1;
}
const evalSeconds = Number(process.hrtime.bigint() - evalStart) / 1e9;
console.log(`\npolicy closed-loop: ${wins}/${EVAL} = ${(100*wins/EVAL).toFixed(1)}%  mean coverage ${(coverageSum/EVAL).toFixed(3)}  (${evalSeconds.toFixed(1)}s)`);
console.log(`expert on the same generator: ${(100*expertWins/DEMOS).toFixed(0)}%`);
