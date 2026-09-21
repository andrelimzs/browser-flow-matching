// Train a single-frame Push-T policy with PPO or Dr.GRPO.
//
//   node tools/rl-train.mjs ppo 100000
//   node tools/rl-train.mjs drgrpo 100000
//
// Optional environment variables: SEED, WIDTH, HORIZON, CURRICULUM, OUT,
// CHECKPOINT_EVERY. Intermediate checkpoints are written beside OUT.

import { writeFileSync } from "node:fs";
import { createRandom } from "../src/pusht/sim.js";
import { PushTRLEnv } from "../src/pusht/rl/env.js";
import { sampleActor } from "../src/pusht/rl/common.js";
import { DEFAULT_DRGRPO_GROUP_SIZE, trainDrGRPO } from "../src/pusht/rl/grpo.js";
import { trainPPO } from "../src/pusht/rl/ppo.js";

const algorithm = (process.argv[2] ?? "ppo").toLowerCase();
const totalSteps = Number(process.argv[3] ?? 1_000_000);
const seed = Number(process.env.SEED ?? 2026);
const width = Number(process.env.WIDTH ?? 64);
const horizon = Number(process.env.HORIZON ?? 200);
const curriculum = process.env.CURRICULUM !== "false";
const checkpointEvery = Number(process.env.CHECKPOINT_EVERY ?? 0);
if (!Number.isInteger(totalSteps) || totalSteps < 1) throw new Error(`invalid total steps: ${process.argv[3]}`);
if (algorithm !== "ppo" && algorithm !== "drgrpo") {
  throw new Error(`algorithm must be ppo or drgrpo, got ${algorithm}`);
}

const random = createRandom(seed);
const env = new PushTRLEnv({ seed: seed + 1, horizon, curriculum });
const started = process.hrtime.bigint();
const algorithmLabel = algorithm === "drgrpo" ? "Dr.GRPO" : "PPO";
let nextSavedCheckpoint = checkpointEvery > 0 ? checkpointEvery : Infinity;

function evaluateFixed(actor, { episodes, deterministic, evaluationSeed }) {
  const evaluationEnv = new PushTRLEnv({ seed: evaluationSeed, horizon, curriculum: false });
  const evaluationRandom = createRandom(evaluationSeed + 1);
  const action = new Float32Array(2);
  let successes = 0;
  let coverage = 0;
  let returns = 0;
  let steps = 0;
  for (let episode = 0; episode < episodes; episode++) {
    let observation = evaluationEnv.reset();
    for (;;) {
      actor.inputBuffer().set(observation, 0);
      const sample = sampleActor(actor, actor.forward(1), 0, evaluationRandom, deterministic);
      action[0] = sample.actionX;
      action[1] = sample.actionY;
      const transition = evaluationEnv.step(action);
      observation = transition.observation;
      if (transition.done) {
        successes += transition.success ? 1 : 0;
        coverage += transition.coverage;
        returns += evaluationEnv.episodeReturn;
        steps += evaluationEnv.episodeSteps;
        break;
      }
    }
  }
  return {
    successes,
    episodes,
    meanCoverage: coverage / episodes,
    meanReturn: returns / episodes,
    meanSteps: steps / episodes,
  };
}

function outputAtStep(output, step) {
  return output.replace(/\.json$/i, `.${step}.json`);
}

function checkpointPayload(actor, step, evaluation) {
  return {
    version: 1,
    algorithm,
    observation: "single normalized frame",
    action: "dx,dy projected onto the unit disk and scaled to max pusher speed",
    reward: "signed block-goal, pusher-goal, and orientation progress + final shape-overlap coverage + completion; -0.01 every step; -0.01 times squared unit-action acceleration; -1 and termination after 5 stationary pusher steps; each wall penalty toggle also controls its termination",
    seed,
    totalSteps: step,
    horizon,
    curriculum,
    actor: actor.toJSON(),
    evaluation,
  };
}

const onProgress = (progress, models) => {
  const losses = progress.algorithm === "ppo"
    ? `policy ${progress.policyLoss.toFixed(4)} value ${progress.valueLoss.toFixed(4)}`
    : `policy ${progress.policyLoss.toFixed(4)} group return ${progress.groupReturnMean.toFixed(3)} ± ${progress.groupReturnStd.toFixed(3)}`;
  const interval = progress.algorithm === "ppo" ? progress.rolloutSuccesses : progress.groupSuccesses;
  console.log(
    `${algorithmLabel} ${progress.steps.toLocaleString()}/${totalSteps.toLocaleString()}` +
    `  completions ${progress.successes}/${progress.episodes} (+${interval})  ${losses}`,
  );
  if (process.env.OUT && progress.steps >= nextSavedCheckpoint) {
    const deterministic = evaluateFixed(models.actor, { episodes: 10, deterministic: true, evaluationSeed: seed + 10_000 });
    const stochastic = evaluateFixed(models.actor, { episodes: 100, deterministic: false, evaluationSeed: seed + 20_000 });
    const evaluation = { deterministic, stochastic };
    const path = outputAtStep(process.env.OUT, progress.steps);
    writeFileSync(path, JSON.stringify(checkpointPayload(models.actor, progress.steps, evaluation)));
    console.log(
      `checkpoint ${progress.steps.toLocaleString()}: deterministic ${deterministic.successes}/${deterministic.episodes}` +
      ` coverage ${deterministic.meanCoverage.toFixed(3)} return ${deterministic.meanReturn.toFixed(3)} steps ${deterministic.meanSteps.toFixed(1)};` +
      ` stochastic ${stochastic.successes}/${stochastic.episodes} coverage ${stochastic.meanCoverage.toFixed(3)}`,
    );
    nextSavedCheckpoint += checkpointEvery;
  }
};

console.log(
  `${algorithmLabel} · single frame (11) · unit-disk action dx/dy (2) · signed distance/orientation progress + final coverage and completion; -0.01/step; -0.01(ddx²+ddy²); inactivity terminates at 5 still steps\n` +
  `seed ${seed} · horizon ${horizon} · width ${width} · curriculum ${curriculum ? "on" : "off"} · steps ${totalSteps.toLocaleString()}`,
);

const result = await (algorithm === "ppo"
  ? trainPPO({ env, random, totalSteps, width, onProgress })
  : trainDrGRPO({
      envs: Array.from({ length: DEFAULT_DRGRPO_GROUP_SIZE }, () => new PushTRLEnv({
        seed: seed + 1,
        horizon,
        curriculum,
      })),
      random,
      totalSteps,
      width,
      groupSize: DEFAULT_DRGRPO_GROUP_SIZE,
      onProgress,
    }));

env.setTrainingProgress(1);
const evaluation = {
  deterministic: evaluateFixed(result.actor, { episodes: 10, deterministic: true, evaluationSeed: seed + 10_000 }),
  stochastic: evaluateFixed(result.actor, { episodes: 100, deterministic: false, evaluationSeed: seed + 20_000 }),
};
const elapsed = Number(process.hrtime.bigint() - started) / 1e9;
console.log(
  `\ntraining completions ${result.successes}/${result.episodes}` +
  `\ndeterministic evaluation ${evaluation.deterministic.successes}/${evaluation.deterministic.episodes}, mean coverage ${evaluation.deterministic.meanCoverage.toFixed(3)}` +
  `\nstochastic evaluation ${evaluation.stochastic.successes}/${evaluation.stochastic.episodes}, mean coverage ${evaluation.stochastic.meanCoverage.toFixed(3)}` +
  `\nelapsed ${elapsed.toFixed(1)}s (${Math.round(totalSteps / elapsed).toLocaleString()} environment steps/s)`,
);

if (process.env.OUT) {
  writeFileSync(process.env.OUT, JSON.stringify(checkpointPayload(result.actor, totalSteps, evaluation)));
  console.log(`saved ${process.env.OUT}`);
}
