// Train a single-frame Push-T policy with PPO or GRPO.
//
//   node tools/rl-train.mjs ppo 100000
//   node tools/rl-train.mjs grpo 100000
//
// Optional environment variables: SEED, WIDTH, HORIZON, CURRICULUM, OUT.

import { writeFileSync } from "node:fs";
import { createRandom } from "../src/pusht/sim.js";
import { PushTRLEnv } from "../src/pusht/rl/env.js";
import { evaluatePolicy } from "../src/pusht/rl/common.js";
import { trainGRPO } from "../src/pusht/rl/grpo.js";
import { trainPPO } from "../src/pusht/rl/ppo.js";

const algorithm = (process.argv[2] ?? "ppo").toLowerCase();
const totalSteps = Number(process.argv[3] ?? 1_000_000);
const seed = Number(process.env.SEED ?? 2026);
const width = Number(process.env.WIDTH ?? 64);
const horizon = Number(process.env.HORIZON ?? 200);
const curriculum = process.env.CURRICULUM !== "false";
if (!Number.isInteger(totalSteps) || totalSteps < 1) throw new Error(`invalid total steps: ${process.argv[3]}`);
if (algorithm !== "ppo" && algorithm !== "grpo") throw new Error(`algorithm must be ppo or grpo, got ${algorithm}`);

const random = createRandom(seed);
const env = new PushTRLEnv({ seed: seed + 1, horizon, curriculum });
const started = process.hrtime.bigint();

const onProgress = (progress) => {
  const losses = progress.algorithm === "ppo"
    ? `policy ${progress.policyLoss.toFixed(4)} value ${progress.valueLoss.toFixed(4)}`
    : `policy ${progress.policyLoss.toFixed(4)} group return ${progress.groupReturnMean.toFixed(3)} ± ${progress.groupReturnStd.toFixed(3)}`;
  const interval = progress.algorithm === "ppo" ? progress.rolloutSuccesses : progress.groupSuccesses;
  console.log(
    `${progress.algorithm.toUpperCase()} ${progress.steps.toLocaleString()}/${totalSteps.toLocaleString()}` +
    `  completions ${progress.successes}/${progress.episodes} (+${interval})  ${losses}`,
  );
};

console.log(
  `${algorithm.toUpperCase()} · single frame (11) · unit-disk action dx/dy (2) · signed distance/orientation progress + final coverage and completion; -0.01/step; inactivity terminates at 5 still steps\n` +
  `seed ${seed} · horizon ${horizon} · width ${width} · curriculum ${curriculum ? "on" : "off"} · steps ${totalSteps.toLocaleString()}`,
);

const result = algorithm === "ppo"
  ? trainPPO({ env, random, totalSteps, width, onProgress })
  : trainGRPO({
      envs: Array.from({ length: 8 }, () => new PushTRLEnv({
        seed: seed + 1,
        horizon,
        curriculum,
      })),
      random,
      totalSteps,
      width,
      groupSize: 8,
      onProgress,
    });

env.setTrainingProgress(1);
const evaluation = evaluatePolicy(result.actor, env, 10);
const elapsed = Number(process.hrtime.bigint() - started) / 1e9;
console.log(
  `\ntraining completions ${result.successes}/${result.episodes}` +
  `\ndeterministic evaluation ${evaluation.successes}/${evaluation.episodes}, mean coverage ${evaluation.meanCoverage.toFixed(3)}` +
  `\nelapsed ${elapsed.toFixed(1)}s (${Math.round(totalSteps / elapsed).toLocaleString()} environment steps/s)`,
);

if (process.env.OUT) {
  writeFileSync(process.env.OUT, JSON.stringify({
    version: 1,
    algorithm,
    observation: "single normalized frame",
    action: "dx,dy projected onto the unit disk and scaled to max pusher speed",
    reward: "signed block-goal, pusher-goal, and orientation progress + final shape-overlap coverage + completion; -0.01 every step; -1 and termination after 5 stationary pusher steps; each wall penalty toggle also controls its termination",
    seed,
    totalSteps,
    horizon,
    curriculum,
    actor: result.actor.toJSON(),
    evaluation,
  }));
  console.log(`saved ${process.env.OUT}`);
}
