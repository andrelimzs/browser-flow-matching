// Train a sparse-reward single-frame Push-T policy with PPO or SAC.
//
//   node tools/rl-train.mjs ppo 200000
//   node tools/rl-train.mjs sac 200000
//
// Optional environment variables: SEED, WIDTH, HORIZON, OUT.

import { writeFileSync } from "node:fs";
import { createRandom } from "../src/pusht/sim.js";
import { PushTRLEnv } from "../src/pusht/rl/env.js";
import { evaluatePolicy } from "../src/pusht/rl/common.js";
import { trainPPO } from "../src/pusht/rl/ppo.js";
import { trainSAC } from "../src/pusht/rl/sac.js";

const algorithm = (process.argv[2] ?? "ppo").toLowerCase();
const totalSteps = Number(process.argv[3] ?? 200_000);
const seed = Number(process.env.SEED ?? 2026);
const width = Number(process.env.WIDTH ?? 128);
const horizon = Number(process.env.HORIZON ?? 600);
if (!Number.isInteger(totalSteps) || totalSteps < 1) throw new Error(`invalid total steps: ${process.argv[3]}`);
if (algorithm !== "ppo" && algorithm !== "sac") throw new Error(`algorithm must be ppo or sac, got ${algorithm}`);

const random = createRandom(seed);
const env = new PushTRLEnv({ seed: seed + 1, horizon });
const started = process.hrtime.bigint();

const onProgress = (progress) => {
  const losses = progress.algorithm === "ppo"
    ? `policy ${progress.policyLoss.toFixed(4)} value ${progress.valueLoss.toFixed(4)}`
    : `actor ${progress.actorLoss.toFixed(4)} critics ${progress.criticLoss.toFixed(4)}`;
  const interval = progress.algorithm === "ppo" ? progress.rolloutSuccesses : progress.intervalSuccesses;
  console.log(
    `${progress.algorithm.toUpperCase()} ${progress.steps.toLocaleString()}/${totalSteps.toLocaleString()}` +
    `  completions ${progress.successes}/${progress.episodes} (+${interval})  ${losses}`,
  );
};

console.log(
  `${algorithm.toUpperCase()} · single frame (11) · action dx/dy (2) · reward distance progress, completion +1, wall -1\n` +
  `seed ${seed} · horizon ${horizon} · width ${width} · steps ${totalSteps.toLocaleString()}`,
);

const result = algorithm === "ppo"
  ? trainPPO({ env, random, totalSteps, width, onProgress })
  : trainSAC({ env, random, totalSteps, width, onProgress });

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
    action: "normalized dx,dy",
    reward: "block-goal distance progress + completion; wall contact -1 and terminal",
    seed,
    totalSteps,
    horizon,
    actor: result.actor.toJSON(),
    evaluation,
  }));
  console.log(`saved ${process.env.OUT}`);
}
