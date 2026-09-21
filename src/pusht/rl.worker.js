import { createRandom } from "./sim.js";
import { PushTRLEnv } from "./rl/env.js";
import { recordPolicyRollout } from "./rl/common.js";
import { trainGRPO } from "./rl/grpo.js";
import { trainPPO } from "./rl/ppo.js";

self.onmessage = ({ data }) => {
  if (data.type !== "start") return;

  const { algorithm, totalSteps, width, horizon, entropyBonus, curriculum, rewardShaping, rewardWeights, seed } = data;
  const env = new PushTRLEnv({ seed, horizon, curriculum, rewardShaping, rewardWeights });
  const random = createRandom(seed + 20_000);

  const logRollout = (progress, models) => {
    const evaluationEnv = new PushTRLEnv({
      seed: seed + 10_000,
      horizon,
      curriculum,
      rewardShaping,
      rewardWeights,
    });
    evaluationEnv.setTrainingProgress(progress.steps / totalSteps);
    const estimateValue = progress.algorithm === "ppo"
      ? (observation) => {
          models.critic.inputBuffer().set(observation, 0);
          return models.critic.forward(1)[0];
        }
      : () => 0;
    const rollout = recordPolicyRollout(models.actor, evaluationEnv, {
      random: createRandom(seed + 30_000),
      deterministic: true,
      estimateValue,
    });
    const groupPaths = (models.groupPaths ?? []).map((points) => Float32Array.from(points));
    const groupAdvantages = Float32Array.from(models.groupAdvantages ?? []);
    self.postMessage({
      type: "rollout",
      progress,
      step: progress.steps,
      hasValueEstimate: progress.algorithm === "ppo",
      groupPaths,
      groupAdvantages,
      ...rollout,
    }, [rollout.frames.buffer, groupAdvantages.buffer, ...groupPaths.map((path) => path.buffer)]);
  };

  self.postMessage({ type: "started", algorithm });
  try {
    if (algorithm === "grpo") {
      const envs = Array.from({ length: 8 }, () => new PushTRLEnv({
        seed,
        horizon,
        curriculum,
        rewardShaping,
        rewardWeights,
      }));
      trainGRPO({
        envs,
        totalSteps,
        width,
        entropyCoefficient: entropyBonus,
        groupSize: 8,
        batchSize: 64,
        progressEvery: 200,
        random,
        onCheckpoint: logRollout,
      });
    } else {
      trainPPO({
        env,
        totalSteps,
        width,
        entropyCoefficient: entropyBonus,
        rolloutSteps: 2048,
        batchSize: 64,
        progressEvery: 200,
        random,
        onCheckpoint: logRollout,
      });
    }
    self.postMessage({ type: "done" });
  } catch (error) {
    self.postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
