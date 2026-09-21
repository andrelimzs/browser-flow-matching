import { createRandom } from "./sim.js";
import { PushTRLEnv } from "./rl/env.js";
import { recordPolicyRollout } from "./rl/common.js";
import { trainPPO } from "./rl/ppo.js";
import { trainSAC } from "./rl/sac.js";

self.onmessage = ({ data }) => {
  if (data.type !== "start") return;

  const { algorithm, totalSteps, width, horizon, entropyBonus, curriculum, rewardShaping, seed } = data;
  const env = new PushTRLEnv({ seed, horizon, curriculum, rewardShaping });
  const random = createRandom(seed + 20_000);

  const logRollout = (progress, models) => {
    const evaluationEnv = new PushTRLEnv({ seed: seed + 10_000, horizon, curriculum, rewardShaping });
    evaluationEnv.setTrainingProgress(progress.steps / totalSteps);
    const estimateValue = progress.algorithm === "ppo"
      ? (observation) => {
          models.critic.inputBuffer().set(observation, 0);
          return models.critic.forward(1)[0];
        }
      : (observation, sample) => {
          const input1 = models.critic1.inputBuffer();
          const input2 = models.critic2.inputBuffer();
          input1.set(observation, 0);
          input2.set(observation, 0);
          input1[observation.length] = sample.actionX;
          input1[observation.length + 1] = sample.actionY;
          input2[observation.length] = sample.actionX;
          input2[observation.length + 1] = sample.actionY;
          const value1 = models.critic1.forward(1)[0];
          const value2 = models.critic2.forward(1)[0];
          return Math.min(value1, value2);
        };
    const rollout = recordPolicyRollout(models.actor, evaluationEnv, {
      random: createRandom(seed + 30_000),
      deterministic: true,
      estimateValue,
    });
    self.postMessage({
      type: "rollout",
      progress,
      step: progress.steps,
      ...rollout,
    }, [rollout.frames.buffer]);
  };

  self.postMessage({ type: "started", algorithm });
  try {
    if (algorithm === "sac") {
      trainSAC({
        env,
        totalSteps,
        width,
        alpha: entropyBonus,
        batchSize: 256,
        progressEvery: 200,
        random,
        onProgress: logRollout,
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
