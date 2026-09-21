import { createRandom } from "./sim.js";
import { PushTRLEnv } from "./rl/env.js";
import { recordPolicyRollout } from "./rl/common.js";
import { trainPPO } from "./rl/ppo.js";
import { trainSAC } from "./rl/sac.js";

self.onmessage = ({ data }) => {
  if (data.type !== "start") return;

  const { algorithm, totalSteps, width, horizon, seed } = data;
  const env = new PushTRLEnv({ seed, horizon });
  const random = createRandom(seed + 20_000);

  const onProgress = (progress, models) => {
    const evaluationEnv = new PushTRLEnv({ seed: seed + 10_000, horizon });
    const rollout = recordPolicyRollout(models.actor, evaluationEnv);
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
        progressEvery: 200,
        random,
        onProgress,
      });
    } else {
      trainPPO({
        env,
        totalSteps,
        width,
        rolloutSteps: 200,
        random,
        onProgress,
      });
    }
    self.postMessage({ type: "done" });
  } catch (error) {
    self.postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
