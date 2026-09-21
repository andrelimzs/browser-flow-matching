import { createRandom } from "./sim.js";
import { PushTRLEnv } from "./rl/env.js";
import { recordPolicyRollout, rolloutBlockTrack, rolloutPusherPath } from "./rl/common.js";
import { DEFAULT_DRGRPO_GROUP_SIZE, trainDrGRPO } from "./rl/grpo.js";
import { trainPPO } from "./rl/ppo.js";

const EVALUATION_ROLLOUTS = 10;
let stochasticEvaluation = false;
let training = false;

const yieldToMessages = () => self.scheduler?.yield
  ? self.scheduler.yield()
  : new Promise((resolve) => setTimeout(resolve, 0));

self.onmessage = async ({ data }) => {
  if (data.type === "set-evaluation-mode") {
    stochasticEvaluation = Boolean(data.stochastic);
    return;
  }
  if (data.type !== "start") return;
  if (training) return;
  training = true;

  const {
    algorithm,
    totalSteps,
    width,
    horizon,
    entropyBonus,
    curriculum,
    rewardShaping,
    rewardWeights,
    seed,
    stochasticEval,
  } = data;
  stochasticEvaluation = Boolean(stochasticEval);
  const env = new PushTRLEnv({ seed, horizon, curriculum, rewardShaping, rewardWeights });
  const random = createRandom(seed + 20_000);
  let lastLoggedGroup = null;

  const logRollout = (progress, models) => {
    const estimateValue = progress.algorithm === "ppo"
      ? (observation) => {
          models.critic.inputBuffer().set(observation, 0);
          return models.critic.forward(1)[0];
        }
      : () => 0;
    const evaluationRollouts = [];
    let evaluationReturn = 0;
    for (let episode = 0; episode < EVALUATION_ROLLOUTS; episode++) {
      const evaluationEnv = new PushTRLEnv({
        seed: seed + 10_000 + episode,
        horizon,
        curriculum,
        rewardShaping,
        rewardWeights,
      });
      evaluationEnv.setTrainingProgress(progress.steps / totalSteps);
      const rollout = recordPolicyRollout(models.actor, evaluationEnv, {
        random: createRandom(seed + 30_000 + episode),
        deterministic: !stochasticEvaluation,
        estimateValue,
      });
      evaluationRollouts.push(rollout);
      evaluationReturn += rollout.return;
    }
    evaluationReturn /= EVALUATION_ROLLOUTS;
    const rollout = evaluationRollouts.reduce((closest, candidate) =>
      Math.abs(candidate.return - evaluationReturn) < Math.abs(closest.return - evaluationReturn)
        ? candidate
        : closest);
    const evaluationPeers = evaluationRollouts.filter((candidate) => candidate !== rollout);
    const evaluationPaths = evaluationPeers.map((candidate) =>
      rolloutPusherPath(candidate.frames, candidate.count));
    const evaluationBlockTracks = evaluationPeers.map((candidate) =>
      rolloutBlockTrack(candidate.frames, candidate.count));
    const hasNewGroup = models.groupPaths && models.groupPaths !== lastLoggedGroup;
    const groupPaths = hasNewGroup
      ? models.groupPaths.map((points) => Float32Array.from(points))
      : [];
    const groupAdvantages = hasNewGroup
      ? Float32Array.from(models.groupAdvantages ?? [])
      : new Float32Array(0);
    if (hasNewGroup) lastLoggedGroup = models.groupPaths;
    self.postMessage({
      type: "rollout",
      progress,
      step: progress.steps,
      stochasticEvaluation,
      evaluationRollouts: EVALUATION_ROLLOUTS,
      representativeReturn: rollout.return,
      trainReturn: progress.trainReturn ?? progress.groupReturnMean ?? null,
      hasValueEstimate: progress.algorithm === "ppo",
      evaluationBlockTracks,
      evaluationPaths,
      groupPaths,
      groupAdvantages,
      ...rollout,
      return: evaluationReturn,
    }, [
      rollout.frames.buffer,
      groupAdvantages.buffer,
      ...evaluationPaths.map((path) => path.buffer),
      ...evaluationBlockTracks.map((track) => track.buffer),
      ...groupPaths.map((path) => path.buffer),
    ]);
  };

  self.postMessage({ type: "started", algorithm });
  try {
    if (algorithm === "drgrpo") {
      const envs = Array.from({ length: DEFAULT_DRGRPO_GROUP_SIZE }, () => new PushTRLEnv({
        seed,
        horizon,
        curriculum,
        rewardShaping,
        rewardWeights,
      }));
      await trainDrGRPO({
        envs,
        totalSteps,
        width,
        entropyCoefficient: entropyBonus,
        groupSize: DEFAULT_DRGRPO_GROUP_SIZE,
        batchSize: 64,
        progressEvery: 200,
        random,
        onCheckpoint: logRollout,
        yieldControl: yieldToMessages,
      });
    } else {
      await trainPPO({
        env,
        totalSteps,
        width,
        entropyCoefficient: entropyBonus,
        rolloutSteps: 2048,
        batchSize: 64,
        progressEvery: 200,
        random,
        onCheckpoint: logRollout,
        yieldControl: yieldToMessages,
      });
    }
    self.postMessage({ type: "done" });
  } catch (error) {
    self.postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) });
  } finally {
    training = false;
  }
};
