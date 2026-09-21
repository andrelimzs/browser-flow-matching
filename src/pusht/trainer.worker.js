// Trains the flow-matching policy off the main thread.
//
// Training is far too slow to run inside a frame — tens of milliseconds per
// optimizer step — so it lives here and reports progress back. The page stays
// interactive throughout, and the finished weights come back as plain arrays.

import { buildDataset, makePolicy, PolicyTrainer } from "./policy.js";

self.onmessage = (event) => {
  const { episodes, steps, width, learningRate, stateNoise = 0, actionNoise = 0 } = event.data;

  const dataset = buildDataset(episodes);
  if (!dataset || dataset.count < 64) {
    self.postMessage({ type: "error", message: "Not enough solved demonstrations to train on." });
    return;
  }

  const policy = makePolicy({
    observationSize: dataset.observationSize,
    width,
    maxBatch: 256,
    random: Math.random,
  });
  const trainer = new PolicyTrainer({ policy, batch: 256, learningRate, stateNoise, actionNoise });

  // Everything the page needs to build a matching policy, before any weights
  // arrive, so it can adopt the first snapshot immediately.
  self.postMessage({
    type: "started",
    transitions: dataset.count,
    episodes: dataset.episodes,
    observationSize: dataset.observationSize,
    sizes: policy.model.sizes,
    scales: Array.from(dataset.scales),
    stateNoise,
    actionNoise,
    params: policy.model.params.reduce((total, buffer) => total + buffer.length, 0),
  });

  const report = Math.max(1, Math.floor(steps / 120));
  // Weights go out often enough to watch the policy improve, rarely enough that
  // the copy is free: 0.019 ms against a ~20 ms step.
  const snapshotEvery = Math.max(report, Math.floor(steps / 25));
  const started = performance.now();
  let loss = 0;
  for (let step = 0; step < steps; step++) {
    loss = trainer.step(dataset);
    if (step % report === 0) {
      self.postMessage({ type: "progress", step, steps, loss, elapsed: performance.now() - started });
    }
    if (step % snapshotEvery === 0) {
      const weights = policy.model.snapshot();
      self.postMessage({ type: "weights", step, weights }, [weights.buffer]);
    }
  }

  const weights = policy.model.snapshot();
  self.postMessage({
    type: "done",
    loss,
    elapsed: performance.now() - started,
    weights,
    scales: Array.from(dataset.scales),
    observationSize: dataset.observationSize,
  }, [weights.buffer]);
};
