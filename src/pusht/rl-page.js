import "./style.css";
import "./rl-style.css";
import { MAX_PUSHER_SPEED, PushWorld, createRandom } from "./sim.js";
import { createView } from "./render.js";
import { rolloutPusherPath, simpleMovingAverage } from "./rl/common.js";

const $ = (selector) => document.querySelector(selector);
const ROLLOUT_FRAME_SIZE = 12;
const BUDGET_MIN = 10_000;
const BUDGET_MAX = 1_000_000;
const RETURN_SMA_WINDOW = 10;
const REWARD_COMPONENTS = [
  "completion",
  "blockDistance",
  "pusherDistance",
  "orientation",
  "closeness",
  "pusherWall",
  "blockWall",
  "inactivity",
  "stepPenalty",
];
const REWARD_WEIGHT_DEFAULTS = {
  completion: 10,
  blockDistance: 1,
  pusherDistance: 0.01 / MAX_PUSHER_SPEED,
  orientation: 1,
  closeness: 5,
  pusherWall: -1,
  blockWall: -10,
  inactivity: -1,
  stepPenalty: -0.01,
};
const entropyConfigs = {
  ppo: { label: "Entropy bonus", min: 0, max: 0.1, step: 0.005, digits: 3 },
  drgrpo: { label: "Entropy bonus", min: 0, max: 0.1, step: 0.005, digits: 3 },
};
const world = new PushWorld({ random: createRandom(41), obstacleCount: 0 });
const view = createView($("#arena"));
const state = {
  algorithm: "ppo",
  budget: 1_000_000,
  horizon: 200,
  curriculum: true,
  stochasticEval: false,
  rewardShaping: {
    completion: true,
    blockDistance: true,
    pusherDistance: true,
    orientation: true,
    closeness: true,
    pusherWall: false,
    blockWall: false,
    inactivity: true,
    stepPenalty: true,
  },
  rewardWeights: { ...REWARD_WEIGHT_DEFAULTS },
  entropyByAlgorithm: { ppo: 0, drgrpo: 0.01 },
  widthByAlgorithm: { ppo: 64, drgrpo: 64 },
  seed: 2026,
  worker: null,
  training: false,
  rollouts: [],
  selected: -1,
  pendingLatest: -1,
  latestGroupPaths: [],
  latestGroupAdvantages: new Float32Array(0),
  frame: 0,
  lastFrameAt: 0,
};

function setStatus(label, note) {
  $("#trainingStatus").textContent = label;
  if (note) $("#statusNote").textContent = note;
}

function setTraining(active, label = active ? "Training" : "Idle") {
  state.training = active;
  $("#startButton").disabled = active;
  $("#stopButton").disabled = !active;
  $("#budgetSlider").disabled = active;
  $("#horizonSlider").disabled = active;
  $("#entropySlider").disabled = active;
  $("#curriculumToggle").disabled = active;
  document.querySelectorAll("[data-algorithm], [data-width], [data-reward-shaping], [data-reward-weight]").forEach((control) => {
    control.disabled = active;
  });
  $("#trainingPill").dataset.active = active ? "true" : "false";
  $(".live-indicator").classList.toggle("is-running", active);
  $(".live-indicator").lastChild.textContent = active ? " Training in background" : " Worker ready";
  setStatus(label);
}

function formatReturn(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}

function formatBudget(value) {
  if (value >= 1_000_000) return `${value / 1_000_000}M`;
  return `${Math.round(value / 1000)}k`;
}

function formatRewardWeight(value) {
  return String(Number(value.toFixed(4)));
}

function formatAlgorithm(algorithm) {
  return algorithm === "drgrpo" ? "Dr.GRPO" : algorithm.toUpperCase();
}

function advantagePathStyle(advantage, positiveScale, negativeScale) {
  if (Math.abs(advantage) < 1e-8) {
    return { color: "rgba(90, 96, 92, .28)", width: 0.9 };
  }
  const scale = advantage > 0 ? positiveScale : negativeScale;
  const normalized = Math.min(1, Math.abs(advantage) / Math.max(1e-8, scale));
  const strength = Math.sqrt(normalized);
  const alpha = 0.3 + strength * 0.6;
  return {
    color: advantage > 0
      ? `rgba(29, 102, 219, ${alpha})`
      : `rgba(218, 92, 43, ${alpha})`,
    width: 0.9 + strength * 1.35,
  };
}

function budgetFromExponent(exponent) {
  return Math.round(10 ** exponent / 1000) * 1000;
}

function updateEntropyControl() {
  const config = entropyConfigs[state.algorithm];
  const slider = $("#entropySlider");
  slider.min = String(config.min);
  slider.max = String(config.max);
  slider.step = String(config.step);
  slider.value = String(state.entropyByAlgorithm[state.algorithm]);
  slider.setAttribute("aria-label", config.label);
  $("#entropyLabel").textContent = config.label;
  $("#entropyOutput").textContent = state.entropyByAlgorithm[state.algorithm].toFixed(config.digits);
  $("#entropyMin").textContent = String(config.min);
  $("#entropyMax").textContent = config.max.toFixed(config.digits === 3 ? 2 : 1);
}

function updateWidthControl() {
  document.querySelectorAll("[data-width]").forEach((button) => {
    button.setAttribute(
      "aria-pressed",
      String(Number(button.dataset.width) === state.widthByAlgorithm[state.algorithm]),
    );
  });
}

function updateStochasticEvalControl() {
  $("#stochasticEvalToggle").setAttribute("aria-pressed", String(state.stochasticEval));
}

function drawReturnChart() {
  const canvas = $("#returnCanvas");
  const context = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  context.strokeStyle = "rgba(25, 28, 27, .08)";
  context.lineWidth = 1;
  for (let line = 1; line < 4; line += 1) {
    const y = (rect.height * line) / 4;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(rect.width, y);
    context.stroke();
  }

  if (!state.rollouts.length) {
    context.fillStyle = "#8a8f89";
    context.font = '10px "DM Mono", monospace';
    context.textAlign = "center";
    context.fillText("Return history appears when training starts", rect.width / 2, rect.height / 2 + 4);
    return;
  }

  const padding = { left: 10, right: 10, top: 12, bottom: 14 };
  const values = state.rollouts.map((rollout) => rollout.return);
  const trainValues = state.rollouts.map((rollout) => rollout.trainReturn);
  const evaluationSma = simpleMovingAverage(values, RETURN_SMA_WINDOW);
  const trainingSma = simpleMovingAverage(trainValues, RETURN_SMA_WINDOW);
  const chartValues = [...evaluationSma, ...trainingSma].filter(Number.isFinite);
  let minimum = Math.min(...chartValues);
  let maximum = Math.max(...chartValues);
  if (minimum === maximum) {
    minimum -= 0.05;
    maximum += 0.05;
  }
  const x = (index) => padding.left + index / Math.max(1, values.length - 1) * (rect.width - padding.left - padding.right);
  const y = (value) => padding.top + (maximum - value) / (maximum - minimum) * (rect.height - padding.top - padding.bottom);

  context.beginPath();
  evaluationSma.forEach((value, index) => {
    if (index === 0) context.moveTo(x(index), y(value));
    else context.lineTo(x(index), y(value));
  });
  context.strokeStyle = "#1d66db";
  context.lineWidth = 2;
  context.stroke();

  context.beginPath();
  let trainSegmentStarted = false;
  trainingSma.forEach((value, index) => {
    if (!Number.isFinite(value)) {
      trainSegmentStarted = false;
      return;
    }
    if (!trainSegmentStarted) context.moveTo(x(index), y(value));
    else context.lineTo(x(index), y(value));
    trainSegmentStarted = true;
  });
  context.save();
  context.setLineDash([4, 3]);
  context.strokeStyle = "#d06a3e";
  context.lineWidth = 1.6;
  context.stroke();
  context.restore();

  if (state.selected >= 0) {
    const selected = state.rollouts[state.selected];
    context.beginPath();
    context.arc(x(state.selected), y(evaluationSma[state.selected]), 4, 0, Math.PI * 2);
    context.fillStyle = "#ed6b55";
    context.fill();
    if (Number.isFinite(trainingSma[state.selected])) {
      context.beginPath();
      context.arc(x(state.selected), y(trainingSma[state.selected]), 3, 0, Math.PI * 2);
      context.fillStyle = "#d06a3e";
      context.fill();
    }
  }
}

function drawRolloutValueChart() {
  const canvas = $("#valueCanvas");
  const context = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);

  const padding = { left: 48, right: 48, top: 15, bottom: 24 };
  const plotWidth = Math.max(1, rect.width - padding.left - padding.right);
  const plotHeight = Math.max(1, rect.height - padding.top - padding.bottom);
  context.strokeStyle = "rgba(25, 28, 27, .08)";
  context.lineWidth = 1;
  for (let line = 0; line <= 4; line += 1) {
    const y = padding.top + plotHeight * line / 4;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(padding.left + plotWidth, y);
    context.stroke();
  }

  if (state.selected < 0) {
    context.fillStyle = "#8a8f89";
    context.font = '10px "DM Mono", monospace';
    context.textAlign = "center";
    context.fillText("Value and reward appear with a logged rollout", rect.width / 2, rect.height / 2 + 4);
    return;
  }

  const rollout = state.rollouts[state.selected];
  const hasValueEstimate = rollout.hasValueEstimate;
  const values = [];
  const rewards = [];
  for (let frame = 0; frame < rollout.count; frame += 1) {
    const offset = frame * ROLLOUT_FRAME_SIZE;
    values.push(rollout.frames[offset + 10]);
    rewards.push(rollout.frames[offset + 11]);
  }
  const range = (series, includeZero = false) => {
    let minimum = Math.min(...series);
    let maximum = Math.max(...series);
    if (includeZero) {
      minimum = Math.min(0, minimum);
      maximum = Math.max(0, maximum);
    }
    if (minimum === maximum) {
      const margin = Math.max(0.05, Math.abs(minimum) * 0.1);
      minimum -= margin;
      maximum += margin;
    } else {
      const margin = (maximum - minimum) * 0.08;
      minimum -= margin;
      maximum += margin;
    }
    return [minimum, maximum];
  };
  const [valueMinimum, valueMaximum] = hasValueEstimate ? range(values) : [0, 1];
  const [rewardMinimum, rewardMaximum] = range(rewards, true);
  const x = (frame) => padding.left + frame / Math.max(1, rollout.count - 1) * plotWidth;
  const valueY = (value) => padding.top + (valueMaximum - value) / (valueMaximum - valueMinimum) * plotHeight;
  const rewardY = (reward) => padding.top + (rewardMaximum - reward) / (rewardMaximum - rewardMinimum) * plotHeight;

  const rewardZero = rewardY(0);
  context.strokeStyle = "rgba(237, 107, 85, .5)";
  context.lineWidth = Math.max(1, Math.min(3, plotWidth / Math.max(1, rollout.count) * 0.7));
  rewards.forEach((reward, frame) => {
    if (reward === 0) return;
    context.beginPath();
    context.moveTo(x(frame), rewardZero);
    context.lineTo(x(frame), rewardY(reward));
    context.stroke();
  });

  if (hasValueEstimate) {
    context.beginPath();
    values.forEach((value, frame) => {
      if (frame === 0) context.moveTo(x(frame), valueY(value));
      else context.lineTo(x(frame), valueY(value));
    });
    context.strokeStyle = "#1d66db";
    context.lineWidth = 2;
    context.stroke();
  }

  const markerX = x(Math.min(state.frame, rollout.count - 1));
  context.save();
  context.setLineDash([3, 4]);
  context.strokeStyle = "rgba(25, 28, 27, .38)";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(markerX, padding.top);
  context.lineTo(markerX, padding.top + plotHeight);
  context.stroke();
  context.restore();

  context.font = '9px "DM Mono", monospace';
  if (hasValueEstimate) {
    context.fillStyle = "#1d66db";
    context.textAlign = "right";
    context.fillText(valueMaximum.toFixed(2), padding.left - 7, padding.top + 3);
    context.fillText(valueMinimum.toFixed(2), padding.left - 7, padding.top + plotHeight);
  }
  context.fillStyle = "#c9513d";
  context.textAlign = "left";
  context.fillText(rewardMaximum.toFixed(2), padding.left + plotWidth + 7, padding.top + 3);
  context.fillText(rewardMinimum.toFixed(2), padding.left + plotWidth + 7, padding.top + plotHeight);
  context.fillStyle = "#8a8f89";
  context.textAlign = "center";
  context.fillText("rollout step", padding.left + plotWidth / 2, rect.height - 7);
}

function selectRollout(index) {
  if (!state.rollouts.length) return;
  state.selected = Math.max(0, Math.min(state.rollouts.length - 1, index));
  state.pendingLatest = state.selected < state.rollouts.length - 1 ? state.rollouts.length - 1 : -1;
  state.frame = 0;
  state.lastFrameAt = 0;
  const rollout = state.rollouts[state.selected];
  const evaluationMode = rollout.stochasticEvaluation ? "Stochastic eval" : "Deterministic eval";
  $("#rolloutSlider").value = String(state.selected);
  $("#rolloutStepOutput").textContent = `step ${rollout.step.toLocaleString()}`;
  $("#selectedStep").textContent = rollout.step.toLocaleString();
  $("#selectedReturn").textContent = formatReturn(rollout.return);
  $("#selectedCoverage").textContent = `${(rollout.coverage * 100).toFixed(1)}%`;
  $("#selectedOutcome").textContent = rollout.success
    ? "Complete"
    : rollout.wallContact
      ? "Wall contact"
      : rollout.blockWallContact
        ? "Block wall"
        : rollout.stalled
          ? "No movement"
          : "Timed out";
  $("#rolloutReturnLabel").textContent =
    `${evaluationMode.toLowerCase()} mean ${formatReturn(rollout.return)} · path ${formatReturn(rollout.representativeReturn)}`;
  $("#rolloutStatus").textContent = `${state.selected + 1} of ${state.rollouts.length}`;
  $("#rolloutPill").dataset.active = "record";
  const peerLabel = `${rollout.evaluationPaths.length} faded eval paths`;
  $("#canvasPathLabel").textContent = rollout.groupPaths.length
    ? `${evaluationMode} path · ${peerLabel} · ${rollout.groupPaths.length} GRPO paths`
    : `${evaluationMode} path · ${peerLabel} · action μ / 1σ radar`;
  $("#advantageLegend").hidden = !rollout.groupPaths.length;
  $("#valueLegendItem").hidden = !rollout.hasValueEstimate;
  $("#valueLegend").textContent = "V(s)";
  $("#valuePlotTitle").textContent = rollout.hasValueEstimate
    ? "Estimated remaining return V(s)"
    : "Reward over rollout · Dr.GRPO has no critic";
  drawReturnChart();
  drawRolloutValueChart();
}

function addRollout(message) {
  const frames = message.frames;
  const path = rolloutPusherPath(frames, message.count, ROLLOUT_FRAME_SIZE);
  if (message.groupPaths?.length) {
    state.latestGroupPaths = message.groupPaths;
    state.latestGroupAdvantages = message.groupAdvantages;
  }
  state.rollouts.push({
    step: message.step,
    return: message.return,
    representativeReturn: message.representativeReturn,
    evaluationRollouts: message.evaluationRollouts,
    stochasticEvaluation: Boolean(message.stochasticEvaluation),
    trainReturn: Number.isFinite(message.trainReturn) ? message.trainReturn : null,
    success: message.success,
    wallContact: message.wallContact,
    blockWallContact: message.blockWallContact,
    stalled: message.stalled,
    coverage: message.coverage,
    count: message.count,
    algorithm: message.progress.algorithm,
    hasValueEstimate: message.hasValueEstimate,
    evaluationBlockTracks: message.evaluationBlockTracks ?? [],
    evaluationPaths: message.evaluationPaths ?? [],
    groupPaths: state.latestGroupPaths,
    groupAdvantages: state.latestGroupAdvantages,
    squashed: message.squashed,
    frames,
    path,
  });
  const slider = $("#rolloutSlider");
  slider.disabled = false;
  slider.max = String(state.rollouts.length - 1);
  $("#rolloutEmpty").hidden = true;
  $("#stepsMetric").textContent = message.progress.steps.toLocaleString();
  $("#episodesMetric").textContent = message.progress.episodes.toLocaleString();
  $("#successMetric").textContent = message.progress.successes.toLocaleString();
  $("#rolloutMetric").textContent = state.rollouts.length.toLocaleString();
  $("#latestReturn").textContent = formatReturn(message.return);
  if (state.selected < 0) selectRollout(0);
  else {
    state.pendingLatest = state.rollouts.length - 1;
    drawReturnChart();
  }
}

function resetRun() {
  state.rollouts = [];
  state.selected = -1;
  state.pendingLatest = -1;
  state.latestGroupPaths = [];
  state.latestGroupAdvantages = new Float32Array(0);
  state.frame = 0;
  const slider = $("#rolloutSlider");
  slider.disabled = true;
  slider.min = "0";
  slider.max = "0";
  slider.value = "0";
  $("#rolloutEmpty").hidden = false;
  $("#rolloutStepOutput").textContent = "—";
  $("#selectedStep").textContent = "—";
  $("#selectedReturn").textContent = "—";
  $("#selectedCoverage").textContent = "—";
  $("#selectedOutcome").textContent = "—";
  $("#rolloutReturnLabel").textContent = "eval mean — · path —";
  $("#canvasPathLabel").textContent = `${state.stochasticEval ? "Stochastic" : "Deterministic"} eval path · action μ / 1σ radar`;
  $("#advantageLegend").hidden = true;
  $("#valueLegendItem").hidden = false;
  $("#valueLegend").textContent = "V(s)";
  $("#valuePlotTitle").textContent = "Estimated remaining return V(s)";
  $("#rolloutStatus").textContent = "Awaiting training";
  $("#latestReturn").textContent = "—";
  $("#stepsMetric").textContent = "0";
  $("#episodesMetric").textContent = "0";
  $("#successMetric").textContent = "0";
  $("#rolloutMetric").textContent = "0";
  drawReturnChart();
  drawRolloutValueChart();
}

function stopTraining(message = "Training stopped. Logged rollouts remain available for inspection.") {
  if (state.worker) state.worker.terminate();
  state.worker = null;
  setTraining(false, "Stopped");
  $("#statusNote").textContent = message;
}

function startTraining() {
  if (state.training) return;
  resetRun();
  state.worker = new Worker(new URL("./rl.worker.js", import.meta.url), { type: "module" });
  state.worker.onmessage = ({ data }) => {
    if (data.type === "started") {
      setStatus(`Training ${formatAlgorithm(data.algorithm)}`);
      return;
    }
    if (data.type === "rollout") {
      addRollout(data);
      return;
    }
    if (data.type === "done") {
      state.worker?.terminate();
      state.worker = null;
      setTraining(false, "Complete");
      $("#statusNote").textContent = "Training complete. Scrub the saved rollouts to inspect how the policy changed.";
      return;
    }
    if (data.type === "error") {
      stopTraining(`Training failed: ${data.message}`);
    }
  };
  state.worker.onerror = (event) => stopTraining(`Training failed: ${event.message}`);
  setTraining(true);
  $("#statusNote").textContent = `Training ${formatAlgorithm(state.algorithm)} for ${state.budget.toLocaleString()} steps. The page remains interactive.`;
  state.worker.postMessage({
    type: "start",
    algorithm: state.algorithm,
    totalSteps: state.budget,
    width: state.widthByAlgorithm[state.algorithm],
    horizon: state.horizon,
    entropyBonus: state.entropyByAlgorithm[state.algorithm],
    curriculum: state.curriculum,
    rewardShaping: { ...state.rewardShaping },
    rewardWeights: { ...state.rewardWeights },
    stochasticEval: state.stochasticEval,
    seed: state.seed,
  });
}

function draw(timestamp) {
  if (state.selected < 0) {
    view.draw(world, { coverage: world.coverage() });
    requestAnimationFrame(draw);
    return;
  }

  const rollout = state.rollouts[state.selected];
  if (!state.lastFrameAt) state.lastFrameAt = timestamp;
  if (timestamp - state.lastFrameAt >= 33) {
    if (state.frame >= rollout.count - 1) {
      if (state.pendingLatest > state.selected) {
        selectRollout(state.pendingLatest);
        requestAnimationFrame(draw);
        return;
      } else {
        state.frame = 0;
      }
    } else {
      state.frame += 1;
    }
    state.lastFrameAt = timestamp;
    drawRolloutValueChart();
  }
  const offset = state.frame * ROLLOUT_FRAME_SIZE;
  world.pusher.x = rollout.frames[offset];
  world.pusher.y = rollout.frames[offset + 1];
  world.block.x = rollout.frames[offset + 2];
  world.block.y = rollout.frames[offset + 3];
  world.block.angle = rollout.frames[offset + 4];
  const positiveAdvantageScale = Math.max(
    1e-8,
    ...rollout.groupAdvantages.filter((value) => value > 0),
  );
  const negativeAdvantageScale = Math.max(
    1e-8,
    ...rollout.groupAdvantages.filter((value) => value < 0).map((value) => -value),
  );
  const groupPaths = rollout.groupPaths.map((points, index) => {
    const style = advantagePathStyle(
      rollout.groupAdvantages[index] ?? 0,
      positiveAdvantageScale,
      negativeAdvantageScale,
    );
    return { points, ...style };
  });
  const evaluationPaths = rollout.evaluationPaths.map((points) => ({
    points,
    cursor: state.frame + 1,
    color: "rgba(29, 102, 219, .14)",
    width: 1.1,
  }));
  const evaluationBlocks = rollout.evaluationBlockTracks.map((track) => {
    const peerFrame = Math.min(state.frame, track.length / 3 - 1) * 3;
    return {
      x: track[peerFrame],
      y: track[peerFrame + 1],
      angle: track[peerFrame + 2],
    };
  });
  view.draw(world, {
    paths: [
      ...groupPaths,
      ...evaluationPaths,
      { points: rollout.path, cursor: state.frame + 1, color: "rgba(25, 28, 27, .78)", width: 1.9 },
    ],
    ghosts: evaluationBlocks,
    coverage: rollout.frames[offset + 5],
    actionDistribution: {
      meanX: rollout.frames[offset + 6],
      meanY: rollout.frames[offset + 7],
      logStdX: rollout.frames[offset + 8],
      logStdY: rollout.frames[offset + 9],
      squashed: rollout.squashed,
    },
  });
  requestAnimationFrame(draw);
}

function registerWebMcpTools() {
  const context = document.modelContext;
  if (!context?.registerTool) return null;
  const lifecycle = new AbortController();
  const register = (tool) => {
    Promise.resolve(context.registerTool(tool, { signal: lifecycle.signal })).catch((error) => {
      console.warn("Unable to register RL page tool", error);
    });
  };

  register({
    name: "start_rl_training",
    title: "Start RL training",
    description: "Start a PPO or Dr.GRPO Push-T training run using the visible page controls.",
    inputSchema: {
      type: "object",
      properties: {
        algorithm: { type: "string", enum: ["ppo", "drgrpo"] },
        budget: { type: "integer", minimum: BUDGET_MIN, maximum: BUDGET_MAX },
        horizon: { type: "integer", minimum: 200, maximum: 1000, multipleOf: 100 },
        width: { type: "integer", enum: [64, 128, 256] },
        entropyBonus: { type: "number", minimum: 0, maximum: 0.5 },
        curriculum: { type: "boolean" },
        stochasticEval: { type: "boolean" },
        rewardShaping: {
          type: "object",
          properties: {
            completion: { type: "boolean" },
            blockDistance: { type: "boolean" },
            pusherDistance: { type: "boolean" },
            orientation: { type: "boolean" },
            closeness: { type: "boolean" },
            pusherWall: { type: "boolean" },
            blockWall: { type: "boolean" },
            inactivity: { type: "boolean" },
            stepPenalty: { type: "boolean" },
          },
          required: REWARD_COMPONENTS,
          additionalProperties: false,
        },
        rewardWeights: {
          type: "object",
          properties: Object.fromEntries(REWARD_COMPONENTS.map((term) => [term, {
            type: "number",
            minimum: -100,
            maximum: 100,
          }])),
          required: REWARD_COMPONENTS,
          additionalProperties: false,
        },
      },
      required: ["algorithm", "budget", "horizon", "width", "entropyBonus", "curriculum", "rewardShaping", "rewardWeights"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false, untrustedContentHint: false },
    execute(input) {
      if (!input || typeof input !== "object") throw new TypeError("Training configuration is required.");
      if (!["ppo", "drgrpo"].includes(input.algorithm)) throw new RangeError("Algorithm must be ppo or drgrpo.");
      if (!Number.isInteger(input.budget) || input.budget < BUDGET_MIN || input.budget > BUDGET_MAX) {
        throw new RangeError("Budget must be an integer from 10000 through 1000000.");
      }
      if (!Number.isInteger(input.horizon) || input.horizon < 200 || input.horizon > 1000 || input.horizon % 100) {
        throw new RangeError("Horizon must be a multiple of 100 from 200 through 1000.");
      }
      if (![64, 128, 256].includes(input.width)) throw new RangeError("Width must be 64, 128, or 256.");
      const entropyConfig = entropyConfigs[input.algorithm];
      if (typeof input.entropyBonus !== "number" || !Number.isFinite(input.entropyBonus) ||
        input.entropyBonus < entropyConfig.min || input.entropyBonus > entropyConfig.max) {
        throw new RangeError(`Entropy must be from ${entropyConfig.min} through ${entropyConfig.max} for ${input.algorithm}.`);
      }
      if (!input.rewardShaping || REWARD_COMPONENTS.some(
        (term) => typeof input.rewardShaping[term] !== "boolean",
      )) {
        throw new TypeError("Every reward-shaping toggle must be true or false.");
      }
      if (!input.rewardWeights || REWARD_COMPONENTS.some((term) =>
        typeof input.rewardWeights[term] !== "number" || !Number.isFinite(input.rewardWeights[term]) ||
        input.rewardWeights[term] < -100 || input.rewardWeights[term] > 100)) {
        throw new TypeError("Every reward weight must be a finite number from -100 through 100.");
      }
      if (state.training) throw new Error("A training run is already active.");

      state.algorithm = input.algorithm;
      state.budget = input.budget;
      state.horizon = input.horizon;
      state.widthByAlgorithm[state.algorithm] = input.width;
      state.entropyByAlgorithm[state.algorithm] = input.entropyBonus;
      state.curriculum = input.curriculum;
      if (typeof input.stochasticEval === "boolean") state.stochasticEval = input.stochasticEval;
      state.rewardShaping = { ...input.rewardShaping };
      state.rewardWeights = { ...input.rewardWeights };
      document.querySelectorAll("[data-algorithm]").forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.algorithm === state.algorithm));
      });
      updateWidthControl();
      $("#budgetSlider").value = String(Math.log10(state.budget));
      $("#budgetOutput").textContent = formatBudget(state.budget);
      $("#horizonSlider").value = String(state.horizon);
      $("#horizonOutput").textContent = state.horizon.toLocaleString();
      updateEntropyControl();
      updateCurriculumControl();
      updateStochasticEvalControl();
      updateRewardShapingControls();
      startTraining();
      return {
        status: "training",
        algorithm: state.algorithm,
        budget: state.budget,
        curriculum: state.curriculum,
        stochasticEval: state.stochasticEval,
        rewardShaping: { ...state.rewardShaping },
        rewardWeights: { ...state.rewardWeights },
      };
    },
  });

  register({
    name: "read_rl_training_status",
    title: "Read RL training status",
    description: "Read the current training state and selected rollout shown on the RL page.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true, untrustedContentHint: false },
    execute() {
      const selected = state.selected >= 0 ? state.rollouts[state.selected] : null;
      return {
        status: state.training ? "training" : "idle",
        algorithm: state.algorithm,
        budget: state.budget,
        horizon: state.horizon,
        width: state.widthByAlgorithm[state.algorithm],
        entropyBonus: state.entropyByAlgorithm[state.algorithm],
        curriculum: state.curriculum,
        stochasticEval: state.stochasticEval,
        rewardShaping: { ...state.rewardShaping },
        rewardWeights: { ...state.rewardWeights },
        loggedRollouts: state.rollouts.length,
        latestStep: state.rollouts.at(-1)?.step ?? 0,
        selectedRollout: selected ? {
          step: selected.step,
          return: selected.return,
          coverage: selected.coverage,
          success: selected.success,
        } : null,
      };
    },
  });

  register({
    name: "stop_rl_training",
    title: "Stop RL training",
    description: "Stop the active RL training worker while preserving all logged rollouts.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: false, untrustedContentHint: false },
    execute() {
      if (!state.training) return { status: "idle", loggedRollouts: state.rollouts.length };
      stopTraining();
      return { status: "stopped", loggedRollouts: state.rollouts.length };
    },
  });

  return lifecycle;
}

document.querySelectorAll("[data-algorithm]").forEach((button) => {
  button.addEventListener("click", () => {
    state.algorithm = button.dataset.algorithm;
    document.querySelectorAll("[data-algorithm]").forEach((option) => {
      option.setAttribute("aria-pressed", String(option === button));
    });
    updateEntropyControl();
    updateWidthControl();
  });
});

document.querySelectorAll("[data-width]").forEach((button) => {
  button.addEventListener("click", () => {
    state.widthByAlgorithm[state.algorithm] = Number(button.dataset.width);
    updateWidthControl();
  });
});

function updateCurriculumControl() {
  const button = $("#curriculumToggle");
  button.setAttribute("aria-pressed", String(state.curriculum));
  button.textContent = state.curriculum ? "Enabled" : "Disabled";
}

function updateRewardShapingControls() {
  document.querySelectorAll("[data-reward-shaping]").forEach((button) => {
    button.setAttribute("aria-pressed", String(state.rewardShaping[button.dataset.rewardShaping]));
  });
  document.querySelectorAll("[data-reward-weight]").forEach((input) => {
    input.value = formatRewardWeight(state.rewardWeights[input.dataset.rewardWeight]);
  });
}

$("#curriculumToggle").addEventListener("click", () => {
  state.curriculum = !state.curriculum;
  updateCurriculumControl();
});

$("#stochasticEvalToggle").addEventListener("click", () => {
  state.stochasticEval = !state.stochasticEval;
  updateStochasticEvalControl();
  state.worker?.postMessage({
    type: "set-evaluation-mode",
    stochastic: state.stochasticEval,
  });
});

document.querySelectorAll("[data-reward-shaping]").forEach((button) => {
  button.addEventListener("click", () => {
    const term = button.dataset.rewardShaping;
    state.rewardShaping[term] = !state.rewardShaping[term];
    updateRewardShapingControls();
  });
});

document.querySelectorAll("[data-reward-weight]").forEach((input) => {
  input.addEventListener("change", () => {
    const value = Number(input.value);
    if (Number.isFinite(value) && value >= -100 && value <= 100) {
      state.rewardWeights[input.dataset.rewardWeight] = value;
    }
    input.value = formatRewardWeight(state.rewardWeights[input.dataset.rewardWeight]);
  });
});

$("#budgetSlider").addEventListener("input", (event) => {
  state.budget = budgetFromExponent(Number(event.target.value));
  $("#budgetOutput").textContent = formatBudget(state.budget);
});
$("#horizonSlider").addEventListener("input", (event) => {
  state.horizon = Number(event.target.value);
  $("#horizonOutput").textContent = state.horizon.toLocaleString();
});
$("#entropySlider").addEventListener("input", (event) => {
  const config = entropyConfigs[state.algorithm];
  state.entropyByAlgorithm[state.algorithm] = Number(event.target.value);
  $("#entropyOutput").textContent = state.entropyByAlgorithm[state.algorithm].toFixed(config.digits);
});
$("#rolloutSlider").addEventListener("input", (event) => {
  selectRollout(Number(event.target.value));
});
$("#startButton").addEventListener("click", startTraining);
$("#stopButton").addEventListener("click", () => stopTraining());
window.addEventListener("resize", () => {
  drawReturnChart();
  drawRolloutValueChart();
});
const webMcpLifecycle = registerWebMcpTools();
window.addEventListener("beforeunload", () => {
  state.worker?.terminate();
  webMcpLifecycle?.abort();
});

updateEntropyControl();
updateWidthControl();
updateCurriculumControl();
updateStochasticEvalControl();
updateRewardShapingControls();
resetRun();
requestAnimationFrame(draw);
