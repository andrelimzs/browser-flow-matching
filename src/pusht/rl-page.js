import "./style.css";
import "./rl-style.css";
import { PushWorld, createRandom } from "./sim.js";
import { createView } from "./render.js";

const $ = (selector) => document.querySelector(selector);
const ROLLOUT_FRAME_SIZE = 12;
const BUDGET_MIN = 10_000;
const BUDGET_MAX = 1_000_000;
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
const entropyConfigs = {
  ppo: { label: "Entropy bonus", min: 0, max: 0.1, step: 0.005, digits: 3 },
  sac: { label: "Temperature α", min: 0, max: 0.5, step: 0.01, digits: 2 },
};
const world = new PushWorld({ random: createRandom(41), obstacleCount: 0 });
const view = createView($("#arena"));
const state = {
  algorithm: "ppo",
  budget: 1_000_000,
  horizon: 200,
  curriculum: true,
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
  entropyByAlgorithm: { ppo: 0, sac: 0.3 },
  widthByAlgorithm: { ppo: 64, sac: 256 },
  seed: 2026,
  worker: null,
  training: false,
  rollouts: [],
  selected: -1,
  pendingLatest: -1,
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
  document.querySelectorAll("[data-algorithm], [data-width], [data-reward-shaping]").forEach((button) => {
    button.disabled = active;
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
  let minimum = Math.min(...values);
  let maximum = Math.max(...values);
  if (minimum === maximum) {
    minimum -= 0.05;
    maximum += 0.05;
  }
  const x = (index) => padding.left + index / Math.max(1, values.length - 1) * (rect.width - padding.left - padding.right);
  const y = (value) => padding.top + (maximum - value) / (maximum - minimum) * (rect.height - padding.top - padding.bottom);

  context.beginPath();
  values.forEach((value, index) => {
    if (index === 0) context.moveTo(x(index), y(value));
    else context.lineTo(x(index), y(value));
  });
  context.strokeStyle = "#1d66db";
  context.lineWidth = 2;
  context.stroke();

  if (state.selected >= 0) {
    const selected = state.rollouts[state.selected];
    context.beginPath();
    context.arc(x(state.selected), y(selected.return), 4, 0, Math.PI * 2);
    context.fillStyle = "#ed6b55";
    context.fill();
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
  const [valueMinimum, valueMaximum] = range(values);
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

  context.beginPath();
  values.forEach((value, frame) => {
    if (frame === 0) context.moveTo(x(frame), valueY(value));
    else context.lineTo(x(frame), valueY(value));
  });
  context.strokeStyle = "#1d66db";
  context.lineWidth = 2;
  context.stroke();

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
  context.fillStyle = "#1d66db";
  context.textAlign = "right";
  context.fillText(valueMaximum.toFixed(2), padding.left - 7, padding.top + 3);
  context.fillText(valueMinimum.toFixed(2), padding.left - 7, padding.top + plotHeight);
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
  $("#rolloutReturnLabel").textContent = `return ${formatReturn(rollout.return)}`;
  $("#rolloutStatus").textContent = `${state.selected + 1} of ${state.rollouts.length}`;
  $("#rolloutPill").dataset.active = "record";
  const valueLabel = rollout.algorithm === "sac" ? "min Q(s,a)" : "V(s)";
  $("#valueLegend").textContent = valueLabel;
  $("#valuePlotTitle").textContent = `Estimated remaining return ${valueLabel}`;
  drawReturnChart();
  drawRolloutValueChart();
}

function addRollout(message) {
  const frames = message.frames;
  const path = new Float32Array(message.count * 2);
  for (let index = 0; index < message.count; index += 1) {
    path[index * 2] = frames[index * ROLLOUT_FRAME_SIZE];
    path[index * 2 + 1] = frames[index * ROLLOUT_FRAME_SIZE + 1];
  }
  state.rollouts.push({
    step: message.step,
    return: message.return,
    success: message.success,
    wallContact: message.wallContact,
    blockWallContact: message.blockWallContact,
    stalled: message.stalled,
    coverage: message.coverage,
    count: message.count,
    algorithm: message.progress.algorithm,
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
  $("#rolloutReturnLabel").textContent = "return —";
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
      setStatus(`Training ${data.algorithm.toUpperCase()}`);
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
  $("#statusNote").textContent = `Training ${state.algorithm.toUpperCase()} for ${state.budget.toLocaleString()} steps. The page remains interactive.`;
  state.worker.postMessage({
    type: "start",
    algorithm: state.algorithm,
    totalSteps: state.budget,
    width: state.widthByAlgorithm[state.algorithm],
    horizon: state.horizon,
    entropyBonus: state.entropyByAlgorithm[state.algorithm],
    curriculum: state.curriculum,
    rewardShaping: { ...state.rewardShaping },
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
  view.draw(world, {
    paths: [{ points: rollout.path, cursor: state.frame + 1, color: "rgba(237, 107, 85, .72)", width: 1.8 }],
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
    description: "Start a PPO or SAC Push-T training run using the visible page controls.",
    inputSchema: {
      type: "object",
      properties: {
        algorithm: { type: "string", enum: ["ppo", "sac"] },
        budget: { type: "integer", minimum: BUDGET_MIN, maximum: BUDGET_MAX },
        horizon: { type: "integer", minimum: 200, maximum: 1000, multipleOf: 100 },
        width: { type: "integer", enum: [64, 128, 256] },
        entropyBonus: { type: "number", minimum: 0, maximum: 0.5 },
        curriculum: { type: "boolean" },
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
      },
      required: ["algorithm", "budget", "horizon", "width", "entropyBonus", "curriculum", "rewardShaping"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false, untrustedContentHint: false },
    execute(input) {
      if (!input || typeof input !== "object") throw new TypeError("Training configuration is required.");
      if (!["ppo", "sac"].includes(input.algorithm)) throw new RangeError("Algorithm must be ppo or sac.");
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
      if (state.training) throw new Error("A training run is already active.");

      state.algorithm = input.algorithm;
      state.budget = input.budget;
      state.horizon = input.horizon;
      state.widthByAlgorithm[state.algorithm] = input.width;
      state.entropyByAlgorithm[state.algorithm] = input.entropyBonus;
      state.curriculum = input.curriculum;
      state.rewardShaping = { ...input.rewardShaping };
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
      updateRewardShapingControls();
      startTraining();
      return {
        status: "training",
        algorithm: state.algorithm,
        budget: state.budget,
        curriculum: state.curriculum,
        rewardShaping: { ...state.rewardShaping },
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
        rewardShaping: { ...state.rewardShaping },
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
}

$("#curriculumToggle").addEventListener("click", () => {
  state.curriculum = !state.curriculum;
  updateCurriculumControl();
});

document.querySelectorAll("[data-reward-shaping]").forEach((button) => {
  button.addEventListener("click", () => {
    const term = button.dataset.rewardShaping;
    state.rewardShaping[term] = !state.rewardShaping[term];
    updateRewardShapingControls();
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
updateRewardShapingControls();
resetRun();
requestAnimationFrame(draw);
