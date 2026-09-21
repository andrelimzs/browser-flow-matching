import "./style.css";
import "./rl-style.css";
import { PushWorld, createRandom } from "./sim.js";
import { createView } from "./render.js";

const $ = (selector) => document.querySelector(selector);
const budgets = [10_000, 50_000, 100_000];
const ROLLOUT_FRAME_SIZE = 10;
const world = new PushWorld({ random: createRandom(41), obstacleCount: 0 });
const view = createView($("#arena"));
const state = {
  algorithm: "ppo",
  budget: budgets[0],
  horizon: 600,
  width: 128,
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
  document.querySelectorAll("[data-algorithm], [data-width]").forEach((button) => {
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
  $("#selectedOutcome").textContent = rollout.success ? "Complete" : rollout.wallContact ? "Wall contact" : "Timed out";
  $("#rolloutReturnLabel").textContent = `return ${formatReturn(rollout.return)}`;
  $("#rolloutStatus").textContent = `${state.selected + 1} of ${state.rollouts.length}`;
  $("#rolloutPill").dataset.active = "record";
  drawReturnChart();
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
    coverage: message.coverage,
    count: message.count,
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
    width: state.width,
    horizon: state.horizon,
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
        budget: { type: "integer", enum: budgets },
        horizon: { type: "integer", minimum: 200, maximum: 1000, multipleOf: 100 },
        width: { type: "integer", enum: [64, 128] },
      },
      required: ["algorithm", "budget", "horizon", "width"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false, untrustedContentHint: false },
    execute(input) {
      if (!input || typeof input !== "object") throw new TypeError("Training configuration is required.");
      if (!["ppo", "sac"].includes(input.algorithm)) throw new RangeError("Algorithm must be ppo or sac.");
      if (!budgets.includes(input.budget)) throw new RangeError("Budget must be 10000, 50000, or 100000.");
      if (!Number.isInteger(input.horizon) || input.horizon < 200 || input.horizon > 1000 || input.horizon % 100) {
        throw new RangeError("Horizon must be a multiple of 100 from 200 through 1000.");
      }
      if (![64, 128].includes(input.width)) throw new RangeError("Width must be 64 or 128.");
      if (state.training) throw new Error("A training run is already active.");

      state.algorithm = input.algorithm;
      state.budget = input.budget;
      state.horizon = input.horizon;
      state.width = input.width;
      document.querySelectorAll("[data-algorithm]").forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.algorithm === state.algorithm));
      });
      document.querySelectorAll("[data-width]").forEach((button) => {
        button.setAttribute("aria-pressed", String(Number(button.dataset.width) === state.width));
      });
      $("#budgetSlider").value = String(budgets.indexOf(state.budget) + 1);
      $("#budgetOutput").textContent = `${state.budget / 1000}k`;
      $("#horizonSlider").value = String(state.horizon);
      $("#horizonOutput").textContent = state.horizon.toLocaleString();
      startTraining();
      return { status: "training", algorithm: state.algorithm, budget: state.budget };
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
  });
});

document.querySelectorAll("[data-width]").forEach((button) => {
  button.addEventListener("click", () => {
    state.width = Number(button.dataset.width);
    document.querySelectorAll("[data-width]").forEach((option) => {
      option.setAttribute("aria-pressed", String(option === button));
    });
  });
});

$("#budgetSlider").addEventListener("input", (event) => {
  state.budget = budgets[Number(event.target.value) - 1];
  $("#budgetOutput").textContent = `${state.budget / 1000}k`;
});
$("#horizonSlider").addEventListener("input", (event) => {
  state.horizon = Number(event.target.value);
  $("#horizonOutput").textContent = state.horizon.toLocaleString();
});
$("#rolloutSlider").addEventListener("input", (event) => {
  selectRollout(Number(event.target.value));
});
$("#startButton").addEventListener("click", startTraining);
$("#stopButton").addEventListener("click", () => stopTraining());
window.addEventListener("resize", drawReturnChart);
const webMcpLifecycle = registerWebMcpTools();
window.addEventListener("beforeunload", () => {
  state.worker?.terminate();
  webMcpLifecycle?.abort();
});

resetRun();
requestAnimationFrame(draw);
