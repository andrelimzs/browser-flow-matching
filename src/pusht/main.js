import "./style.css";
import { PushWorld, createRandom, SUCCESS_COVERAGE } from "./sim.js";
import { ScriptedExpert } from "./expert.js";
import { DemoStore } from "./demos.js";
import { createView } from "./render.js";
import { makePolicy, sampleChunk, ACTION_DIM, CHUNK } from "./policy.js";
import { MLP } from "../flow/mlp.js";

const EPISODE_CAP = 2200;
const TRAIL_LENGTH = 220;
const EXECUTE = 16;
// The raw lift sign is wrong often enough that an isolated flip would send the
// pusher through the block instead of into it; switching only on a confident
// value and holding otherwise turns those into no-ops.
const LIFT_ON = 0.4;
const LIFT_OFF = -0.4;

const $ = (selector) => document.querySelector(selector);

const random = createRandom(Date.now() >>> 0);
const world = new PushWorld({ random, obstacleCount: 1 });
const expert = new ScriptedExpert({ random });
const store = new DemoStore().restore();
const view = createView($("#arena"));

const state = {
  mode: "expert",
  running: false,
  recording: false,
  speed: 2,
  pointer: null,
  lifted: false,
  trail: [],
  lastAction: null,
  finished: null,
  episodeStart: null,
  policy: null,
  policyScales: null,
  policyObstacles: 0,
  policyReady: false,
  chunk: null,
  chunkCursor: Infinity,
  lift: 0,
  trainSteps: 4000,
  training: false,
  lossHistory: [],
};

let worker = null;

// Reset replays the current layout so a hard episode can be retried; New world
// draws a fresh one. Both clear any part-recorded demonstration.
function resetEpisode({ newWorld = false } = {}) {
  if (state.recording) store.discard();
  state.recording = false;
  if (newWorld || !state.episodeStart) {
    world.reset({ obstacleCount: Number($("#obstacleRange").value) });
    state.episodeStart = world.snapshot();
  } else {
    world.restore(state.episodeStart);
  }
  expert.reset();
  state.trail = [];
  state.lastAction = null;
  state.finished = null;
  state.chunk = null;
  state.chunkCursor = Infinity;
  state.lift = 0;
  syncUI();
}

function beginRecording() {
  world.reset({ obstacleCount: Number($("#obstacleRange").value) });
  state.episodeStart = world.snapshot();
  expert.reset();
  state.trail = [];
  state.lastAction = null;
  state.finished = null;
  store.begin(world, state.mode === "teleop" ? "mouse" : "scripted");
  state.recording = true;
  state.running = true;
  syncUI();
}

function finishRecording(keep = true) {
  if (!state.recording) return;
  const episode = store.end(world, { keep });
  state.recording = false;
  syncUI();
  return episode;
}

// Storage failures must reach the user: a hand-recorded demonstration that
// vanishes on reload is worse than one that was never offered.
function reportStorage(fallback) {
  setStatus(store.storageWarning ?? fallback);
}

function nextAction() {
  if (state.mode === "teleop") {
    // Hold position when the cursor leaves the arena, so the block is not
    // yanked by a stray pointer event. Held pointer button lifts the pusher.
    const target = state.pointer ?? [world.pusher.x, world.pusher.y];
    return [target[0], target[1], state.lifted ? 1 : 0];
  }
  if (state.mode === "policy") return policyAction();
  return expert.act(world);
}

// Action chunking: sample a chunk, execute part of it open loop, resample.
function policyAction() {
  if (!state.policyReady || world.obstacles.length !== state.policyObstacles) {
    return [world.pusher.x, world.pusher.y, 0];
  }
  if (state.chunkCursor >= EXECUTE || !state.chunk) {
    state.chunk = sampleChunk(state.policy, world.writeObservation(), {
      steps: 10,
      random: Math.random,
      scales: state.policyScales,
    });
    state.chunkCursor = 0;
  }
  const base = state.chunkCursor * ACTION_DIM;
  const raw = state.chunk[base + 2];
  if (raw > LIFT_ON) state.lift = 1;
  else if (raw < LIFT_OFF) state.lift = 0;
  state.chunkCursor += 1;
  return [state.chunk[base], state.chunk[base + 1], state.lift];
}

function advance() {
  if (state.finished) return;
  const [actionX, actionY, lift] = nextAction();
  if (state.recording) store.record(world.writeObservation(), actionX, actionY, lift);
  world.step(actionX, actionY, lift);
  state.lastAction = [actionX, actionY, lift];
  state.trail.push([world.pusher.x, world.pusher.y]);
  if (state.trail.length > TRAIL_LENGTH) state.trail.shift();

  if (world.coverage() >= SUCCESS_COVERAGE || world.steps >= EPISODE_CAP) {
    const solved = world.coverage() >= SUCCESS_COVERAGE;
    state.finished = solved ? "solved" : "timeout";
    state.running = false;
    const wasRecording = state.recording;
    const episode = finishRecording(true);
    if (wasRecording) {
      const outcome = solved ? "Solved" : "Timed out";
      reportStorage(episode ? `${outcome} — episode saved.` : `${outcome} — episode too short to keep.`);
    }
    syncUI();
  }
}

// Batch collection runs the expert without rendering: at roughly 70k steps per
// second a ten-episode set costs a few tens of milliseconds, so it can be done
// synchronously without blocking the frame budget in any visible way.
function collect(count) {
  const collector = new ScriptedExpert({ random, tieBreak: expert.tieBreak });
  const obstacleCount = Number($("#obstacleRange").value);
  let kept = 0;
  for (let episode = 0; episode < count; episode++) {
    world.reset({ obstacleCount });
    collector.reset();
    store.begin(world, "scripted");
    for (let step = 0; step < EPISODE_CAP; step++) {
      if (world.coverage() >= SUCCESS_COVERAGE) break;
      const observation = world.writeObservation();
      const [actionX, actionY, lift] = collector.act(world);
      store.record(observation, actionX, actionY, lift);
      world.step(actionX, actionY, lift);
    }
    if (store.end(world, { keep: true })) kept += 1;
  }
  world.reset({ obstacleCount });
  state.episodeStart = world.snapshot();
  expert.reset();
  state.trail = [];
  state.lastAction = null;
  state.finished = null;
  state.running = false;
  reportStorage(`Collected ${kept} scripted episodes.`);
  syncUI();
}

function startTraining() {
  const solved = store.episodes.filter((episode) => episode.success !== false);
  if (solved.length < 3) {
    setStatus("Collect some demonstrations first — the policy trains on solved episodes.");
    return;
  }
  state.training = true;
  state.lossHistory = [];
  state.running = false;
  $("#trainProgress").hidden = false;

  worker = new Worker(new URL("./trainer.worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (event) => {
    const data = event.data;
    if (data.type === "started") {
      // Build the policy up front so the first weight snapshot can go straight in.
      preparePolicy(data);
      $("#trainStatus").textContent = `${data.episodes} episodes · ${data.transitions.toLocaleString()} transitions · ${data.params.toLocaleString()} params`;
    } else if (data.type === "weights") {
      adoptWeights(data.weights);
    } else if (data.type === "progress") {
      state.lossHistory.push(data.loss);
      const share = data.step / data.steps;
      const remaining = share > 0.02 ? (data.elapsed / share - data.elapsed) / 1000 : null;
      $("#trainStatus").textContent =
        `step ${data.step.toLocaleString()} / ${data.steps.toLocaleString()} · loss ${data.loss.toFixed(2)}` +
        (remaining ? ` · ~${Math.ceil(remaining)}s left` : "");
      drawLoss();
    } else if (data.type === "done") {
      adoptWeights(data.weights);
      stopTraining();
      setStatus(`Trained in ${(data.elapsed / 1000).toFixed(0)}s, final loss ${data.loss.toFixed(2)}.`);
    } else if (data.type === "error") {
      stopTraining();
      setStatus(data.message);
    }
  };
  worker.postMessage({ episodes: solved, steps: state.trainSteps, width: 128, learningRate: 0.002 });
  setStatus("Training in a background worker — the page stays interactive.");
  syncUI();
}

function stopTraining() {
  if (worker) worker.terminate();
  worker = null;
  state.training = false;
  syncUI();
}

// Builds an inference-sized copy of the network the worker is training, ready
// to receive weight snapshots.
function preparePolicy({ observationSize, sizes, scales }) {
  const policy = makePolicy({ observationSize, width: 128, maxBatch: 1, random: Math.random });
  policy.model = new MLP({ sizes, maxBatch: 1, random: Math.random });
  state.policy = policy;
  state.policyScales = Float32Array.from(scales);
  // The observation width is 11 + 3 per obstacle, so a policy is tied to the
  // obstacle count it trained on. Switching the slider afterwards would feed it
  // the wrong shape, so the count is recorded and restored with the mode.
  state.policyObstacles = world.obstacles.length;
  state.policyReady = false;
}

// Snapshots arrive throughout training, so Policy mode can be watched while it
// is still learning. Writing into the existing buffers allocates nothing, and
// the main thread is single-threaded, so a sample can never read a half-written
// network.
function adoptWeights(weights) {
  if (!state.policy) return;
  state.policy.model.loadSnapshot(weights);
  if (!state.policyReady) {
    state.policyReady = true;
    $("#modePolicy").disabled = false;
    setStatus("Policy is live — switch to Policy mode to watch it improve as it trains.");
  }
}

function drawLoss() {
  const canvas = $("#lossCanvas");
  const context = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  const history = state.lossHistory;
  if (history.length < 2) return;
  const maximum = Math.max(...history);
  const minimum = Math.min(...history);
  const span = Math.max(1e-6, maximum - minimum);
  context.beginPath();
  history.forEach((value, index) => {
    const x = (index / (history.length - 1)) * rect.width;
    const y = rect.height - ((value - minimum) / span) * (rect.height - 6) - 3;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.strokeStyle = "#1d66db";
  context.lineWidth = 1.4;
  context.stroke();
}

function setStatus(message) {
  $("#statusNote").textContent = message;
}

function syncUI() {
  const coverage = world.coverage();
  const stats = store.stats();

  $("#coverageMetric").textContent = `${(coverage * 100).toFixed(1)}%`;
  $("#coverageLabel").textContent = `coverage ${(coverage * 100).toFixed(0)}%`;
  $("#stepsMetric").textContent = String(world.steps);
  $("#stateMetric").textContent =
    state.mode === "teleop" ? "teleop"
      : state.mode === "policy" ? (state.training ? "flow policy (training)" : "flow policy")
      : expert.describe();
  $("#demoMetric").textContent = `${stats.total}`;
  $("#successMetric").textContent = stats.total ? `${(stats.successRate * 100).toFixed(0)}%` : "—";
  $("#transitionMetric").textContent = stats.steps.toLocaleString();

  $("#modeHeading").textContent =
    state.mode === "teleop" ? "Mouse teleoperation" : state.mode === "policy" ? "Learned policy" : "Scripted expert";
  $("#runButton").textContent = state.running ? "Pause" : "Run";
  $("#recordButton").textContent = state.recording ? "Stop" : "Record";
  $("#recordButton").dataset.active = String(state.recording);
  $("#collectButton").disabled = state.recording || state.mode !== "expert" || state.training;
  $("#clearButton").disabled = state.recording || state.training;
  $("#trainButton").disabled = state.training;
  $("#stopTrainButton").disabled = !state.training;
  $("#recordButton").disabled = state.training || state.mode === "policy";
  $("#teleopHint").hidden = state.mode !== "teleop";
  $(".canvas-shell").dataset.teleop = String(state.mode === "teleop");

  const pill = $(".phase-pill");
  const label = $("#phaseLabel");
  if (state.recording) {
    pill.dataset.active = "record";
    label.textContent = "Recording";
  } else if (state.finished === "solved") {
    pill.dataset.active = "true";
    label.textContent = "Solved";
  } else if (state.finished === "timeout") {
    pill.dataset.active = "false";
    label.textContent = "Timed out";
  } else if (state.running) {
    pill.dataset.active = "true";
    label.textContent = "Running";
  } else {
    pill.dataset.active = "false";
    label.textContent = "Idle";
  }
}

function animate() {
  if (state.running) {
    const steps = state.mode === "teleop" ? 1 : state.speed;
    for (let index = 0; index < steps; index++) advance();
    syncUI();
  }
  view.draw(world, {
    coverage: world.coverage(),
    trail: state.trail,
    action: state.lastAction,
    showAction: state.mode !== "teleop",
  });
  requestAnimationFrame(animate);
}

// Controls

$("#runButton").addEventListener("click", () => {
  if (state.finished) resetEpisode({ newWorld: true });
  state.running = !state.running;
  syncUI();
});

$("#resetButton").addEventListener("click", () => {
  state.running = false;
  resetEpisode({ newWorld: false });
});

$("#newWorld").addEventListener("click", () => {
  state.running = false;
  resetEpisode({ newWorld: true });
});

$("#recordButton").addEventListener("click", () => {
  if (state.recording) {
    state.running = false;
    const episode = finishRecording(true);
    reportStorage(episode ? "Episode saved." : "Episode too short to keep.");
  } else {
    beginRecording();
    setStatus(
      state.mode === "teleop"
        ? "Recording: steer the pusher with the cursor. Press Stop or solve the task to save."
        : "Recording the scripted expert.",
    );
  }
});

$("#collectButton").addEventListener("click", () => collect(10));

$("#exportButton").addEventListener("click", () => {
  if (!store.episodes.length) {
    setStatus("No demonstrations to export yet.");
    return;
  }
  store.download();
});

$("#clearButton").addEventListener("click", () => {
  // Clearing mid-recording would drop the in-flight episode while the UI still
  // claimed to be recording, so the button is disabled then; this is a guard in
  // case it is reached another way.
  if (state.recording) {
    setStatus("Stop the recording before clearing.");
    return;
  }
  store.clear();
  reportStorage("Demonstrations cleared.");
  syncUI();
});

for (const button of document.querySelectorAll("[data-speed]")) {
  button.addEventListener("click", () => {
    state.speed = Number(button.dataset.speed);
    for (const other of document.querySelectorAll("[data-speed]")) {
      other.setAttribute("aria-pressed", String(other === button));
    }
  });
}

for (const button of document.querySelectorAll("[data-tie]")) {
  button.addEventListener("click", () => {
    expert.tieBreak = button.dataset.tie;
    for (const other of document.querySelectorAll("[data-tie]")) {
      other.setAttribute("aria-pressed", String(other === button));
    }
  });
}

function setMode(mode) {
  if (mode === "policy") {
    if (!state.policyReady) return;
    // Restore the layout the policy was trained for.
    const range = $("#obstacleRange");
    if (Number(range.value) !== state.policyObstacles) {
      range.value = String(state.policyObstacles);
      $("#obstacleOutput").textContent = range.value;
      setStatus(`Policy was trained with ${state.policyObstacles} obstacle${state.policyObstacles === 1 ? "" : "s"}; the arena has been set to match.`);
    }
  }
  state.mode = mode;
  state.running = false;
  $("#modeExpert").setAttribute("aria-pressed", String(mode === "expert"));
  $("#modeTeleop").setAttribute("aria-pressed", String(mode === "teleop"));
  $("#modePolicy").setAttribute("aria-pressed", String(mode === "policy"));
  resetEpisode({ newWorld: true });
}

$("#modeExpert").addEventListener("click", () => setMode("expert"));
$("#modeTeleop").addEventListener("click", () => setMode("teleop"));
$("#modePolicy").addEventListener("click", () => { if (state.policyReady) setMode("policy"); });

$("#trainButton").addEventListener("click", startTraining);
$("#stopTrainButton").addEventListener("click", () => {
  stopTraining();
  setStatus("Training stopped.");
});

for (const button of document.querySelectorAll("[data-steps]")) {
  button.addEventListener("click", () => {
    state.trainSteps = Number(button.dataset.steps);
    for (const other of document.querySelectorAll("[data-steps]")) {
      other.setAttribute("aria-pressed", String(other === button));
    }
  });
}

$("#obstacleRange").addEventListener("input", (event) => {
  $("#obstacleOutput").textContent = event.target.value;
});

$("#obstacleRange").addEventListener("change", () => {
  state.running = false;
  resetEpisode({ newWorld: true });
});

const arena = $("#arena");
arena.addEventListener("pointermove", (event) => {
  if (state.mode !== "teleop") return;
  state.pointer = view.fromClient(event.clientX, event.clientY);
});
arena.addEventListener("pointerleave", () => {
  state.pointer = null;
});
arena.addEventListener("pointerdown", (event) => {
  if (state.mode !== "teleop") return;
  arena.setPointerCapture(event.pointerId);
  state.lifted = true;
  state.pointer = view.fromClient(event.clientX, event.clientY);
  if (!state.running && !state.finished) {
    state.running = true;
    syncUI();
  }
});

for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
  arena.addEventListener(name, (event) => {
    state.lifted = false;
    if (arena.hasPointerCapture?.(event.pointerId)) arena.releasePointerCapture(event.pointerId);
  });
}

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.recording) {
    state.running = false;
    finishRecording(false);
    setStatus("Recording discarded.");
  }
});

window.addEventListener("resize", () => view.resize());

state.episodeStart = world.snapshot();
syncUI();
requestAnimationFrame(animate);
