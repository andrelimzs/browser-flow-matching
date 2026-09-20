import "./style.css";
import { PushWorld, createRandom, SUCCESS_COVERAGE } from "./sim.js";
import { ScriptedExpert } from "./expert.js";
import { DemoStore } from "./demos.js";
import { createView } from "./render.js";

const EPISODE_CAP = 2200;
const TRAIL_LENGTH = 220;

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
  trail: [],
  lastAction: null,
  finished: null,
  episodeStart: null,
};

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
    // yanked by a stray pointer event.
    return state.pointer ?? [world.pusher.x, world.pusher.y];
  }
  return expert.act(world);
}

function advance() {
  if (state.finished) return;
  const [actionX, actionY] = nextAction();
  if (state.recording) store.record(world.writeObservation(), actionX, actionY);
  world.step(actionX, actionY);
  state.lastAction = [actionX, actionY];
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
      const [actionX, actionY] = collector.act(world);
      store.record(observation, actionX, actionY);
      world.step(actionX, actionY);
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

function setStatus(message) {
  $("#statusNote").textContent = message;
}

function syncUI() {
  const coverage = world.coverage();
  const stats = store.stats();

  $("#coverageMetric").textContent = `${(coverage * 100).toFixed(1)}%`;
  $("#coverageLabel").textContent = `coverage ${(coverage * 100).toFixed(0)}%`;
  $("#stepsMetric").textContent = String(world.steps);
  $("#stateMetric").textContent = state.mode === "teleop" ? "teleop" : expert.describe();
  $("#demoMetric").textContent = `${stats.total}`;
  $("#successMetric").textContent = stats.total ? `${(stats.successRate * 100).toFixed(0)}%` : "—";
  $("#transitionMetric").textContent = stats.steps.toLocaleString();

  $("#modeHeading").textContent = state.mode === "teleop" ? "Mouse teleoperation" : "Scripted expert";
  $("#runButton").textContent = state.running ? "Pause" : "Run";
  $("#recordButton").textContent = state.recording ? "Stop" : "Record";
  $("#recordButton").dataset.active = String(state.recording);
  $("#collectButton").disabled = state.recording || state.mode === "teleop";
  $("#clearButton").disabled = state.recording;
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
    showAction: state.mode === "expert",
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
  state.mode = mode;
  state.running = false;
  $("#modeExpert").setAttribute("aria-pressed", String(mode === "expert"));
  $("#modeTeleop").setAttribute("aria-pressed", String(mode === "teleop"));
  resetEpisode({ newWorld: true });
}

$("#modeExpert").addEventListener("click", () => setMode("expert"));
$("#modeTeleop").addEventListener("click", () => setMode("teleop"));

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
  state.pointer = view.fromClient(event.clientX, event.clientY);
  if (!state.running && !state.finished) {
    state.running = true;
    syncUI();
  }
});

for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
  arena.addEventListener(name, (event) => {
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
