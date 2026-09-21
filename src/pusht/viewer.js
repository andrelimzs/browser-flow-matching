// Replays a recorded demonstration next to the learned policy rolled out from
// the identical starting pose, so the step where they part company is visible.
//
// Both the demonstrations and the trained policy come from the browser storage
// the main page writes, so this is a view onto that data rather than a second
// copy of the training machinery.

import "./style.css";
import { PushWorld, createRandom, SUCCESS_COVERAGE } from "./sim.js";
import { DemoStore } from "./demos.js";
import { loadPolicy } from "./policy-cache.js";
import { makePolicy, sampleChunk, ACTION_DIM } from "./policy.js";
import { MLP } from "../flow/mlp.js";
import { createView } from "./render.js";

const EXECUTE = 16;
const LIFT_ON = 0.4;
const LIFT_OFF = -0.4;
const POLICY_CAP = 600;
const DIVERGENCE = 0.05;

const $ = (selector) => document.querySelector(selector);

const random = createRandom(Date.now() >>> 0);
const world = new PushWorld({ random, obstacleCount: 0 });
const view = createView($("#arena"));
const store = new DemoStore().restore();

const state = {
  episode: null,
  index: -1,
  expert: null,
  policy: null,
  cursor: 0,
  playing: false,
  showTrajectoryOverview: false,
};
let policy = null;
let policyScales = null;
let policyObstacles = 0;

function setStatus(message) {
  $("#statusNote").textContent = message;
}

function loadCachedPolicy() {
  const cached = loadPolicy();
  if (!cached) return false;
  const built = makePolicy({ observationSize: cached.observationSize, width: 128, maxBatch: 1, random });
  built.model = new MLP({ sizes: cached.sizes, maxBatch: 1, random }).loadSnapshot(cached.weights);
  policy = built;
  policyScales = cached.scales;
  policyObstacles = cached.obstacleCount;
  return true;
}

// The recorded poses are already in the stored observations: pusher xy, block
// xy, block cos/sin. No replay needed.
function expertTrace(episode) {
  const count = episode.observations.length;
  const pusher = new Float32Array(count * 2);
  const block = [];
  for (let index = 0; index < count; index++) {
    const observation = episode.observations[index];
    pusher[index * 2] = observation[0];
    pusher[index * 2 + 1] = observation[1];
    block.push({ x: observation[2], y: observation[3], angle: Math.atan2(observation[5], observation[4]) });
  }
  return { pusher, block, steps: count, coverage: episode.coverage ?? null, success: episode.success };
}

function policyTrace(episode) {
  if (!policy) return null;
  world.restore(episode.initial);
  if (world.obstacles.length !== policyObstacles) return { mismatch: true };

  const pusher = [];
  const block = [];
  let chunk = null;
  let cursor = Infinity;
  let lift = 0;
  for (let step = 0; step < POLICY_CAP; step++) {
    if (world.coverage() >= SUCCESS_COVERAGE) break;
    if (cursor >= EXECUTE) {
      chunk = sampleChunk(policy, world.writeObservation(), { steps: 10, random, scales: policyScales });
      cursor = 0;
    }
    const base = cursor * ACTION_DIM;
    const raw = chunk[base + 2];
    if (raw > LIFT_ON) lift = 1;
    else if (raw < LIFT_OFF) lift = 0;
    cursor += 1;
    world.step(chunk[base], chunk[base + 1], lift);
    pusher.push(world.pusher.x, world.pusher.y);
    block.push({ x: world.block.x, y: world.block.y, angle: world.block.angle });
  }
  return {
    pusher: Float32Array.from(pusher),
    block,
    steps: block.length,
    coverage: world.coverage(),
    success: world.coverage() >= SUCCESS_COVERAGE,
  };
}

// First step at which the two pusher paths are meaningfully apart.
function divergence(a, b) {
  const shared = Math.min(a.steps, b.steps);
  for (let index = 0; index < shared; index++) {
    const dx = a.pusher[index * 2] - b.pusher[index * 2];
    const dy = a.pusher[index * 2 + 1] - b.pusher[index * 2 + 1];
    if (Math.hypot(dx, dy) > DIVERGENCE) return index;
  }
  return shared === 0 ? 0 : null;
}

function selectEpisode(index) {
  const episode = store.episodes[index];
  if (!episode?.initial) {
    setStatus("That episode has no stored start pose — record a fresh one on the main page.");
    return;
  }
  state.episode = episode;
  state.index = index;
  state.expert = expertTrace(episode);
  state.policy = policyTrace(episode);
  state.cursor = 0;
  state.playing = false;
  state.showTrajectoryOverview = true;

  if (state.policy?.mismatch) {
    setStatus(`This episode has ${episode.obstacleCount ?? "?"} obstacles but the cached policy was trained with ${policyObstacles}. Train a policy at a matching count to compare.`);
    state.policy = null;
  } else if (!policy) {
    setStatus("No cached policy found — train one on the main page and it will appear here.");
  } else {
    setStatus("Scrub to the step where the two paths separate.");
  }
  syncUI();
}

function buildList() {
  const list = $("#episodeList");
  list.innerHTML = "";
  if (!store.episodes.length) {
    list.innerHTML = '<p class="empty-note">No demonstrations stored. Collect some on the main page.</p>';
    return;
  }
  store.episodes.forEach((episode, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "episode-row";
    button.setAttribute("aria-current", String(index === state.index));
    button.innerHTML =
      `<span class="tag">${episode.source === "mouse" ? "hand" : "expert"} #${index + 1}</span>` +
      `<span>${episode.observations.length} steps</span>` +
      `<span class="${episode.success ? "ok" : "bad"}">${episode.success ? "solved" : "failed"}</span>`;
    button.addEventListener("click", () => selectEpisode(index));
    list.appendChild(button);
  });
}

function longest() {
  return Math.max(state.expert?.steps ?? 0, state.policy?.steps ?? 0);
}

function syncUI() {
  const total = longest();
  $("#scrub").max = String(Math.max(1, total - 1));
  $("#scrub").value = String(Math.min(state.cursor, Math.max(0, total - 1)));
  $("#stepOutput").textContent = String(state.cursor);
  $("#stepLabel").textContent = `step ${state.cursor} / ${Math.max(0, total - 1)}`;
  $("#playButton").textContent = state.playing ? "❚❚" : "▶";
  $("#episodeHeading").textContent =
    state.index >= 0 ? `Episode ${state.index + 1}` : store.episodes.length ? "Select an episode" : "No demonstrations";
  $("#phaseLabel").textContent = state.playing ? "Playing" : state.episode ? "Ready" : "Idle";
  $(".phase-pill").dataset.active = String(state.playing);

  $("#expertSteps").textContent = state.expert ? String(state.expert.steps) : "—";
  $("#policySteps").textContent = state.policy ? String(state.policy.steps) : "—";
  $("#expertCoverage").textContent = state.expert?.coverage != null ? `${(state.expert.coverage * 100).toFixed(0)}%` : "—";
  $("#policyCoverage").textContent = state.policy ? `${(state.policy.coverage * 100).toFixed(0)}%` : "—";
  const split = state.expert && state.policy ? divergence(state.expert, state.policy) : null;
  $("#divergeStep").textContent = state.expert && state.policy
    ? (split === null ? "never (within 0.05)" : `step ${split}`)
    : "—";
  $("#rerunButton").disabled = !state.episode || !policy;
  buildListSelection();
}

function buildListSelection() {
  document.querySelectorAll(".episode-row").forEach((row, index) => {
    row.setAttribute("aria-current", String(index === state.index));
  });
}

function poseAt(trace, cursor) {
  if (!trace || !trace.block.length) return null;
  return trace.block[Math.min(trace.block.length - 1, cursor)];
}

function frame() {
  if (state.playing) {
    const total = longest();
    state.cursor += 1;
    if (state.cursor >= total) { state.cursor = total - 1; state.playing = false; }
    syncUI();
  }

  if (state.episode) {
    // The arena is drawn at the policy's pose where there is one, with the
    // recorded pose behind it as a ghost.
    const policyPose = poseAt(state.policy, state.cursor);
    const expertPose = poseAt(state.expert, state.cursor);
    world.restore(state.episode.initial);
    if (policyPose) {
      world.block = { ...policyPose };
      world.pusher = {
        x: state.policy.pusher[Math.min(state.policy.steps - 1, state.cursor) * 2],
        y: state.policy.pusher[Math.min(state.policy.steps - 1, state.cursor) * 2 + 1],
      };
    } else if (expertPose) {
      world.block = { ...expertPose };
      world.pusher = {
        x: state.expert.pusher[Math.min(state.expert.steps - 1, state.cursor) * 2],
        y: state.expert.pusher[Math.min(state.expert.steps - 1, state.cursor) * 2 + 1],
      };
    }

    const paths = [];
    if (state.expert) {
      paths.push({
        points: state.expert.pusher,
        cursor: state.showTrajectoryOverview ? state.expert.steps : state.cursor + 1,
        color: "rgba(29, 102, 219, .85)",
        width: 1.8,
        head: !state.showTrajectoryOverview,
      });
    }
    if (state.policy) {
      paths.push({
        points: state.policy.pusher,
        cursor: state.showTrajectoryOverview ? state.policy.steps : state.cursor + 1,
        color: "rgba(237, 107, 85, .9)",
        width: 1.8,
        dashed: true,
        head: !state.showTrajectoryOverview,
      });
    }
    view.draw(world, { paths, ghost: policyPose && expertPose ? expertPose : null, coverage: world.coverage() });
  } else {
    view.draw(world, {});
  }
  requestAnimationFrame(frame);
}

$("#scrub").addEventListener("input", (event) => {
  state.cursor = Number(event.target.value);
  state.playing = false;
  syncUI();
});

$("#playButton").addEventListener("click", () => {
  if (!state.episode) return;
  if (state.cursor >= longest() - 1) state.cursor = 0;
  state.showTrajectoryOverview = false;
  state.playing = !state.playing;
  syncUI();
});

$("#rerunButton").addEventListener("click", () => {
  if (!state.episode) return;
  state.policy = policyTrace(state.episode);
  state.cursor = 0;
  state.playing = false;
  state.showTrajectoryOverview = true;
  setStatus("Policy re-sampled — a fresh draw from the same distribution.");
  syncUI();
});

window.addEventListener("resize", () => view.resize());

const hasPolicy = loadCachedPolicy();
buildList();
if (store.episodes.length) selectEpisode(store.episodes.findIndex((e) => e.initial) >= 0 ? store.episodes.findIndex((e) => e.initial) : 0);
else setStatus("No demonstrations stored in this browser. Collect some on the main page first.");
if (!hasPolicy) setStatus("No cached policy found — train one on the main page and it will appear here.");
syncUI();
requestAnimationFrame(frame);
