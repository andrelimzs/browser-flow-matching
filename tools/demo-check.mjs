// End-to-end check of the recording path: collect episodes exactly the way the
// page's "Collect" button does, then confirm the dataset is well formed.
globalThis.window = { localStorage: new Map([["getItem", null]]) };
globalThis.window.localStorage = {
  store: new Map(),
  getItem(key) { return this.store.get(key) ?? null; },
  setItem(key, value) { this.store.set(key, value); },
};

const { PushWorld, createRandom, SUCCESS_COVERAGE } = await import("../src/pusht/sim.js");
const { ScriptedExpert } = await import("../src/pusht/expert.js");
const { DemoStore } = await import("../src/pusht/demos.js");

const random = createRandom(4242);
const world = new PushWorld({ random, obstacleCount: 1 });
const expert = new ScriptedExpert({ random });
const store = new DemoStore();

const EPISODE_CAP = 2200;
for (let episode = 0; episode < 12; episode++) {
  world.reset({ obstacleCount: 1 });
  expert.reset();
  store.begin(world, "scripted");
  for (let step = 0; step < EPISODE_CAP; step++) {
    if (world.coverage() >= SUCCESS_COVERAGE) break;
    const observation = world.writeObservation();
    const [x, y, lift] = expert.act(world);
    store.record(observation, x, y);
    world.step(x, y, lift);
  }
  store.end(world, { keep: true });
}

const stats = store.stats();
console.log("episodes", stats.total, "steps", stats.steps, "successRate", (stats.successRate * 100).toFixed(0) + "%");

const dataset = store.toDataset();
console.log("observationSize", dataset.observationSize, "transitions", dataset.count);
console.log("observations length", dataset.observations.length, "expected", dataset.count * dataset.observationSize);
console.log("actions length", dataset.actions.length, "expected", dataset.count * 2);
console.log("boundaries", dataset.boundaries.length, "= episodes + 1 ->", dataset.boundaries.length === stats.total + 1);

const finite = dataset.observations.every(Number.isFinite) && dataset.actions.every(Number.isFinite);
console.log("all finite:", finite);

let inRange = true;
for (let i = 0; i < dataset.actions.length; i++) if (dataset.actions[i] < -0.1 || dataset.actions[i] > 1.1) inRange = false;
console.log("actions within arena bounds:", inRange);

// Persistence round trip.
store.persist();
const restored = new DemoStore().restore();
console.log("restored episodes:", restored.episodes.length, "matches:", restored.episodes.length === stats.total);

// Episodes recorded at different obstacle counts have different observation
// widths. They must never be run together into one matrix: that either
// overruns the buffer or pads short rows with zeros, shifting every field.
const mixed = new DemoStore();
for (const [obstacleCount, steps] of [[1, 20], [3, 20], [1, 20]]) {
  const w = new PushWorld({ random, obstacleCount });
  const e = new ScriptedExpert({ random });
  e.reset();
  mixed.begin(w, "scripted");
  for (let i = 0; i < steps; i++) {
    const observation = w.writeObservation();
    const [x, y, lift] = e.act(w);
    mixed.record(observation, x, y);
    w.step(x, y);
  }
  mixed.end(w, { keep: true, minimumSteps: 1 });
}
const mixedSet = mixed.toDataset();
console.log("\nmixed-width set:", JSON.stringify(mixedSet.widths), "->",
  `kept ${mixedSet.episodes} episodes at width ${mixedSet.observationSize}, excluded ${mixedSet.excluded}`);
console.log("  buffer matches kept rows:", mixedSet.observations.length === mixedSet.count * mixedSet.observationSize);
let padded = 0;
for (let row = 0; row < mixedSet.count; row++) {
  const slice = mixedSet.observations.subarray(row * mixedSet.observationSize, (row + 1) * mixedSet.observationSize);
  if (slice.every((v) => v === 0)) padded += 1;
}
console.log("  all-zero rows (0 = no padding corruption):", padded);

// Malformed stored data must not take the page down on first render.
window.localStorage.setItem("pusht-demos-v1", JSON.stringify({ version: 1, episodes: [{ source: "mouse" }, null, 7] }));
const salvaged = new DemoStore().restore();
console.log("  malformed storage dropped:", salvaged.episodes.length === 0, "and stats() survives:",
  (() => { try { salvaged.stats(); return true; } catch { return false; } })());

// JSON payload size, since this is what a user downloads.
const bytes = JSON.stringify({ version: 1, episodes: store.episodes }).length;
console.log(`export size ${(bytes / 1024).toFixed(0)} KB for ${stats.steps} transitions (${(bytes / stats.steps).toFixed(0)} B/transition)`);
