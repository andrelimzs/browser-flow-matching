// Demonstration recording, shared by the scripted expert and mouse teleoperation.
//
// Episodes are stored as flat arrays so the whole set converts to JSON without
// a schema step, and so a training loop can concatenate them into one typed
// array without re-walking object graphs.

const STORAGE_KEY = "pusht-demos-v1";
// Version 7 is the no-lift, zero-degree task with no orient-first phase. Earlier
// no-lift demonstrations still contain rotate-translate-rotate trajectories.
const VERSION = 7;

// World coordinates live in [0, 1] and the pusher moves 0.017 per step, so four
// decimals is far finer than the simulation can distinguish. Full float
// precision triples the JSON size for no benefit, and localStorage quota is the
// binding constraint on how many demonstrations a session can keep.
const PRECISION = 1e4;
const round = (value) => Math.round(value * PRECISION) / PRECISION;

function isWellFormed(episode) {
  if (!episode || typeof episode !== "object") return false;
  if (!Array.isArray(episode.observations) || !Array.isArray(episode.actions)) return false;
  if (episode.observations.length !== episode.actions.length) return false;
  return episode.observations.every((row) => Array.isArray(row)) &&
    episode.actions.every((row) => Array.isArray(row) && row.length === 3);
}

export class DemoStore {
  constructor() {
    this.episodes = [];
    this.current = null;
  }

  begin(world, source) {
    this.current = {
      source,
      // The observation width is 11 + 3 per obstacle, so episodes recorded at
      // different obstacle counts are not the same shape. Stamping it here lets
      // toDataset() group them instead of running them together.
      observationSize: world.observationSize(),
      obstacleCount: world.obstacles.length,
      initial: world.snapshot(),
      observations: [],
      actions: [],
      success: false,
      coverage: 0,
    };
    return this.current;
  }

  record(observation, actionX, actionY, lift = 0) {
    if (!this.current) return;
    this.current.observations.push(Array.from(observation, round));
    this.current.actions.push([round(actionX), round(actionY), lift > 0.5 ? 1 : 0]);
  }

  // Episodes shorter than a couple of pushes are almost always a misclick.
  end(world, { keep = true, minimumSteps = 12 } = {}) {
    if (!this.current) return null;
    const episode = this.current;
    this.current = null;
    episode.coverage = world.coverage();
    episode.success = world.succeeded();
    if (!keep || episode.observations.length < minimumSteps) return null;
    this.episodes.push(episode);
    this.persist();
    return episode;
  }

  discard() {
    this.current = null;
  }

  clear() {
    this.episodes = [];
    this.current = null;
    this.persist();
  }

  stats() {
    const counts = { scripted: 0, mouse: 0 };
    let steps = 0;
    let successes = 0;
    for (const episode of this.episodes) {
      counts[episode.source] = (counts[episode.source] ?? 0) + 1;
      steps += episode.observations.length;
      if (episode.success) successes += 1;
    }
    return {
      total: this.episodes.length,
      scripted: counts.scripted,
      mouse: counts.mouse,
      steps,
      successes,
      successRate: this.episodes.length ? successes / this.episodes.length : 0,
    };
  }

  // How many episodes exist at each observation width.
  widths() {
    const counts = new Map();
    for (const episode of this.episodes) {
      const size = episode.observationSize ?? episode.observations[0]?.length ?? 0;
      counts.set(size, (counts.get(size) ?? 0) + 1);
    }
    return counts;
  }

  // Flattens to the layout a training loop wants: one observation matrix, one
  // action matrix, plus episode boundaries for action chunking.
  //
  // Only episodes of a single observation width can share a matrix. Mixing them
  // either overruns the buffer or, worse, pads the short rows with zeros so the
  // goal and obstacle fields silently land in the wrong columns. So one width is
  // selected — `observationSize`, else the most common — and the rest are
  // excluded and reported rather than quietly mangled.
  toDataset({ observationSize } = {}) {
    if (!this.episodes.length) return null;
    const widths = this.widths();
    let width = observationSize;
    if (width === undefined) {
      let best = 0;
      for (const [size, count] of widths) {
        if (count > best) {
          best = count;
          width = size;
        }
      }
    }

    const sizeOf = (episode) => episode.observationSize ?? episode.observations[0]?.length ?? 0;
    const included = this.episodes.filter((episode) => sizeOf(episode) === width);
    const excluded = this.episodes.length - included.length;
    if (!included.length) return null;

    const steps = included.reduce((total, episode) => total + episode.observations.length, 0);
    const observations = new Float32Array(steps * width);
    const actions = new Float32Array(steps * 2);
    const boundaries = [];
    let cursor = 0;
    for (const episode of included) {
      boundaries.push(cursor);
      for (let index = 0; index < episode.observations.length; index++) {
        const row = episode.observations[index];
        if (row.length !== width) continue;
        observations.set(row, (cursor + index) * width);
        actions[(cursor + index) * 2] = episode.actions[index][0];
        actions[(cursor + index) * 2 + 1] = episode.actions[index][1];
      }
      cursor += episode.observations.length;
    }
    boundaries.push(cursor);
    return {
      observations,
      actions,
      boundaries,
      observationSize: width,
      count: cursor,
      episodes: included.length,
      excluded,
      widths: Object.fromEntries(widths),
    };
  }

  // Returns null on success, or a reason the set could not be saved. Callers
  // surface that: silently dropping a demonstration someone just recorded by
  // hand is the one failure this page must not hide.
  persist() {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: VERSION, episodes: this.episodes }));
      this.storageWarning = null;
      return null;
    } catch (error) {
      const quota = error instanceof DOMException &&
        (error.name === "QuotaExceededError" || error.name === "NS_ERROR_DOM_QUOTA_REACHED");
      this.storageWarning = quota
        ? "Browser storage is full — demos are kept in memory only. Export to JSON before reloading."
        : "Browser storage is unavailable — demos are kept in memory only.";
      return this.storageWarning;
    }
  }

  // Stored data is validated, not trusted: a single malformed episode would
  // otherwise throw out of stats() during the page's first render, before any
  // handler is attached, leaving no way to recover but clearing site data.
  restore() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return this;
      const parsed = JSON.parse(raw);
      if (parsed?.version !== VERSION || !Array.isArray(parsed.episodes)) return this;
      this.episodes = parsed.episodes.filter(isWellFormed);
    } catch {
      this.episodes = [];
    }
    return this;
  }

  download(filename = "pusht-demos.json") {
    const payload = JSON.stringify({ version: VERSION, episodes: this.episodes });
    const blob = new Blob([payload], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }
}
