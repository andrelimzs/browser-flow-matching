import "./style.css";

const INPUT_WIDTH = 20;
const WIDTH = 64;
const BATCH = 96;
const LR = 0.002;
const PARTICLE_COUNT = 180;
const GRID_SIZE = 44;
const PATH_SAMPLES = 96;
const START = [0.16, 0.16];
const GOAL = [0.84, 0.84];
const REGION_RADIUS = 0.055;
const ROBOT_RADIUS = 0.014;
const CLEARANCE = 0.065;
const PATH_WIDTH = 0.006;
const PLAN_CLEARANCE = CLEARANCE + PATH_WIDTH;
const BARRIER_INFLUENCE = 0.12;
const BARRIER_STRENGTH = 0.7;
const WALL_THICKNESS = 0.04;

const $ = (selector) => document.querySelector(selector);
const flowCanvas = $("#flowCanvas");
const lossCanvas = $("#lossCanvas");
const flowCtx = flowCanvas.getContext("2d");
const lossCtx = lossCanvas.getContext("2d");

let seed = (Date.now() ^ 0x9e3779b9) >>> 0;
function random() {
  seed += 0x6d2b79f5;
  let value = seed;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
}

function sampleDisk(radius) {
  const angle = random() * Math.PI * 2;
  const radiusSample = Math.sqrt(random()) * radius;
  return [Math.cos(angle) * radiusSample, Math.sin(angle) * radiusSample];
}

function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function sourceProgressScore(point) {
  const startDistance = distance(point, START);
  const goalDistance = distance(point, GOAL);
  return startDistance / Math.max(1e-6, startDistance + goalDistance);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function pointSegmentDistance(point, start, end) {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const denominator = dx * dx + dy * dy;
  if (denominator === 0) return distance(point, start);
  const amount = clamp(((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator, 0, 1);
  return Math.hypot(point[0] - (start[0] + dx * amount), point[1] - (start[1] + dy * amount));
}

class MinHeap {
  constructor() { this.items = []; }

  push(node, priority) {
    const item = { node, priority };
    this.items.push(item);
    let index = this.items.length - 1;
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (this.items[parent].priority <= priority) break;
      this.items[index] = this.items[parent];
      index = parent;
    }
    this.items[index] = item;
  }

  pop() {
    if (this.items.length === 0) return null;
    const root = this.items[0];
    const tail = this.items.pop();
    if (this.items.length > 0) {
      let index = 0;
      while (true) {
        const left = index * 2 + 1;
        if (left >= this.items.length) break;
        const right = left + 1;
        const child = right < this.items.length && this.items[right].priority < this.items[left].priority ? right : left;
        if (this.items[child].priority >= tail.priority) break;
        this.items[index] = this.items[child];
        index = child;
      }
      this.items[index] = tail;
    }
    return root;
  }
}

function safetySlackAt(x, y, obstacles, padding = CLEARANCE) {
  const boundary = WALL_THICKNESS + ROBOT_RADIUS + padding;
  let slack = Math.min(x - boundary, 1 - boundary - x, y - boundary, 1 - boundary - y);
  for (const obstacle of obstacles) {
    slack = Math.min(slack, Math.hypot(x - obstacle.x, y - obstacle.y) - obstacle.r - ROBOT_RADIUS - padding);
  }
  return slack;
}

function clearanceBarrierAt(x, y, obstacles) {
  const slack = safetySlackAt(x, y, obstacles, PLAN_CLEARANCE);
  if (slack <= 0) return Infinity;
  if (slack >= BARRIER_INFLUENCE) return 0;
  return BARRIER_STRENGTH * -Math.log(slack / BARRIER_INFLUENCE);
}

function blockedAt(x, y, obstacles, padding = PLAN_CLEARANCE) {
  return safetySlackAt(x, y, obstacles, padding) <= 0;
}

function segmentIsClear(start, end, obstacles, padding = PLAN_CLEARANCE) {
  const boundary = WALL_THICKNESS + ROBOT_RADIUS + padding;
  if (start[0] < boundary || start[0] > 1 - boundary || start[1] < boundary || start[1] > 1 - boundary) return false;
  if (end[0] < boundary || end[0] > 1 - boundary || end[1] < boundary || end[1] > 1 - boundary) return false;
  return obstacles.every((obstacle) => pointSegmentDistance([obstacle.x, obstacle.y], start, end) > obstacle.r + ROBOT_RADIUS + padding);
}

function segmentTraversalCost(start, end, obstacles) {
  const length = distance(start, end);
  const samples = Math.max(2, Math.ceil(length * GRID_SIZE * 2));
  let multiplier = 0;
  for (let index = 0; index <= samples; index++) {
    const amount = index / samples;
    const x = start[0] * (1 - amount) + end[0] * amount;
    const y = start[1] * (1 - amount) + end[1] * amount;
    const barrier = clearanceBarrierAt(x, y, obstacles);
    if (!Number.isFinite(barrier)) return Infinity;
    multiplier += 1 + barrier;
  }
  return length * multiplier / (samples + 1);
}

function pathTraversalCost(path, obstacles, start = 0, end = path.length - 1) {
  let cost = 0;
  for (let index = start + 1; index <= end; index++) {
    cost += segmentTraversalCost(path[index - 1], path[index], obstacles);
  }
  return cost;
}

function pathIsClear(path, obstacles, padding = PLAN_CLEARANCE) {
  for (let index = 1; index < path.length; index++) {
    if (!segmentIsClear(path[index - 1], path[index], obstacles, padding)) return false;
  }
  return true;
}

function planGridPath(obstacles) {
  const size = GRID_SIZE;
  const total = size * size;
  const toIndex = (x, y) => y * size + x;
  const toPoint = (x, y) => [x / (size - 1), y / (size - 1)];
  const startX = Math.round(START[0] * (size - 1));
  const startY = Math.round(START[1] * (size - 1));
  const goalX = Math.round(GOAL[0] * (size - 1));
  const goalY = Math.round(GOAL[1] * (size - 1));
  const startIndex = toIndex(startX, startY);
  const goalIndex = toIndex(goalX, goalY);
  const blocked = new Uint8Array(total);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const point = toPoint(x, y);
      blocked[toIndex(x, y)] = blockedAt(point[0], point[1], obstacles) ? 1 : 0;
    }
  }
  blocked[startIndex] = 0;
  blocked[goalIndex] = 0;

  const costs = new Float64Array(total);
  costs.fill(Infinity);
  costs[startIndex] = 0;
  const previous = new Int32Array(total);
  previous.fill(-1);
  const closed = new Uint8Array(total);
  const open = new MinHeap();
  open.push(startIndex, distance(START, GOAL));
  const neighbors = [
    [-1, 0, 1], [1, 0, 1], [0, -1, 1], [0, 1, 1],
    [-1, -1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [1, 1, Math.SQRT2],
  ];

  while (open.items.length > 0) {
    const current = open.pop().node;
    if (closed[current]) continue;
    if (current === goalIndex) break;
    closed[current] = 1;
    const currentX = current % size;
    const currentY = Math.floor(current / size);
    const currentPoint = toPoint(currentX, currentY);
    for (const [dx, dy, moveCost] of neighbors) {
      const nextX = currentX + dx;
      const nextY = currentY + dy;
      if (nextX < 0 || nextX >= size || nextY < 0 || nextY >= size) continue;
      const next = toIndex(nextX, nextY);
      if (blocked[next] || closed[next]) continue;
      if (dx !== 0 && dy !== 0 && (blocked[toIndex(currentX + dx, currentY)] || blocked[toIndex(currentX, currentY + dy)])) continue;
      const point = toPoint(nextX, nextY);
      if (!segmentIsClear(currentPoint, point, obstacles)) continue;
      const barrier = clearanceBarrierAt(point[0], point[1], obstacles);
      if (!Number.isFinite(barrier)) continue;
      const nextCost = costs[current] + moveCost * (1 + barrier);
      if (nextCost >= costs[next]) continue;
      costs[next] = nextCost;
      previous[next] = current;
      open.push(next, nextCost + Math.hypot(goalX - nextX, goalY - nextY));
    }
  }

  if (previous[goalIndex] === -1) return null;
  const path = [];
  let current = goalIndex;
  while (current !== -1) {
    path.push(toPoint(current % size, Math.floor(current / size)));
    if (current === startIndex) break;
    current = previous[current];
  }
  path.reverse();
  path[0] = [...START];
  path[path.length - 1] = [...GOAL];
  return path;
}

function shortcutPath(path, obstacles) {
  const result = [path[0]];
  let anchor = 0;
  while (anchor < path.length - 1) {
    let next = anchor + 1;
    for (let candidate = path.length - 1; candidate > anchor + 1; candidate--) {
      const directCost = segmentTraversalCost(path[anchor], path[candidate], obstacles);
      const replacedCost = pathTraversalCost(path, obstacles, anchor, candidate);
      if (segmentIsClear(path[anchor], path[candidate], obstacles) && Number.isFinite(directCost) && directCost <= replacedCost * 1.02) {
        next = candidate;
        break;
      }
    }
    result.push(path[next]);
    anchor = next;
  }
  return result;
}

function chaikinPath(path, iterations) {
  let result = path;
  for (let iteration = 0; iteration < iterations; iteration++) {
    const refined = [result[0]];
    for (let i = 0; i < result.length - 1; i++) {
      const start = result[i];
      const end = result[i + 1];
      refined.push(
        [start[0] * 0.75 + end[0] * 0.25, start[1] * 0.75 + end[1] * 0.25],
        [start[0] * 0.25 + end[0] * 0.75, start[1] * 0.25 + end[1] * 0.75],
      );
    }
    refined.push(result[result.length - 1]);
    result = refined;
  }
  return result;
}

function smoothSafePath(path, obstacles) {
  const originalCost = pathTraversalCost(path, obstacles);
  for (let iterations = 3; iterations >= 1; iterations--) {
    const candidate = chaikinPath(path, iterations);
    let safe = true;
    for (let i = 1; i < candidate.length; i++) {
      if (!segmentIsClear(candidate[i - 1], candidate[i], obstacles)) {
        safe = false;
        break;
      }
    }
    if (safe && pathTraversalCost(candidate, obstacles) <= originalCost * 1.04) return candidate;
  }
  return path;
}

function resamplePath(path, count) {
  const cumulative = [0];
  for (let i = 1; i < path.length; i++) cumulative.push(cumulative[i - 1] + distance(path[i - 1], path[i]));
  const totalLength = cumulative[cumulative.length - 1];
  const samples = [];
  let segment = 1;
  for (let i = 0; i < count; i++) {
    const target = (i / (count - 1)) * totalLength;
    while (segment < cumulative.length - 1 && cumulative[segment] < target) segment++;
    const startLength = cumulative[segment - 1];
    const endLength = cumulative[segment];
    const amount = endLength === startLength ? 0 : (target - startLength) / (endLength - startLength);
    samples.push([
      path[segment - 1][0] * (1 - amount) + path[segment][0] * amount,
      path[segment - 1][1] * (1 - amount) + path[segment][1] * amount,
    ]);
  }
  return { samples, length: totalLength };
}

function nearestPathProgress(point) {
  let bestDistance = Infinity;
  let bestProgress = 0;
  for (let i = 1; i < referencePath.length; i++) {
    const start = referencePath[i - 1];
    const end = referencePath[i];
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const denominator = dx * dx + dy * dy;
    const amount = denominator === 0
      ? 0
      : clamp(((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator, 0, 1);
    const projected = [start[0] + dx * amount, start[1] + dy * amount];
    const candidateDistance = distance(point, projected);
    if (candidateDistance < bestDistance) {
      bestDistance = candidateDistance;
      bestProgress = (i - 1 + amount) / (referencePath.length - 1);
    }
  }
  return { distance: bestDistance, progress: bestProgress };
}

function makeRandomObstacles(count) {
  const density = (count - 2) / 8;
  const minimumRadius = 0.042 - density * 0.026;
  const maximumRadius = 0.06 - density * 0.035;
  const minimumGap = 0.055 - density * 0.03;
  let best = [];
  for (let restart = 0; restart < 20; restart++) {
    const obstacles = [];
    let currentPath = planGridPath(obstacles);
    while (obstacles.length < count && currentPath) {
      const pathSamples = resamplePath(currentPath, 128).samples;
      const candidates = [];
      for (let trial = 0; trial < 24; trial++) {
        const progress = 0.12 + ((trial + random()) / 24) * 0.76;
        const pathIndex = Math.min(pathSamples.length - 1, Math.floor(progress * (pathSamples.length - 1)));
        const point = pathSamples[pathIndex];
        const obstacle = {
          x: point[0],
          y: point[1],
          r: minimumRadius + random() * (maximumRadius - minimumRadius),
        };
        const edge = WALL_THICKNESS + ROBOT_RADIUS + obstacle.r + 0.004;
        if (obstacle.x < edge || obstacle.x > 1 - edge || obstacle.y < edge || obstacle.y > 1 - edge) continue;
        if (distance([obstacle.x, obstacle.y], START) < obstacle.r + CLEARANCE + REGION_RADIUS * 0.75) continue;
        if (distance([obstacle.x, obstacle.y], GOAL) < obstacle.r + CLEARANCE + REGION_RADIUS * 0.75) continue;
        let separation = obstacles.length === 0
          ? Math.min(distance([obstacle.x, obstacle.y], START), distance([obstacle.x, obstacle.y], GOAL))
          : Infinity;
        let spaced = true;
        for (const other of obstacles) {
          const gap = distance([obstacle.x, obstacle.y], [other.x, other.y]) - obstacle.r - other.r;
          if (gap < minimumGap) {
            spaced = false;
            break;
          }
          separation = Math.min(separation, gap);
        }
        if (!spaced) continue;
        const nextPath = planGridPath([...obstacles, obstacle]);
        if (!nextPath) continue;
        const nextLength = resamplePath(nextPath, 64).length;
        candidates.push({ obstacle, nextPath, score: separation * 0.25 - nextLength + random() * 0.001 });
      }
      candidates.sort((a, b) => b.score - a.score);
      const choice = candidates[0];
      if (!choice) break;
      obstacles.push(choice.obstacle);
      currentPath = choice.nextPath;
    }
    if (obstacles.length > best.length) best = obstacles;
    if (obstacles.length === count) return obstacles;
  }
  return best;
}

function generateWorld(count) {
  for (let attempt = 0; attempt < 8; attempt++) {
    const obstacles = makeRandomObstacles(count);
    if (obstacles.length < count) continue;
    const gridPath = planGridPath(obstacles);
    if (!gridPath) continue;
    const shortened = shortcutPath(gridPath, obstacles);
    const smoothed = smoothSafePath(shortened, obstacles);
    const route = resamplePath(smoothed, PATH_SAMPLES);
    if (!pathIsClear(route.samples, obstacles)) continue;
    return { obstacles, path: route.samples, length: route.length, waypoints: shortened.length };
  }
  const obstacles = makeRandomObstacles(count);
  const gridPath = planGridPath(obstacles);
  const shortened = shortcutPath(gridPath, obstacles);
  const route = resamplePath(smoothSafePath(shortened, obstacles), PATH_SAMPLES);
  return { obstacles, path: route.samples, length: route.length, waypoints: shortened.length };
}

function referenceState(time) {
  const scaled = clamp(time, 0, 1) * (referencePath.length - 1);
  const index = Math.min(referencePath.length - 2, Math.floor(scaled));
  const amount = scaled - index;
  const start = referencePath[index];
  const end = referencePath[index + 1];
  return [start[0] * (1 - amount) + end[0] * amount, start[1] * (1 - amount) + end[1] * amount];
}

function sampleUniformFree() {
  const minimum = WALL_THICKNESS + ROBOT_RADIUS + 0.008;
  for (let attempt = 0; attempt < 200; attempt++) {
    const point = [minimum + random() * (1 - minimum * 2), minimum + random() * (1 - minimum * 2)];
    if (!blockedAt(point[0], point[1], obstacles, 0)) return point;
  }
  return [...START];
}

function samplePathPoint(progress) {
  const point = referenceState(progress);
  const before = referenceState(Math.max(0, progress - 0.01));
  const after = referenceState(Math.min(1, progress + 0.01));
  const tangentX = after[0] - before[0];
  const tangentY = after[1] - before[1];
  const tangentLength = Math.max(1e-6, Math.hypot(tangentX, tangentY));
  for (let attempt = 0; attempt < 8; attempt++) {
    const offset = (random() * 2 - 1) * PATH_WIDTH;
    const candidate = [point[0] - (tangentY / tangentLength) * offset, point[1] + (tangentX / tangentLength) * offset];
    if (!blockedAt(candidate[0], candidate[1], obstacles, CLEARANCE)) return candidate;
  }
  return point;
}

function xavier(size, fanIn, fanOut) {
  const values = new Float32Array(size);
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  for (let i = 0; i < size; i++) values[i] = (random() * 2 - 1) * limit;
  return values;
}

class TinyMLP {
  constructor() {
    this.w1 = xavier(WIDTH * INPUT_WIDTH, INPUT_WIDTH, WIDTH);
    this.b1 = new Float32Array(WIDTH);
    this.w2 = xavier(WIDTH * WIDTH, WIDTH, WIDTH);
    this.b2 = new Float32Array(WIDTH);
    this.w3 = xavier(2 * WIDTH, WIDTH, 2);
    this.b3 = new Float32Array(2);
    this.params = [this.w1, this.b1, this.w2, this.b2, this.w3, this.b3];
    this.grads = this.params.map((parameter) => new Float32Array(parameter.length));
    this.moments = this.params.map((parameter) => new Float32Array(parameter.length));
    this.velocities = this.params.map((parameter) => new Float32Array(parameter.length));
    this.step = 0;
    this.input = new Float32Array(INPUT_WIDTH);
    this.h1 = new Float32Array(WIDTH);
    this.h2 = new Float32Array(WIDTH);
    this.dh1 = new Float32Array(WIDTH);
    this.dh2 = new Float32Array(WIDTH);
  }

  forward(x, y, flowTime, pathProgress) {
    const input = this.input;
    const normalizedX = x * 2 - 1;
    const normalizedY = y * 2 - 1;
    input[0] = normalizedX;
    input[1] = normalizedY;
    input[2] = flowTime;
    input[3] = pathProgress;
    input[4] = Math.sin(Math.PI * 2 * flowTime);
    input[5] = Math.cos(Math.PI * 2 * flowTime);
    input[6] = Math.sin(Math.PI * 2 * pathProgress);
    input[7] = Math.cos(Math.PI * 2 * pathProgress);
    input[8] = Math.sin(Math.PI * normalizedX);
    input[9] = Math.cos(Math.PI * normalizedX);
    input[10] = Math.sin(Math.PI * normalizedY);
    input[11] = Math.cos(Math.PI * normalizedY);
    input[12] = Math.sin(Math.PI * 2 * normalizedX);
    input[13] = Math.cos(Math.PI * 2 * normalizedX);
    input[14] = Math.sin(Math.PI * 2 * normalizedY);
    input[15] = Math.cos(Math.PI * 2 * normalizedY);
    input[16] = Math.sin(Math.PI * 4 * normalizedX);
    input[17] = Math.cos(Math.PI * 4 * normalizedX);
    input[18] = Math.sin(Math.PI * 4 * normalizedY);
    input[19] = Math.cos(Math.PI * 4 * normalizedY);
    for (let j = 0; j < WIDTH; j++) {
      const offset = j * INPUT_WIDTH;
      let sum = this.b1[j];
      for (let i = 0; i < INPUT_WIDTH; i++) sum += this.w1[offset + i] * input[i];
      this.h1[j] = Math.tanh(sum);
    }
    for (let j = 0; j < WIDTH; j++) {
      const offset = j * WIDTH;
      let sum = this.b2[j];
      for (let i = 0; i < WIDTH; i++) sum += this.w2[offset + i] * this.h1[i];
      this.h2[j] = Math.tanh(sum);
    }
    let outputX = this.b3[0];
    let outputY = this.b3[1];
    for (let i = 0; i < WIDTH; i++) {
      outputX += this.w3[i] * this.h2[i];
      outputY += this.w3[WIDTH + i] * this.h2[i];
    }
    return [outputX, outputY];
  }

  trainBatch() {
    for (const gradient of this.grads) gradient.fill(0);
    const sources = Array.from({ length: BATCH }, sampleUniformFree);
    if (usePathProgress) sources.sort((a, b) => sourceProgressScore(a) - sourceProgressScore(b));
    const progresses = new Float32Array(BATCH);
    const targets = Array.from({ length: BATCH }, (_, index) => {
      progresses[index] = usePathProgress ? (index + random()) / BATCH : random();
      return samplePathPoint(progresses[index]);
    });
    let loss = 0;
    for (let sample = 0; sample < BATCH; sample++) {
      const source = sources[sample];
      const target = targets[sample];
      const progress = progresses[sample];
      const timeSample = random() < 0.5 ? random() : 1 - random() ** 2;
      const time = 0.01 + timeSample * 0.98;
      const x = source[0] * (1 - time) + target[0] * time;
      const y = source[1] * (1 - time) + target[1] * time;
      const targetX = target[0] - source[0];
      const targetY = target[1] - source[1];
      const output = this.forward(x, y, time, usePathProgress ? progress : 0);
      const errorX = output[0] - targetX;
      const errorY = output[1] - targetY;
      const deltaX = errorX / BATCH;
      const deltaY = errorY / BATCH;
      loss += 0.5 * (errorX * errorX + errorY * errorY);

      for (let j = 0; j < WIDTH; j++) {
        this.grads[4][j] += deltaX * this.h2[j];
        this.grads[4][WIDTH + j] += deltaY * this.h2[j];
        this.dh2[j] = (this.w3[j] * deltaX + this.w3[WIDTH + j] * deltaY) * (1 - this.h2[j] ** 2);
      }
      this.grads[5][0] += deltaX;
      this.grads[5][1] += deltaY;

      this.dh1.fill(0);
      for (let j = 0; j < WIDTH; j++) {
        const offset = j * WIDTH;
        const delta = this.dh2[j];
        this.grads[3][j] += delta;
        for (let i = 0; i < WIDTH; i++) {
          this.grads[2][offset + i] += delta * this.h1[i];
          this.dh1[i] += this.w2[offset + i] * delta;
        }
      }

      for (let j = 0; j < WIDTH; j++) {
        const delta = this.dh1[j] * (1 - this.h1[j] ** 2);
        const offset = j * INPUT_WIDTH;
        this.grads[1][j] += delta;
        for (let i = 0; i < INPUT_WIDTH; i++) this.grads[0][offset + i] += delta * this.input[i];
      }
    }

    this.step++;
    const firstCorrection = 1 - 0.9 ** this.step;
    const secondCorrection = 1 - 0.999 ** this.step;
    for (let parameterIndex = 0; parameterIndex < this.params.length; parameterIndex++) {
      const values = this.params[parameterIndex];
      const moments = this.moments[parameterIndex];
      const velocities = this.velocities[parameterIndex];
      const gradients = this.grads[parameterIndex];
      for (let i = 0; i < values.length; i++) {
        moments[i] = 0.9 * moments[i] + 0.1 * gradients[i];
        velocities[i] = 0.999 * velocities[i] + 0.001 * gradients[i] * gradients[i];
        values[i] -= LR * (moments[i] / firstCorrection) / (Math.sqrt(velocities[i] / secondCorrection) + 1e-8);
      }
    }
    return loss / BATCH;
  }
}

let obstacleCount = 4;
let world = generateWorld(obstacleCount);
let obstacles = world.obstacles;
let referencePath = world.path;
let usePathProgress = true;
let model = new TinyMLP();
let training = true;
let flowPlaying = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let inferenceSteps = 10;
let simTime = 0;
let playbackTime = 0;
let lastInferenceCheckpoint = 0;
let holdUntil = 0;
let particles = [];
let lossHistory = [];
let lossRecordEvery = 6;
let smoothedLoss = null;
let lastFrame = performance.now();
let telemetryAt = performance.now();
let telemetrySteps = 0;
let measuredSpeed = 0;
let latestPathRate = null;
let latestCoverageRate = null;

function resetParticles() {
  particles = Array.from({ length: PARTICLE_COUNT }, () => {
    const [x, y] = sampleUniformFree();
    return { x, y, x0: x, y0: y, progress: 0, tail: [], path: [] };
  });
  if (usePathProgress) {
    particles.sort((a, b) => sourceProgressScore([a.x0, a.y0]) - sourceProgressScore([b.x0, b.y0]));
    particles.forEach((particle, index) => {
      particle.progress = (index + 0.5) / particles.length;
    });
  }
  buildParticlePaths();
  simTime = 0;
  playbackTime = 0;
  lastInferenceCheckpoint = 0;
  holdUntil = 0;
  syncTimeUI();
}

function buildParticlePaths() {
  const dt = 1 / inferenceSteps;
  for (const particle of particles) {
    let x = particle.x0;
    let y = particle.y0;
    particle.path = [[x, y]];
    for (let step = 0; step < inferenceSteps; step++) {
      const [velocityX, velocityY] = model.forward(x, y, step * dt, usePathProgress ? particle.progress : 0);
      x += velocityX * dt;
      y += velocityY * dt;
      particle.path.push([x, y]);
    }
    particle.tail = [];
  }
}

function setParticlesTo(time, keepTails = false, rounding = "nearest") {
  const scaledTime = clamp(time * inferenceSteps, 0, inferenceSteps);
  const checkpoint = rounding === "floor" ? Math.floor(scaledTime) : Math.round(scaledTime);
  const checkpointChanged = checkpoint !== lastInferenceCheckpoint;
  const traceCheckpoint = inferenceSteps === 1 ? checkpoint : keepTails && model.step > 50 ? checkpoint : 0;
  for (const particle of particles) {
    if (traceCheckpoint > 0 && (checkpointChanged || !keepTails || particle.tail.length === 0)) {
      particle.tail = particle.path.slice(Math.max(0, traceCheckpoint - 10), traceCheckpoint);
    } else if (traceCheckpoint === 0) {
      particle.tail = [];
    }
    particle.x = particle.path[checkpoint][0];
    particle.y = particle.path[checkpoint][1];
  }
  lastInferenceCheckpoint = checkpoint;
  simTime = checkpoint / inferenceSteps;
  syncTimeUI();
}

function syncTimeUI() {
  $("#timeSlider").value = simTime;
  $("#timeOutput").value = simTime.toFixed(2);
  const phase = simTime < 0.04 ? "Uniform source" : simTime > 0.96 ? "Full trajectory" : "Matching path";
  $("#phaseLabel").textContent = phase;
}

function resizeCanvas(canvas, context) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return rect;
}

function drawFlow() {
  const rect = resizeCanvas(flowCanvas, flowCtx);
  const width = rect.width;
  const height = rect.height;
  const pad = Math.max(22, Math.min(width, height) * 0.055);
  const fieldSize = Math.min(width - pad * 2, height - pad * 2);
  const offsetX = (width - fieldSize) / 2;
  const offsetY = (height - fieldSize) / 2;
  const toX = (x) => offsetX + x * fieldSize;
  const toY = (y) => offsetY + (1 - y) * fieldSize;
  flowCtx.clearRect(0, 0, width, height);

  const gradient = flowCtx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#f8f7f2");
  gradient.addColorStop(1, "#e6e4dd");
  flowCtx.fillStyle = gradient;
  flowCtx.fillRect(0, 0, width, height);

  flowCtx.strokeStyle = "rgba(25, 28, 27, .055)";
  flowCtx.lineWidth = 1;
  for (let i = 0; i <= 10; i++) {
    const x = offsetX + (i / 10) * fieldSize;
    const y = offsetY + (i / 10) * fieldSize;
    flowCtx.beginPath(); flowCtx.moveTo(x, offsetY); flowCtx.lineTo(x, offsetY + fieldSize); flowCtx.stroke();
    flowCtx.beginPath(); flowCtx.moveTo(offsetX, y); flowCtx.lineTo(offsetX + fieldSize, y); flowCtx.stroke();
  }

  const wallPixels = WALL_THICKNESS * fieldSize;
  flowCtx.fillStyle = "#202420";
  flowCtx.fillRect(offsetX, offsetY, fieldSize, wallPixels);
  flowCtx.fillRect(offsetX, offsetY + fieldSize - wallPixels, fieldSize, wallPixels);
  flowCtx.fillRect(offsetX, offsetY, wallPixels, fieldSize);
  flowCtx.fillRect(offsetX + fieldSize - wallPixels, offsetY, wallPixels, fieldSize);

  for (const obstacle of obstacles) {
    flowCtx.beginPath();
    flowCtx.arc(toX(obstacle.x), toY(obstacle.y), (obstacle.r + ROBOT_RADIUS + CLEARANCE) * fieldSize, 0, Math.PI * 2);
    flowCtx.fillStyle = "rgba(75, 80, 77, .06)";
    flowCtx.fill();
    flowCtx.setLineDash([4, 5]);
    flowCtx.strokeStyle = "rgba(75, 80, 77, .42)";
    flowCtx.stroke();
    flowCtx.setLineDash([]);
    flowCtx.beginPath();
    flowCtx.arc(toX(obstacle.x), toY(obstacle.y), obstacle.r * fieldSize, 0, Math.PI * 2);
    flowCtx.fillStyle = "#2c312e";
    flowCtx.fill();
    flowCtx.strokeStyle = "rgba(255,255,255,.28)";
    flowCtx.stroke();
  }

  const drawRegion = (center, fill, stroke, label) => {
    flowCtx.beginPath();
    flowCtx.arc(toX(center[0]), toY(center[1]), REGION_RADIUS * fieldSize, 0, Math.PI * 2);
    flowCtx.fillStyle = fill;
    flowCtx.fill();
    flowCtx.strokeStyle = stroke;
    flowCtx.lineWidth = 1.5;
    flowCtx.stroke();
    flowCtx.fillStyle = stroke;
    flowCtx.font = "9px DM Mono, monospace";
    flowCtx.textAlign = "center";
    flowCtx.fillText(label, toX(center[0]), toY(center[1]) - REGION_RADIUS * fieldSize - 8);
  };
  drawRegion(START, "rgba(29,102,219,.12)", "#1d66db", "START");
  drawRegion(GOAL, "rgba(112,160,42,.16)", "#638c2d", "GOAL");

  flowCtx.beginPath();
  referencePath.forEach((point, index) => {
    if (index === 0) flowCtx.moveTo(toX(point[0]), toY(point[1]));
    else flowCtx.lineTo(toX(point[0]), toY(point[1]));
  });
  flowCtx.setLineDash([6, 5]);
  flowCtx.strokeStyle = "rgba(20, 23, 22, .88)";
  flowCtx.lineWidth = 2.2;
  flowCtx.stroke();
  flowCtx.setLineDash([]);

  for (const particle of particles) {
    if (particle.tail.length > 0) {
      flowCtx.beginPath();
      particle.tail.forEach((point, index) => {
        if (index === 0) flowCtx.moveTo(toX(point[0]), toY(point[1]));
        else flowCtx.lineTo(toX(point[0]), toY(point[1]));
      });
      flowCtx.lineTo(toX(particle.x), toY(particle.y));
      flowCtx.strokeStyle = "rgba(237, 107, 85, .18)";
      flowCtx.lineWidth = 1;
      flowCtx.stroke();
    }
    flowCtx.beginPath();
    flowCtx.arc(toX(particle.x), toY(particle.y), width < 520 ? 2.1 : 2.45, 0, Math.PI * 2);
    flowCtx.fillStyle = "rgba(226, 79, 58, .8)";
    flowCtx.fill();
  }

  flowCtx.fillStyle = "rgba(25, 28, 27, .44)";
  flowCtx.font = "10px DM Mono, monospace";
  flowCtx.textAlign = "left";
  flowCtx.fillText(`LEARNED PATH DISTRIBUTION · STEP ${model.step.toLocaleString()}`, 15, height - 16);
}

function drawLoss() {
  const rect = resizeCanvas(lossCanvas, lossCtx);
  const width = rect.width;
  const height = rect.height;
  lossCtx.clearRect(0, 0, width, height);
  lossCtx.strokeStyle = "rgba(25, 28, 27, .07)";
  lossCtx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (height * i) / 4;
    lossCtx.beginPath(); lossCtx.moveTo(0, y); lossCtx.lineTo(width, y); lossCtx.stroke();
  }
  const plotTop = 6;
  const plotBottom = height - 18;
  if (lossHistory.length < 2) return;
  const logs = lossHistory.map((point) => Math.log(Math.max(point.loss, 1e-5)));
  let min = Math.min(...logs);
  let max = Math.max(...logs);
  if (max - min < 0.25) { max += 0.125; min -= 0.125; }
  lossCtx.beginPath();
  lossHistory.forEach((point, index) => {
    const x = (point.step / Math.max(1, model.step)) * width;
    const y = plotBottom - ((logs[index] - min) / (max - min)) * (plotBottom - plotTop);
    if (index === 0) lossCtx.moveTo(x, y); else lossCtx.lineTo(x, y);
  });
  lossCtx.strokeStyle = "#1d66db";
  lossCtx.lineWidth = 1.8;
  lossCtx.stroke();
  lossCtx.fillStyle = "rgba(25, 28, 27, .48)";
  lossCtx.font = "9px DM Mono, monospace";
  lossCtx.textAlign = "left";
  lossCtx.fillText("0", 0, height - 3);
  lossCtx.textAlign = "right";
  lossCtx.fillText(model.step.toLocaleString(), width, height - 3);
}

function recordLoss(loss) {
  if (model.step === 1) {
    lossHistory = [{ step: 0, loss }];
    return;
  }
  if (model.step % lossRecordEvery !== 0) return;
  lossHistory.push({ step: model.step, loss });
  if (lossHistory.length <= 1200) return;
  const compacted = [lossHistory[0]];
  for (let i = 1; i < lossHistory.length; i += 2) {
    const first = lossHistory[i];
    const second = lossHistory[i + 1];
    compacted.push(second ? { step: second.step, loss: (first.loss + second.loss) * 0.5 } : first);
  }
  lossHistory = compacted;
  lossRecordEvery *= 2;
}

function updateTelemetry() {
  $("#stepMetric").textContent = model.step.toLocaleString();
  $("#lossMetric").textContent = smoothedLoss === null ? "—" : smoothedLoss.toFixed(4);
  $("#speedMetric").textContent = measuredSpeed ? Math.round(measuredSpeed) : "—";
  $("#hitMetric").textContent = latestPathRate === null
    ? "—"
    : `${Math.round(latestPathRate * 100)}% ${usePathProgress ? "at s" : "on path"} · ${Math.round(latestCoverageRate * 100)}% covered`;
  $("#obstacleSummary").textContent = String(obstacles.length);
  $("#routeSummary").textContent = world.length.toFixed(2);
  if (lossHistory.length > 12) {
    const old = lossHistory[Math.max(0, lossHistory.length - 12)].loss;
    const newest = lossHistory[lossHistory.length - 1].loss;
    const change = ((newest - old) / old) * 100;
    $("#lossTrend").textContent = change < -1 ? `${Math.abs(change).toFixed(0)}% ↓ recent` : "Stabilizing";
  }
}

function evaluateRollouts() {
  const occupiedBins = new Set();
  let onPath = 0;
  const binCount = 20;
  for (const particle of particles) {
    const endpoint = particle.path[particle.path.length - 1];
    const match = usePathProgress
      ? { distance: distance(endpoint, referenceState(particle.progress)), progress: particle.progress }
      : nearestPathProgress(endpoint);
    if (match.distance <= 0.035) {
      onPath++;
      occupiedBins.add(Math.min(binCount - 1, Math.floor(match.progress * binCount)));
    }
  }
  latestPathRate = onPath / particles.length;
  latestCoverageRate = occupiedBins.size / binCount;
}

function trainingStatusText() {
  return usePathProgress ? "Learning ordered positions along the path" : "Matching uniform samples to the path";
}

function animate(now) {
  const elapsed = Math.min(50, now - lastFrame);
  lastFrame = now;
  if (training) {
    const deadline = performance.now() + 11;
    let trained = 0;
    while (trained < 3 && performance.now() < deadline) {
      const loss = model.trainBatch();
      smoothedLoss = smoothedLoss === null ? loss : smoothedLoss * 0.96 + loss * 0.04;
      recordLoss(smoothedLoss);
      trained++;
      telemetrySteps++;
    }
  }

  if (flowPlaying) {
    if (simTime >= 1) {
      if (!holdUntil) {
        evaluateRollouts();
        holdUntil = now + 1200;
        updateTelemetry();
      } else if (now >= holdUntil) {
        resetParticles();
      }
    } else {
      playbackTime = inferenceSteps === 1 ? 1 : Math.min(1, playbackTime + elapsed / (inferenceSteps * 64));
      setParticlesTo(playbackTime, true, "floor");
    }
  }

  if (now - telemetryAt > 500) {
    measuredSpeed = (telemetrySteps * 1000) / (now - telemetryAt);
    telemetrySteps = 0;
    telemetryAt = now;
    updateTelemetry();
    drawLoss();
  }
  drawFlow();
  requestAnimationFrame(animate);
}

function resetModelForWorld() {
  model = new TinyMLP();
  lossHistory = [];
  lossRecordEvery = 6;
  smoothedLoss = null;
  latestPathRate = null;
  latestCoverageRate = null;
  training = true;
  flowPlaying = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  $("#trainingToggle").innerHTML = '<span aria-hidden="true">Ⅱ</span> Pause';
  $("#trainingStatus").textContent = trainingStatusText();
  $("#lossTrend").textContent = "Collecting data";
  $("#flowToggle").innerHTML = flowPlaying ? '<span aria-hidden="true">Ⅱ</span>' : '<span aria-hidden="true">▶</span>';
  $("#flowToggle").setAttribute("aria-label", flowPlaying ? "Pause flow animation" : "Play flow animation");
  resetParticles();
  updateTelemetry();
  drawLoss();
}

$("#trainingToggle").addEventListener("click", () => {
  training = !training;
  $("#trainingToggle").innerHTML = training ? '<span aria-hidden="true">Ⅱ</span> Pause' : '<span aria-hidden="true">▶</span> Train';
  $("#trainingStatus").textContent = training ? trainingStatusText() : "Flow weights are frozen";
});

$("#progressToggle").addEventListener("click", () => {
  usePathProgress = !usePathProgress;
  $("#progressToggle").textContent = usePathProgress ? "On" : "Off";
  $("#progressToggle").setAttribute("aria-pressed", String(usePathProgress));
  $("#hitMetricLabel").textContent = usePathProgress
    ? "Progress accuracy / trajectory coverage"
    : "Samples on path / trajectory coverage";
  $("#conditioningSummary").textContent = usePathProgress ? "Flow τ · progress s" : "Flow τ only";
  $("#couplingSummary").textContent = usePathProgress ? "Ranked by proximity" : "Independent";
  resetModelForWorld();
});

$("#flowToggle").addEventListener("click", () => {
  flowPlaying = !flowPlaying;
  if (flowPlaying) playbackTime = simTime;
  $("#flowToggle").innerHTML = flowPlaying ? '<span aria-hidden="true">Ⅱ</span>' : '<span aria-hidden="true">▶</span>';
  $("#flowToggle").setAttribute("aria-label", flowPlaying ? "Pause flow animation" : "Play flow animation");
});

$("#timeSlider").addEventListener("input", (event) => {
  flowPlaying = false;
  $("#flowToggle").innerHTML = '<span aria-hidden="true">▶</span>';
  $("#flowToggle").setAttribute("aria-label", "Play flow animation");
  playbackTime = Number(event.target.value);
  setParticlesTo(playbackTime);
  playbackTime = simTime;
  holdUntil = 0;
});

$("#timeSlider").addEventListener("pointerdown", () => {
  buildParticlePaths();
  setParticlesTo(simTime);
});

document.querySelectorAll("[data-inference-steps]").forEach((button) => {
  button.addEventListener("click", () => {
    const currentTime = simTime;
    inferenceSteps = Number(button.dataset.inferenceSteps);
    $("#timeSlider").step = String(1 / inferenceSteps);
    document.querySelectorAll("[data-inference-steps]").forEach((option) => {
      option.setAttribute("aria-pressed", String(option === button));
    });
    $("#inferenceSummary").textContent = `${inferenceSteps} Euler ${inferenceSteps === 1 ? "step" : "steps"}`;
    latestPathRate = null;
    latestCoverageRate = null;
    buildParticlePaths();
    lastInferenceCheckpoint = -1;
    playbackTime = currentTime;
    setParticlesTo(playbackTime);
    playbackTime = simTime;
    holdUntil = 0;
    updateTelemetry();
  });
});

$("#resampleButton").addEventListener("click", () => {
  seed = (Date.now() ^ 0x85ebca6b) >>> 0;
  world = generateWorld(obstacleCount);
  obstacles = world.obstacles;
  referencePath = world.path;
  resetModelForWorld();
});

$("#obstacleSlider").addEventListener("input", (event) => {
  $("#obstacleOutput").value = event.target.value;
});

$("#obstacleSlider").addEventListener("change", (event) => {
  obstacleCount = Number(event.target.value);
  seed = (Date.now() ^ obstacleCount ^ 0x27d4eb2d) >>> 0;
  world = generateWorld(obstacleCount);
  obstacles = world.obstacles;
  referencePath = world.path;
  resetModelForWorld();
});

$("#resetButton").addEventListener("click", () => {
  seed = (Date.now() ^ 0xc2b2ae35) >>> 0;
  resetModelForWorld();
});

window.addEventListener("resize", () => { drawFlow(); drawLoss(); });

resetParticles();
updateTelemetry();
if (!flowPlaying) {
  $("#flowToggle").innerHTML = '<span aria-hidden="true">▶</span>';
  $("#flowToggle").setAttribute("aria-label", "Play flow animation");
}
requestAnimationFrame(animate);
