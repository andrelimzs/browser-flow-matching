// Grid path planning for the block's centroid.
//
// Local avoidance is not enough once obstacles sit deliberately between the
// start and the goal: the tee wedges against an obstacle, the straight line to
// the goal points back through it, and the expert oscillates. Planning a route
// first removes that failure entirely.
//
// The tee is approximated by a disk while planning. That is conservative — the
// real shape can slip through gaps the disk cannot — so when no route exists at
// full width the search retries at reduced width rather than giving up.

import { WALL_THICKNESS } from "./sim.js";

const GRID = 56;
const SQRT2 = Math.SQRT2;

class MinHeap {
  constructor() {
    this.items = [];
  }

  get size() {
    return this.items.length;
  }

  push(node, priority) {
    const items = this.items;
    items.push({ node, priority });
    let index = items.length - 1;
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (items[parent].priority <= priority) break;
      items[index] = items[parent];
      index = parent;
    }
    items[index] = { node, priority };
  }

  pop() {
    const items = this.items;
    if (items.length === 0) return null;
    const root = items[0];
    const tail = items.pop();
    if (items.length > 0) {
      let index = 0;
      for (;;) {
        const left = index * 2 + 1;
        if (left >= items.length) break;
        const right = left + 1;
        const child = right < items.length && items[right].priority < items[left].priority ? right : left;
        if (items[child].priority >= tail.priority) break;
        items[index] = items[child];
        index = child;
      }
      items[index] = tail;
    }
    return root.node;
  }
}

function toCell(value) {
  return Math.max(0, Math.min(GRID - 1, Math.round(value * (GRID - 1))));
}

function toWorldCoordinate(cell) {
  return cell / (GRID - 1);
}

function buildBlocked(obstacles, radius, extra) {
  const blocked = new Uint8Array(GRID * GRID);
  const low = WALL_THICKNESS + radius;
  const high = 1 - WALL_THICKNESS - radius;
  for (let row = 0; row < GRID; row++) {
    const y = toWorldCoordinate(row);
    for (let column = 0; column < GRID; column++) {
      const x = toWorldCoordinate(column);
      let hit = x < low || x > high || y < low || y > high;
      if (!hit) {
        for (const obstacle of obstacles) {
          if (Math.hypot(x - obstacle.x, y - obstacle.y) < obstacle.r + radius) {
            hit = true;
            break;
          }
        }
      }
      if (!hit && extra) hit = extra(x, y);
      if (hit) blocked[row * GRID + column] = 1;
    }
  }
  return blocked;
}

function search(blocked, startIndex, goalIndex) {
  const came = new Int32Array(GRID * GRID).fill(-1);
  const cost = new Float32Array(GRID * GRID).fill(Infinity);
  const open = new MinHeap();
  const goalRow = Math.floor(goalIndex / GRID);
  const goalColumn = goalIndex % GRID;

  cost[startIndex] = 0;
  open.push(startIndex, 0);

  while (open.size > 0) {
    const current = open.pop();
    if (current === goalIndex) break;
    const row = Math.floor(current / GRID);
    const column = current % GRID;
    const base = cost[current];

    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if (dr === 0 && dc === 0) continue;
        const nextRow = row + dr;
        const nextColumn = column + dc;
        if (nextRow < 0 || nextRow >= GRID || nextColumn < 0 || nextColumn >= GRID) continue;
        const next = nextRow * GRID + nextColumn;
        if (blocked[next]) continue;
        // No cutting diagonally between two blocked cells.
        if (dr !== 0 && dc !== 0 && (blocked[row * GRID + nextColumn] || blocked[nextRow * GRID + column])) continue;
        const step = dr !== 0 && dc !== 0 ? SQRT2 : 1;
        const candidate = base + step;
        if (candidate >= cost[next]) continue;
        cost[next] = candidate;
        came[next] = current;
        const heuristic = Math.hypot(nextRow - goalRow, nextColumn - goalColumn);
        open.push(next, candidate + heuristic);
      }
    }
  }

  if (cost[goalIndex] === Infinity) return null;
  const path = [];
  for (let node = goalIndex; node !== -1; node = came[node]) {
    path.push([toWorldCoordinate(node % GRID), toWorldCoordinate(Math.floor(node / GRID))]);
  }
  return path.reverse();
}

// Straightens the grid path by dropping waypoints that can be skipped without
// clipping an obstacle, so the expert aims at real corners rather than at the
// staircase the grid produces.
function shortcut(path, obstacles, radius, extra) {
  if (path.length < 3) return path;
  const clear = (from, to) => {
    const steps = Math.max(2, Math.ceil(Math.hypot(to[0] - from[0], to[1] - from[1]) * GRID));
    const low = WALL_THICKNESS + radius;
    const high = 1 - WALL_THICKNESS - radius;
    for (let index = 0; index <= steps; index++) {
      const amount = index / steps;
      const x = from[0] + (to[0] - from[0]) * amount;
      const y = from[1] + (to[1] - from[1]) * amount;
      if (x < low || x > high || y < low || y > high) return false;
      for (const obstacle of obstacles) {
        if (Math.hypot(x - obstacle.x, y - obstacle.y) < obstacle.r + radius) return false;
      }
      if (extra && extra(x, y)) return false;
    }
    return true;
  };

  const result = [path[0]];
  let anchor = 0;
  while (anchor < path.length - 1) {
    let furthest = anchor + 1;
    for (let candidate = path.length - 1; candidate > anchor; candidate--) {
      if (clear(path[anchor], path[candidate])) {
        furthest = candidate;
        break;
      }
    }
    result.push(path[furthest]);
    anchor = furthest;
  }
  return result;
}

// Returns a list of world-space waypoints from `from` to `to`, or null.
// `radius` is the clearance the block needs; it is relaxed on failure.
// `extra` optionally blocks additional cells, which is how a caller pins the
// route to one side of an obstacle: block a thin ray out from the obstacle
// centre and the only surviving paths are the ones passing the other way.
export function planPath(from, to, obstacles, radius, extra) {
  for (const scale of [1, 0.75, 0.55, 0.4]) {
    const effective = radius * scale;
    const blocked = buildBlocked(obstacles, effective, extra);
    const startIndex = toCell(from[1]) * GRID + toCell(from[0]);
    const goalIndex = toCell(to[1]) * GRID + toCell(to[0]);
    // The block may already be overlapping inflated geometry; let it escape.
    blocked[startIndex] = 0;
    if (blocked[goalIndex]) continue;
    const path = search(blocked, startIndex, goalIndex);
    if (path) return shortcut(path, obstacles, effective, extra);
  }
  return null;
}

export function pathLength(path) {
  let total = 0;
  for (let index = 1; index < path.length; index++) {
    total += Math.hypot(path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1]);
  }
  return total;
}

// Route for the pusher itself. The block is an obstacle here — the point of the
// manoeuvre is to get to the far side of it without shoving it on the way — so
// the caller supplies a predicate for "the pusher would be touching the block
// at this point". Both endpoints are forced open: the pusher legitimately ends
// up resting against the block.
export function planPusherPath(from, to, obstacles, touchesBlock, radius) {
  const blocked = buildBlocked(obstacles, radius, touchesBlock);
  const startIndex = toCell(from[1]) * GRID + toCell(from[0]);
  const goalIndex = toCell(to[1]) * GRID + toCell(to[0]);
  blocked[startIndex] = 0;
  blocked[goalIndex] = 0;
  const path = search(blocked, startIndex, goalIndex);
  if (!path) return null;
  const trimmed = shortcut(path, obstacles, radius, touchesBlock);
  // Snap the final waypoint onto the exact contact pose rather than the cell
  // centre the grid landed on.
  trimmed[trimmed.length - 1] = [to[0], to[1]];
  return trimmed;
}
