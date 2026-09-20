// Quasi-static planar pushing.
//
// Inertia is ignored: the block's motion is determined entirely by the wrench
// currently applied to it. Under the ellipsoidal limit-surface approximation
// (Lynch & Mason) that relationship is linear and diagonal,
//
//   velocity      proportional to  applied force
//   angular rate  proportional to  applied torque / c^2
//
// with c the geometric constant computed in geometry.js. Contacts are solved as
// projections: each one applies the wrench that would separate it, relaxed and
// iterated Gauss-Seidel style. Large timesteps stay stable because there is no
// momentum to integrate, which is exactly why this model suits a browser demo
// running at display rate.

import {
  circleRectContact,
  clamp,
  makeTeeShape,
  pointInShape,
  rectCorners,
  toLocal,
  wrapAngle,
} from "./geometry.js";

export const BLOCK_UNIT = 0.05;
export const PUSHER_RADIUS = 0.022;
export const WALL_THICKNESS = 0.028;
export const MAX_PUSHER_SPEED = 0.017;
export const SUCCESS_COVERAGE = 0.9;

const SUBSTEPS = 4;
const SOLVER_ITERATIONS = 6;
const RELAXATION = 0.5;
const MAX_SPIN_PER_ITERATION = 0.06;
const PUSHER_FRICTION = 0.6;
const OBSTACLE_FRICTION = 0.3;
const COVERAGE_SAMPLES = 640;

// Obstacles are proposed inside the corridor running from the block's start
// pose to the goal pose, stratified along it and jittered sideways. Scattering
// them uniformly over the arena mostly produced layouts where nothing was in
// the way, which is not a pushing problem worth planning around.
const CORRIDOR_NEAR = 0.25;
const CORRIDOR_FAR = 0.75;

// A near-centred obstacle is what makes the two ways around it equally good:
// at this offset the two routes differ by ~15% in length, against ~57% at the
// 0.14 this used to be. Crowded corridors still need lateral room, so the
// spread opens up with the count.
const corridorLateral = (count) => 0.04 + Math.max(0, count - 1) * 0.045;

export const TEE = makeTeeShape(BLOCK_UNIT);

const OBSTACLE_GAP = TEE.unit * 3;

// Start and goal must be far enough apart to leave room for the corridor; with
// them close together there is nowhere to put an obstacle that is not already
// touching one of the two footprints.
function minimumSeparation(count) {
  if (count === 0) return TEE.radius * 1.6;
  return 0.46 + Math.max(0, count - 2) * 0.1;
}

export function createRandom(seedValue) {
  let seed = (seedValue ^ 0x9e3779b9) >>> 0;
  return function random() {
    seed += 0x6d2b79f5;
    let value = seed;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

// Fixed stratified samples covering the tee in its local frame. Coverage is
// then a membership test of the goal footprint against the current footprint,
// which is both cheap and free of polygon-clipping edge cases.
const COVERAGE_POINTS = (() => {
  const random = createRandom(0x51ed270b);
  const points = [];
  const extent = BLOCK_UNIT * 4;
  while (points.length < COVERAGE_SAMPLES) {
    const x = (random() * 2 - 1) * extent;
    const y = (random() * 2 - 1) * extent;
    if (pointInShape(x, y, TEE.parts)) points.push([x, y]);
  }
  return points;
})();

const contact = { depth: 0, nx: 0, ny: 0, px: 0, py: 0 };
const scratchA = { x: 0, y: 0 };
const scratchB = { x: 0, y: 0 };

function insetBounds() {
  return { min: WALL_THICKNESS, max: 1 - WALL_THICKNESS };
}

export class PushWorld {
  constructor(options = {}) {
    this.random = options.random ?? createRandom(Date.now() >>> 0);
    this.obstacleCount = options.obstacleCount ?? 1;
    this.block = { x: 0.5, y: 0.5, angle: 0 };
    this.goal = { x: 0.5, y: 0.5, angle: 0 };
    this.pusher = { x: 0.5, y: 0.2 };
    this.command = { x: 0.5, y: 0.2 };
    this.obstacles = [];
    this.previous = { x: 0, y: 0, angle: 0 };
    this.steps = 0;
    this.reset();
  }

  reset(options = {}) {
    const random = this.random;
    const count = options.obstacleCount ?? this.obstacleCount;
    this.obstacleCount = count;
    const bounds = insetBounds();
    const margin = TEE.radius + 0.01;
    const separation = minimumSeparation(count);

    let best = null;
    for (let attempt = 0; attempt < 400; attempt++) {
      const goal = {
        x: bounds.min + margin + random() * (bounds.max - bounds.min - margin * 2),
        y: bounds.min + margin + random() * (bounds.max - bounds.min - margin * 2),
        angle: random() * Math.PI * 2,
      };
      const block = {
        x: bounds.min + margin + random() * (bounds.max - bounds.min - margin * 2),
        y: bounds.min + margin + random() * (bounds.max - bounds.min - margin * 2),
        angle: random() * Math.PI * 2,
      };
      if (Math.hypot(block.x - goal.x, block.y - goal.y) < separation) continue;

      const obstacles = this.placeObstacles(count, block, goal);
      const pusher = this.findPusherStart(block, obstacles);
      if (!pusher) continue;

      const layout = { goal, block, obstacles, pusher };
      if (obstacles.length === count) return this.commit(layout);
      // A crowded corridor cannot always take every obstacle. Keeping the
      // fullest layout found beats discarding it for an empty arena, which is
      // what the caller least expects when they asked for obstacles.
      if (!best || obstacles.length > best.obstacles.length) best = layout;
    }

    if (best) return this.commit(best);

    // Nothing at all was placeable: fall back to a bare, well-posed task.
    return this.commit({
      goal: { x: 0.5, y: 0.72, angle: 0 },
      block: { x: 0.5, y: 0.28, angle: Math.PI / 3 },
      obstacles: [],
      pusher: { x: 0.5, y: 0.12 },
    });
  }

  commit({ goal, block, obstacles, pusher }) {
    this.goal = goal;
    this.block = block;
    this.obstacles = obstacles;
    this.pusher = pusher;
    this.command = { ...pusher };
    this.previous = { x: block.x, y: block.y, angle: block.angle };
    this.steps = 0;
    return this;
  }

  placeObstacles(count, block, goal) {
    if (count === 0) return [];
    const random = this.random;
    const bounds = insetBounds();
    const axisX = goal.x - block.x;
    const axisY = goal.y - block.y;
    const span = Math.hypot(axisX, axisY);
    if (span < 1e-6) return [];
    const alongX = axisX / span;
    const alongY = axisY / span;
    const lateralX = -alongY;
    const lateralY = alongX;

    const obstacles = [];
    for (let index = 0; index < count; index++) {
      let placed = null;
      for (let attempt = 0; attempt < 240 && !placed; attempt++) {
        const radius = 0.035 + random() * 0.03;
        // One stratum per obstacle, so several of them spread down the corridor
        // instead of clustering at one point along it.
        const slot = (index + random()) / count;
        const along = CORRIDOR_NEAR + slot * (CORRIDOR_FAR - CORRIDOR_NEAR);
        const lateral = (random() * 2 - 1) * corridorLateral(count);
        const candidate = {
          x: block.x + alongX * span * along + lateralX * lateral,
          y: block.y + alongY * span * along + lateralY * lateral,
          r: radius,
        };
        if (candidate.x < bounds.min + radius || candidate.x > bounds.max - radius) continue;
        if (candidate.y < bounds.min + radius || candidate.y > bounds.max - radius) continue;
        // Keep both the start and goal footprints clear, with enough slack that
        // the pusher can still reach every face of the block in those poses.
        const clearance = PUSHER_RADIUS * 2 + 0.02;
        if (this.overlapsShape(candidate, block, clearance)) continue;
        if (this.overlapsShape(candidate, goal, clearance)) continue;
        const spaced = obstacles.every(
          (other) => Math.hypot(candidate.x - other.x, candidate.y - other.y) > candidate.r + other.r + OBSTACLE_GAP,
        );
        if (!spaced) continue;
        placed = candidate;
      }
      if (!placed) return obstacles;
      obstacles.push(placed);
    }
    return obstacles;
  }

  overlapsShape(circle, pose, extra = 0) {
    toLocal(pose, circle.x, circle.y, scratchA);
    for (const part of TEE.parts) {
      if (circleRectContact(scratchA.x, scratchA.y, circle.r + extra, part, contact)) return true;
    }
    return false;
  }

  findPusherStart(block, obstacles) {
    const random = this.random;
    const bounds = insetBounds();
    for (let attempt = 0; attempt < 200; attempt++) {
      const angle = random() * Math.PI * 2;
      const distance = TEE.radius + PUSHER_RADIUS + 0.01 + random() * 0.06;
      const candidate = {
        x: block.x + Math.cos(angle) * distance,
        y: block.y + Math.sin(angle) * distance,
      };
      if (candidate.x < bounds.min + PUSHER_RADIUS || candidate.x > bounds.max - PUSHER_RADIUS) continue;
      if (candidate.y < bounds.min + PUSHER_RADIUS || candidate.y > bounds.max - PUSHER_RADIUS) continue;
      if (this.pusherBlocked(candidate.x, candidate.y, obstacles)) continue;
      if (this.pusherTouchesBlock(candidate.x, candidate.y, block)) continue;
      return candidate;
    }
    return null;
  }

  pusherBlocked(x, y, obstacles = this.obstacles) {
    return obstacles.some((obstacle) => Math.hypot(x - obstacle.x, y - obstacle.y) < obstacle.r + PUSHER_RADIUS);
  }

  pusherTouchesBlock(x, y, block = this.block) {
    toLocal(block, x, y, scratchA);
    for (const part of TEE.parts) {
      if (circleRectContact(scratchA.x, scratchA.y, PUSHER_RADIUS, part, contact)) return true;
    }
    return false;
  }

  // Action space: an absolute target position for the pusher, matching the
  // convention used by the diffusion-policy Push-T benchmark. The pusher is
  // position-controlled and travels toward the target at a capped speed, so the
  // action is literally a point in the plane.
  step(targetX, targetY) {
    const bounds = insetBounds();
    this.command.x = clamp(targetX, bounds.min + PUSHER_RADIUS, bounds.max - PUSHER_RADIUS);
    this.command.y = clamp(targetY, bounds.min + PUSHER_RADIUS, bounds.max - PUSHER_RADIUS);

    this.previous.x = this.block.x;
    this.previous.y = this.block.y;
    this.previous.angle = this.block.angle;

    for (let substep = 0; substep < SUBSTEPS; substep++) {
      const beforeX = this.pusher.x;
      const beforeY = this.pusher.y;
      this.advancePusher(MAX_PUSHER_SPEED / SUBSTEPS);
      const velocityX = this.pusher.x - beforeX;
      const velocityY = this.pusher.y - beforeY;
      for (let iteration = 0; iteration < SOLVER_ITERATIONS; iteration++) {
        this.solveContacts(velocityX, velocityY);
      }
    }

    this.block.angle = wrapAngle(this.block.angle);
    this.steps += 1;
    return this;
  }

  advancePusher(maximumStep) {
    const dx = this.command.x - this.pusher.x;
    const dy = this.command.y - this.pusher.y;
    const distance = Math.hypot(dx, dy);
    if (distance > 1e-9) {
      const scale = Math.min(1, maximumStep / distance);
      this.pusher.x += dx * scale;
      this.pusher.y += dy * scale;
    }
    // The pusher is rigid and position-controlled, so static geometry simply
    // truncates its motion rather than pushing back on it.
    for (const obstacle of this.obstacles) {
      const offsetX = this.pusher.x - obstacle.x;
      const offsetY = this.pusher.y - obstacle.y;
      const length = Math.hypot(offsetX, offsetY);
      const minimum = obstacle.r + PUSHER_RADIUS;
      if (length < minimum && length > 1e-9) {
        this.pusher.x = obstacle.x + (offsetX / length) * minimum;
        this.pusher.y = obstacle.y + (offsetY / length) * minimum;
      }
    }
    const bounds = insetBounds();
    this.pusher.x = clamp(this.pusher.x, bounds.min + PUSHER_RADIUS, bounds.max - PUSHER_RADIUS);
    this.pusher.y = clamp(this.pusher.y, bounds.min + PUSHER_RADIUS, bounds.max - PUSHER_RADIUS);
  }

  solveContacts(pusherVelocityX, pusherVelocityY) {
    this.solvePusherContact(pusherVelocityX, pusherVelocityY);
    this.solveObstacleContacts();
    this.solveWallContacts();
  }

  solvePusherContact(velocityX, velocityY) {
    const block = this.block;
    toLocal(block, this.pusher.x, this.pusher.y, scratchA);
    for (const part of TEE.parts) {
      if (!circleRectContact(scratchA.x, scratchA.y, PUSHER_RADIUS, part, contact)) continue;

      const cos = Math.cos(block.angle);
      const sin = Math.sin(block.angle);
      const normalX = contact.nx * cos - contact.ny * sin;
      const normalY = contact.nx * sin + contact.ny * cos;
      const pointX = block.x + contact.px * cos - contact.py * sin;
      const pointY = block.y + contact.px * sin + contact.py * cos;

      // Normal force pushes the block away from the pusher.
      let forceX = -normalX * contact.depth;
      let forceY = -normalY * contact.depth;

      // Friction drags the block along with the pusher's tangential motion,
      // bounded by the friction cone. This is what lets a sideways sweep turn
      // the tee instead of merely sliding off it.
      const alongNormal = velocityX * normalX + velocityY * normalY;
      const tangentX = velocityX - alongNormal * normalX;
      const tangentY = velocityY - alongNormal * normalY;
      const tangentLength = Math.hypot(tangentX, tangentY);
      if (tangentLength > 1e-9) {
        const limit = Math.min(tangentLength, PUSHER_FRICTION * contact.depth);
        forceX += (tangentX / tangentLength) * limit;
        forceY += (tangentY / tangentLength) * limit;
      }

      this.applyWrench(pointX, pointY, forceX, forceY);
    }
  }

  solveObstacleContacts() {
    const block = this.block;
    for (const obstacle of this.obstacles) {
      toLocal(block, obstacle.x, obstacle.y, scratchA);
      for (const part of TEE.parts) {
        if (!circleRectContact(scratchA.x, scratchA.y, obstacle.r, part, contact)) continue;

        const cos = Math.cos(block.angle);
        const sin = Math.sin(block.angle);
        const normalX = contact.nx * cos - contact.ny * sin;
        const normalY = contact.nx * sin + contact.ny * cos;
        const pointX = block.x + contact.px * cos - contact.py * sin;
        const pointY = block.y + contact.px * sin + contact.py * cos;

        let forceX = -normalX * contact.depth;
        let forceY = -normalY * contact.depth;

        // Static friction resists the block's own sliding at the contact, which
        // is what makes the tee catch on an obstacle and pivot around it.
        const radiusX = pointX - block.x;
        const radiusY = pointY - block.y;
        const spin = wrapAngle(block.angle - this.previous.angle);
        const surfaceX = block.x - this.previous.x - spin * radiusY;
        const surfaceY = block.y - this.previous.y + spin * radiusX;
        const alongNormal = surfaceX * normalX + surfaceY * normalY;
        const tangentX = surfaceX - alongNormal * normalX;
        const tangentY = surfaceY - alongNormal * normalY;
        const tangentLength = Math.hypot(tangentX, tangentY);
        if (tangentLength > 1e-9) {
          const limit = Math.min(tangentLength, OBSTACLE_FRICTION * contact.depth);
          forceX -= (tangentX / tangentLength) * limit;
          forceY -= (tangentY / tangentLength) * limit;
        }

        this.applyWrench(pointX, pointY, forceX, forceY);
      }
    }
  }

  solveWallContacts() {
    const block = this.block;
    const bounds = insetBounds();
    const cos = Math.cos(block.angle);
    const sin = Math.sin(block.angle);
    for (const part of TEE.parts) {
      for (const [localX, localY] of rectCorners(part)) {
        const worldX = block.x + localX * cos - localY * sin;
        const worldY = block.y + localX * sin + localY * cos;
        let forceX = 0;
        let forceY = 0;
        if (worldX < bounds.min) forceX = bounds.min - worldX;
        else if (worldX > bounds.max) forceX = bounds.max - worldX;
        if (worldY < bounds.min) forceY = bounds.min - worldY;
        else if (worldY > bounds.max) forceY = bounds.max - worldY;
        if (forceX !== 0 || forceY !== 0) this.applyWrench(worldX, worldY, forceX, forceY);
      }
    }
  }

  applyWrench(pointX, pointY, forceX, forceY) {
    const block = this.block;
    const radiusX = pointX - block.x;
    const radiusY = pointY - block.y;
    const torque = radiusX * forceY - radiusY * forceX;
    block.x += RELAXATION * forceX;
    block.y += RELAXATION * forceY;
    const spin = (RELAXATION * torque) / TEE.characteristicSquared;
    block.angle += clamp(spin, -MAX_SPIN_PER_ITERATION, MAX_SPIN_PER_ITERATION);
  }

  // Fraction of the goal footprint currently covered by the block.
  coverage() {
    const block = this.block;
    const goal = this.goal;
    const goalCos = Math.cos(goal.angle);
    const goalSin = Math.sin(goal.angle);
    const blockCos = Math.cos(block.angle);
    const blockSin = Math.sin(block.angle);
    let inside = 0;
    for (const [localX, localY] of COVERAGE_POINTS) {
      const worldX = goal.x + localX * goalCos - localY * goalSin;
      const worldY = goal.y + localX * goalSin + localY * goalCos;
      const dx = worldX - block.x;
      const dy = worldY - block.y;
      if (pointInShape(dx * blockCos + dy * blockSin, -dx * blockSin + dy * blockCos, TEE.parts)) inside += 1;
    }
    return inside / COVERAGE_POINTS.length;
  }

  succeeded() {
    return this.coverage() >= SUCCESS_COVERAGE;
  }

  // The tee has two-fold symmetry about nothing, so the orientation error is a
  // plain wrapped difference rather than a modular one.
  poseError() {
    return {
      position: Math.hypot(this.goal.x - this.block.x, this.goal.y - this.block.y),
      angle: Math.abs(wrapAngle(this.goal.angle - this.block.angle)),
    };
  }

  observationSize() {
    return 10 + this.obstacles.length * 3;
  }

  writeObservation(out = new Float32Array(this.observationSize())) {
    out[0] = this.pusher.x;
    out[1] = this.pusher.y;
    out[2] = this.block.x;
    out[3] = this.block.y;
    out[4] = Math.cos(this.block.angle);
    out[5] = Math.sin(this.block.angle);
    out[6] = this.goal.x;
    out[7] = this.goal.y;
    out[8] = Math.cos(this.goal.angle);
    out[9] = Math.sin(this.goal.angle);
    let index = 10;
    for (const obstacle of this.obstacles) {
      out[index++] = obstacle.x;
      out[index++] = obstacle.y;
      out[index++] = obstacle.r;
    }
    return out;
  }

  snapshot() {
    return {
      block: { ...this.block },
      goal: { ...this.goal },
      pusher: { ...this.pusher },
      obstacles: this.obstacles.map((obstacle) => ({ ...obstacle })),
    };
  }

  restore(state) {
    this.block = { ...state.block };
    this.goal = { ...state.goal };
    this.pusher = { ...state.pusher };
    this.command = { ...state.pusher };
    this.obstacles = state.obstacles.map((obstacle) => ({ ...obstacle }));
    this.previous = { x: this.block.x, y: this.block.y, angle: this.block.angle };
    this.steps = 0;
    return this;
  }

  blockCorners() {
    const block = this.block;
    const cos = Math.cos(block.angle);
    const sin = Math.sin(block.angle);
    return TEE.parts.map((part) =>
      rectCorners(part).map(([x, y]) => [block.x + x * cos - y * sin, block.y + x * sin + y * cos]),
    );
  }

  goalCorners() {
    const goal = this.goal;
    const cos = Math.cos(goal.angle);
    const sin = Math.sin(goal.angle);
    return TEE.parts.map((part) =>
      rectCorners(part).map(([x, y]) => [goal.x + x * cos - y * sin, goal.y + x * sin + y * cos]),
    );
  }
}
