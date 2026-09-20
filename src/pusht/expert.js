// Scripted Push-T expert.
//
// The policy is a two-state machine. In PUSH it drives the pusher into a chosen
// contact on the tee; in APPROACH it relocates the pusher to the next contact
// without disturbing the block, by orbiting the block's circumscribed circle.
//
// Contact selection is where the physics enters. A push at surface point p with
// inward direction f moves the block by a velocity proportional to f and spins
// it at a rate proportional to (r x f) / c^2. That is a linear map from contact
// choice to block motion, so the expert simply scores every candidate contact
// against the motion it currently wants and takes the best one.
//
// The orbit direction is the deliberate source of multimodality: when both ways
// around the block are clear, the choice is a coin flip, and the two options
// produce completely different action sequences for identical observations.
// That is the structure a unimodal regression policy cannot represent, and it
// is the whole reason this task is worth learning with flow matching.

import { circleRectContact, clamp, toLocal, wrapAngle } from "./geometry.js";
import { pathLength, planPath } from "./plan.js";
import { MAX_PUSHER_SPEED, PUSHER_RADIUS, SUCCESS_COVERAGE, TEE, WALL_THICKNESS } from "./sim.js";

const APPROACH_GAP = 0.004;
const ARRIVAL_TOLERANCE = 0.012;
const PUSH_STROKE_FAR = 0.06;
const PUSH_STROKE_NEAR = 0.022;
const MAX_PUSH_STEPS = 14;
const MAX_APPROACH_STEPS = 140;
const ROTATION_WEIGHT = 1;
const TRAVEL_WEIGHT = 0.7;
// Effective radius of the tee when deciding whether an obstacle is in its way.
// The circumradius is over-conservative because the tee is mostly not there.
const BLOCK_SWEEP = 0.78;
// Orientation can wait until the block is near the end of its route; insisting
// on it while still in transit drives the tee back into what it is going around.
const TRANSIT_ROTATION_SCALE = 0.35;
const LOOKAHEAD = 0.16;
const REPLAN_INTERVAL = 70;
const PATH_DRIFT = 0.1;
// Two ways round an obstacle count as equally good below this cost ratio, and
// the expert then picks at random. With the obstacle near the corridor centre
// almost every layout qualifies; the gate only rejects the lopsided ones.
const ROUTE_COST_GATE = 1.25;
// Half-width of the ray used to pin a route to one side of an obstacle.
const BARRIER_HALF_WIDTH = 0.019;
const STUCK_LIMIT = 60;
const SAMPLES_PER_UNIT = 90;
// Emit short, acceleration-limited waypoints instead of discontinuously
// jumping the command between distant contact targets. The simulator already
// caps speed; bounding acceleration here also makes the demonstrations smooth.
const MAX_TARGET_ACCELERATION = 0.0035;

const probe = { depth: 0, nx: 0, ny: 0, px: 0, py: 0 };
const scratch = { x: 0, y: 0 };

// Candidate contacts live in the block's local frame, so they are computed once
// and reused for every episode.
const CANDIDATES = buildCandidates();

function buildCandidates() {
  const candidates = [];
  for (const part of TEE.parts) {
    const edges = [
      { ax: part.minX, ay: part.minY, bx: part.maxX, by: part.minY, nx: 0, ny: -1 },
      { ax: part.maxX, ay: part.minY, bx: part.maxX, by: part.maxY, nx: 1, ny: 0 },
      { ax: part.maxX, ay: part.maxY, bx: part.minX, by: part.maxY, nx: 0, ny: 1 },
      { ax: part.minX, ay: part.maxY, bx: part.minX, by: part.minY, nx: -1, ny: 0 },
    ];
    for (const edge of edges) {
      const length = Math.hypot(edge.bx - edge.ax, edge.by - edge.ay);
      const count = Math.max(2, Math.round(length * SAMPLES_PER_UNIT));
      for (let index = 0; index < count; index++) {
        const amount = (index + 0.5) / count;
        const x = edge.ax + (edge.bx - edge.ax) * amount;
        const y = edge.ay + (edge.by - edge.ay) * amount;
        const standoff = PUSHER_RADIUS + APPROACH_GAP;
        const pusherX = x + edge.nx * standoff;
        const pusherY = y + edge.ny * standoff;
        // Reject contacts tucked into the tee's inner corners, where the pusher
        // would have to overlap the other rectangle to reach them.
        let reachable = true;
        for (const other of TEE.parts) {
          if (circleRectContact(pusherX, pusherY, PUSHER_RADIUS, other, probe)) {
            reachable = false;
            break;
          }
        }
        if (!reachable) continue;
        candidates.push({ x, y, nx: edge.nx, ny: edge.ny, pusherX, pusherY });
      }
    }
  }
  return candidates;
}

export class ScriptedExpert {
  // tieBreak: "random" keeps both orbit directions in the data, "cw"/"ccw"
  // collapse the policy to a single mode for comparison.
  constructor(options = {}) {
    this.random = options.random ?? Math.random;
    this.tieBreak = options.tieBreak ?? "random";
    this.rotationWeight = options.rotationWeight ?? ROTATION_WEIGHT;
    this.travelWeight = options.travelWeight ?? TRAVEL_WEIGHT;
    // 0 disables branching entirely: always take the shorter way round.
    this.routeCostGate = options.routeCostGate ?? ROUTE_COST_GATE;
    this.reset();
  }

  reset() {
    this.routeObstacle = null;
    this.routeSide = 0;
    this.routeSides = new Map();
    this.stuck = 0;
    this.path = null;
    this.replanIn = 0;
    this.state = "select";
    this.contact = null;
    this.pushSteps = 0;
    this.approachSteps = 0;
    this.bestCoverage = 0;
    this.stalled = 0;
    this.targetVelocityX = 0;
    this.targetVelocityY = 0;
  }

  // Returns [x, y, lift] for this control step. Lifted, the pusher travels in a
  // straight line to the next contact instead of walking around the block, so
  // repositioning is no longer a navigation problem.
  act(world) {
    const coverage = world.coverage();
    if (coverage >= SUCCESS_COVERAGE) return [world.pusher.x, world.pusher.y, 0];

    this.replanIn -= 1;
    if (this.state === "select") this.selectContact(world);
    if (!this.contact) {
      // No candidate contact is reachable. In practice the contact solver jostles
      // the world enough to unstick this within a step, but nothing else
      // increments a counter here, so guard against a true livelock.
      this.stuck += 1;
      if (this.stuck > STUCK_LIMIT) {
        this.stuck = 0;
        this.path = null;
        this.routeObstacle = null;
        this.routeSide = 0;
        this.routeSides.clear();
      }
      this.targetVelocityX = 0;
      this.targetVelocityY = 0;
      return [world.pusher.x, world.pusher.y, 0];
    }
    this.stuck = 0;

    const target = this.state === "push" ? this.pushTarget(world) : this.approachTarget(world);
    this.advance(world, coverage);
    return this.smoothTarget(world, target);
  }

  smoothTarget(world, target) {
    const dx = target[0] - world.pusher.x;
    const dy = target[1] - world.pusher.y;
    const distance = Math.hypot(dx, dy);
    if (distance < 1e-9) {
      this.targetVelocityX = 0;
      this.targetVelocityY = 0;
      return [world.pusher.x, world.pusher.y, target[2]];
    }

    // Slow early enough to arrive without an abrupt final step. This is the
    // usual stopping-speed bound v <= sqrt(2ad), paired with a vector-valued
    // acceleration limit for smooth turns between contacts.
    const speed = Math.min(MAX_PUSHER_SPEED, Math.sqrt(2 * MAX_TARGET_ACCELERATION * distance));
    const desiredX = (dx / distance) * speed;
    const desiredY = (dy / distance) * speed;
    const changeX = desiredX - this.targetVelocityX;
    const changeY = desiredY - this.targetVelocityY;
    const change = Math.hypot(changeX, changeY);
    const scale = change > MAX_TARGET_ACCELERATION ? MAX_TARGET_ACCELERATION / change : 1;
    this.targetVelocityX += changeX * scale;
    this.targetVelocityY += changeY * scale;

    const step = Math.hypot(this.targetVelocityX, this.targetVelocityY);
    if (step > distance) {
      this.targetVelocityX = dx;
      this.targetVelocityY = dy;
    }
    return [
      world.pusher.x + this.targetVelocityX,
      world.pusher.y + this.targetVelocityY,
      target[2],
    ];
  }

  // Keeps a planned route for the block's centroid current. Replanning is cheap
  // relative to a push, and the block is nudged off its route constantly by the
  // contact dynamics, so a periodic replan is simpler than trying to track one.
  ensurePath(world) {
    if (this.path && this.replanIn > 0 && this.pathOffset(world) < PATH_DRIFT) return;
    this.path = this.planRoute(world);
    this.replanIn = REPLAN_INTERVAL;
  }

  // The obstacle the block has to get past next: the one nearest the straight
  // line to the goal, and still ahead of the block.
  branchObstacle(world) {
    const block = world.block;
    const dx = world.goal.x - block.x;
    const dy = world.goal.y - block.y;
    const span = Math.hypot(dx, dy);
    if (span < 1e-6) return null;
    const alongX = dx / span;
    const alongY = dy / span;
    let nearest = null;
    for (const obstacle of world.obstacles) {
      const offsetX = obstacle.x - block.x;
      const offsetY = obstacle.y - block.y;
      const along = offsetX * alongX + offsetY * alongY;
      if (along <= 0 || along > span) continue;
      const lateral = Math.abs(offsetX * alongY - offsetY * alongX);
      if (lateral > obstacle.r + TEE.radius * BLOCK_SWEEP) continue;
      if (!nearest || along < nearest.along) nearest = { obstacle, along, alongX, alongY };
    }
    return nearest;
  }

  // Blocks a thin ray running out from the obstacle centre on `side`, so any
  // route that survives must pass on the other side.
  sideBarrier(branch, side) {
    const { obstacle, alongX, alongY } = branch;
    return (x, y) => {
      const dx = x - obstacle.x;
      const dy = y - obstacle.y;
      const lateral = dx * -alongY + dy * alongX;
      return lateral * side > 0 && Math.abs(dx * alongX + dy * alongY) < BARRIER_HALF_WIDTH;
    };
  }

  // Plans the block's route, choosing which way round the next obstacle to go.
  // When both ways cost about the same the choice is a free coin flip, and it
  // is latched for as long as that obstacle is still the one in the way —
  // resampling it on every replan would make the block dither between the two
  // and produce neither mode cleanly.
  planRoute(world) {
    const block = world.block;
    const from = [block.x, block.y];
    const to = [world.goal.x, world.goal.y];
    const radius = TEE.radius * BLOCK_SWEEP;
    const branch = this.branchObstacle(world);

    // No obstacle on the straight line right now. The commitment is kept rather
    // than cleared: swinging around an obstacle routinely takes it out of this
    // test for a step or two, and clearing here made the next replan flip a
    // fresh coin — the dither the latch exists to prevent.
    if (!branch) return planPath(from, to, world.obstacles, radius);

    const planSide = (side) => planPath(from, to, world.obstacles, radius, this.sideBarrier(branch, -side));

    const commit = (side, path) => {
      this.routeSides.set(branch.obstacle, side);
      this.routeObstacle = branch.obstacle;
      this.routeSide = side;
      return path;
    };

    // Commitments are held per obstacle, not just for the current one. With
    // more than one obstacle the branch target alternates between them as the
    // block moves, and a single slot loses the other's decision every time it
    // switches — which is a fresh coin flip on the way back.
    const remembered = this.routeSides.get(branch.obstacle);
    if (remembered) {
      const held = planSide(remembered);
      if (held) return commit(remembered, held);
    }

    const left = planSide(1);
    const right = planSide(-1);

    if (left && right) {
      const ratio = Math.max(pathLength(left), pathLength(right)) / Math.max(1e-6, Math.min(pathLength(left), pathLength(right)));
      if (ratio <= this.routeCostGate) {
        const side = this.tieBreak === "cw" ? -1 : this.tieBreak === "ccw" ? 1 : this.random() < 0.5 ? 1 : -1;
        return commit(side, side > 0 ? left : right);
      }
      const side = pathLength(left) <= pathLength(right) ? 1 : -1;
      return commit(side, side > 0 ? left : right);
    }
    if (left) return commit(1, left);
    if (right) return commit(-1, right);

    this.routeObstacle = null;
    this.routeSide = 0;
    return planPath(from, to, world.obstacles, radius);
  }

  // Distance from the block to its route, used to notice it has been shoved off.
  pathOffset(world) {
    if (!this.path || this.path.length < 2) return Infinity;
    const block = world.block;
    let nearest = Infinity;
    for (let index = 1; index < this.path.length; index++) {
      const [ax, ay] = this.path[index - 1];
      const [bx, by] = this.path[index];
      const dx = bx - ax;
      const dy = by - ay;
      const lengthSquared = dx * dx + dy * dy;
      const amount = lengthSquared < 1e-12 ? 0 : clamp(((block.x - ax) * dx + (block.y - ay) * dy) / lengthSquared, 0, 1);
      nearest = Math.min(nearest, Math.hypot(block.x - (ax + dx * amount), block.y - (ay + dy * amount)));
    }
    return nearest;
  }

  // A point a fixed distance ahead along the route, plus how far remains.
  routeTarget(world) {
    const block = world.block;
    if (!this.path || this.path.length < 2) {
      return { x: world.goal.x, y: world.goal.y, remaining: Math.hypot(world.goal.x - block.x, world.goal.y - block.y) };
    }

    // Closest point on the route, then walk forward by the lookahead distance.
    let bestSegment = 0;
    let bestAmount = 0;
    let bestDistance = Infinity;
    for (let index = 1; index < this.path.length; index++) {
      const [ax, ay] = this.path[index - 1];
      const [bx, by] = this.path[index];
      const dx = bx - ax;
      const dy = by - ay;
      const lengthSquared = dx * dx + dy * dy;
      const amount = lengthSquared < 1e-12 ? 0 : clamp(((block.x - ax) * dx + (block.y - ay) * dy) / lengthSquared, 0, 1);
      const distance = Math.hypot(block.x - (ax + dx * amount), block.y - (ay + dy * amount));
      if (distance < bestDistance) {
        bestDistance = distance;
        bestSegment = index;
        bestAmount = amount;
      }
    }

    let remaining = 0;
    for (let index = bestSegment; index < this.path.length; index++) {
      const [ax, ay] = this.path[index - 1];
      const [bx, by] = this.path[index];
      const segment = Math.hypot(bx - ax, by - ay);
      remaining += index === bestSegment ? segment * (1 - bestAmount) : segment;
    }

    let budget = LOOKAHEAD;
    let currentX = this.path[bestSegment - 1][0] + (this.path[bestSegment][0] - this.path[bestSegment - 1][0]) * bestAmount;
    let currentY = this.path[bestSegment - 1][1] + (this.path[bestSegment][1] - this.path[bestSegment - 1][1]) * bestAmount;
    for (let index = bestSegment; index < this.path.length; index++) {
      const [bx, by] = this.path[index];
      const segment = Math.hypot(bx - currentX, by - currentY);
      if (segment >= budget) {
        const amount = budget / segment;
        return { x: currentX + (bx - currentX) * amount, y: currentY + (by - currentY) * amount, remaining };
      }
      budget -= segment;
      currentX = bx;
      currentY = by;
    }
    return { x: world.goal.x, y: world.goal.y, remaining };
  }

  selectContact(world) {
    const block = world.block;
    const cos = Math.cos(block.angle);
    const sin = Math.sin(block.angle);
    this.ensurePath(world);
    const route = this.routeTarget(world);
    const errorX = route.x - block.x;
    const errorY = route.y - block.y;
    const errorAngle = wrapAngle(world.goal.angle - block.angle);
    const scale = Math.sqrt(TEE.characteristicSquared);

    // Orientation only matters once the block is close to the end of its route.
    const arriving = route.remaining < LOOKAHEAD * 1.5;
    let desiredX = errorX;
    let desiredY = errorY;
    let desiredSpin =
      errorAngle * scale * this.rotationWeight * (arriving ? 1 : TRANSIT_ROTATION_SCALE);
    const desiredLength = Math.hypot(desiredX, desiredY, desiredSpin);
    if (desiredLength < 1e-9) return;
    desiredX /= desiredLength;
    desiredY /= desiredLength;
    desiredSpin /= desiredLength;

    let best = null;
    let bestScore = -Infinity;
    for (const candidate of CANDIDATES) {
      // Push direction is the inward surface normal, rotated into the world.
      const forceX = -(candidate.nx * cos - candidate.ny * sin);
      const forceY = -(candidate.nx * sin + candidate.ny * cos);
      const radiusX = candidate.x * cos - candidate.y * sin;
      const radiusY = candidate.x * sin + candidate.y * cos;
      const torque = radiusX * forceY - radiusY * forceX;
      const spin = (torque / TEE.characteristicSquared) * scale;
      const length = Math.hypot(forceX, forceY, spin);
      if (length < 1e-9) continue;
      const score = (forceX * desiredX + forceY * desiredY + spin * desiredSpin) / length;

      const approachX = block.x + (candidate.pusherX * cos - candidate.pusherY * sin);
      const approachY = block.y + (candidate.pusherX * sin + candidate.pusherY * cos);
      if (!this.reachable(world, approachX, approachY)) continue;

      // Repositioning dominates the episode if contact choice ignores it: the
      // pusher will happily orbit half the block for a marginally better push.
      // Charging for the trip keeps it working the face it is already on.
      const total = score - this.travelWeight * this.travelCost(world, approachX, approachY);
      if (total > bestScore) {
        bestScore = total;
        best = { candidate, approachX, approachY, forceX, forceY };
      }
    }

    if (!best) return;
    this.contact = best;
    this.approachSteps = 0;
    this.pushSteps = 0;
    this.bestCoverage = world.coverage();
    this.stalled = 0;

    // Re-aiming usually lands on a contact the pusher is already loading. Going
    // back out to the orbit radius in that case wastes most of the episode, so
    // stay in contact whenever the new target is effectively where we are.
    const settled =
      Math.hypot(world.pusher.x - best.approachX, world.pusher.y - best.approachY) < PUSHER_RADIUS * 1.5;
    if (settled) {
      this.state = "push";
      return;
    }
    this.state = "approach";
  }

  // With lifting, getting anywhere is a straight line.
  travelCost(world, targetX, targetY) {
    return Math.hypot(world.pusher.x - targetX, world.pusher.y - targetY);
  }

  reachable(world, x, y) {
    const low = WALL_THICKNESS + PUSHER_RADIUS;
    const high = 1 - WALL_THICKNESS - PUSHER_RADIUS;
    if (x < low || x > high || y < low || y > high) return false;
    return !world.pusherBlocked(x, y);
  }

  approachTarget(world) {
    // Straight to the contact, in the air. Nothing to route around.
    return [this.contact.approachX, this.contact.approachY, 1];
  }

  segmentHitsBlock(world, fromX, fromY, toX, toY) {
    const steps = Math.max(2, Math.ceil(Math.hypot(toX - fromX, toY - fromY) / 0.01));
    for (let index = 1; index < steps; index++) {
      const amount = index / steps;
      const x = fromX + (toX - fromX) * amount;
      const y = fromY + (toY - fromY) * amount;
      toLocal(world.block, x, y, scratch);
      for (const part of TEE.parts) {
        if (circleRectContact(scratch.x, scratch.y, PUSHER_RADIUS + 0.002, part, probe)) return true;
      }
    }
    return false;
  }

  pushTarget(world) {
    const contact = this.contact;
    // Aim past the contact along the push direction so the pusher keeps loading
    // the face instead of stopping the moment it touches. The stroke shortens as
    // the pose error closes, which is what stops the endgame oscillating between
    // over-rotating and over-translating.
    const error = world.poseError();
    const severity = Math.min(1, error.position / 0.12 + error.angle / 0.9);
    const stroke = PUSH_STROKE_NEAR + (PUSH_STROKE_FAR - PUSH_STROKE_NEAR) * severity;
    return [
      contact.approachX + contact.forceX * stroke,
      contact.approachY + contact.forceY * stroke,
      0,
    ];
  }

  advance(world, coverage) {
    if (this.state === "approach") {
      this.approachSteps += 1;
      const reached =
        Math.hypot(world.pusher.x - this.contact.approachX, world.pusher.y - this.contact.approachY) <
        ARRIVAL_TOLERANCE + MAX_PUSHER_SPEED;
      if (reached) {
        this.state = "push";
        this.pushSteps = 0;
        this.bestCoverage = coverage;
        this.stalled = 0;
      } else if (this.approachSteps > MAX_APPROACH_STEPS) {
        this.state = "select";
        this.contact = null;
      }
      return;
    }

    this.pushSteps += 1;
    if (coverage > this.bestCoverage + 1e-4) {
      this.bestCoverage = coverage;
      this.stalled = 0;
    } else {
      this.stalled += 1;
    }
    // Re-plan once the stroke is spent or the push stops paying off; the block
    // has rotated under the pusher by then and the best contact has moved.
    if (this.pushSteps >= MAX_PUSH_STEPS || this.stalled > 10) {
      this.state = "select";
      this.contact = null;
    }
  }

  describe() {
    if (!this.contact) return this.state;
    return this.state === "approach" ? "approach (lifted)" : "push";
  }
}

// Rolls one episode with the scripted expert, returning the trajectory.
export function rollout(world, expert, options = {}) {
  const horizon = options.horizon ?? 600;
  const observations = [];
  const actions = [];
  expert.reset();
  let success = false;
  let best = 0;
  for (let step = 0; step < horizon; step++) {
    const coverage = world.coverage();
    best = Math.max(best, coverage);
    if (coverage >= SUCCESS_COVERAGE) {
      success = true;
      break;
    }
    const observation = world.writeObservation();
    const [targetX, targetY, lift] = expert.act(world);
    observations.push(observation);
    actions.push(Float32Array.from([targetX, targetY, lift]));
    world.step(targetX, targetY, lift);
  }
  return { observations, actions, success, coverage: world.coverage(), best, steps: observations.length };
}
