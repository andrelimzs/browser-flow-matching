// Geometry helpers for the planar pushing simulator.
//
// The block is a union of axis-aligned rectangles expressed in its own local
// frame, so every contact query reduces to circle-versus-AABB once the query
// point has been rotated into that frame. That keeps the solver allocation-free
// and avoids transforming vertices on every substep.

export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function wrapAngle(angle) {
  let wrapped = (angle + Math.PI) % (Math.PI * 2);
  if (wrapped < 0) wrapped += Math.PI * 2;
  return wrapped - Math.PI;
}

export function makeRect(centerX, centerY, width, height) {
  return {
    minX: centerX - width / 2,
    maxX: centerX + width / 2,
    minY: centerY - height / 2,
    maxY: centerY + height / 2,
  };
}

export function rectCorners(rect) {
  return [
    [rect.minX, rect.minY],
    [rect.maxX, rect.minY],
    [rect.maxX, rect.maxY],
    [rect.minX, rect.maxY],
  ];
}

// Builds the Push-T tee: a 4u by 1u bar with a 1u by 3u stem hanging below it,
// translated so the local origin sits on the area centroid.
export function makeTeeShape(unit) {
  const barArea = 4 * unit * unit;
  const stemArea = 3 * unit * unit;
  const centroidY = (barArea * 0 + stemArea * -2 * unit) / (barArea + stemArea);
  const parts = [
    makeRect(0, -centroidY, 4 * unit, unit),
    makeRect(0, -2 * unit - centroidY, unit, 3 * unit),
  ];
  // Single closed outline of the union, so the tee can be stroked as one shape
  // rather than as two rectangles with a seam down the middle.
  const shift = -centroidY;
  const outline = [
    [-2 * unit, 0.5 * unit + shift],
    [2 * unit, 0.5 * unit + shift],
    [2 * unit, -0.5 * unit + shift],
    [0.5 * unit, -0.5 * unit + shift],
    [0.5 * unit, -3.5 * unit + shift],
    [-0.5 * unit, -3.5 * unit + shift],
    [-0.5 * unit, -0.5 * unit + shift],
    [-2 * unit, -0.5 * unit + shift],
  ];

  return {
    unit,
    parts,
    outline,
    area: barArea + stemArea,
    radius: circumradius(parts),
    characteristicSquared: characteristicLengthSquared(parts),
  };
}

// Largest distance from the centroid to any corner: the orbit radius the
// pusher must clear to travel around the block without touching it.
function circumradius(parts) {
  let maximum = 0;
  for (const part of parts) {
    for (const [x, y] of rectCorners(part)) {
      maximum = Math.max(maximum, Math.hypot(x, y));
    }
  }
  return maximum;
}

// c^2 = (1/A) * integral of r^2 dA about the centroid. Under the ellipsoidal
// limit-surface approximation this is exactly the constant that couples applied
// torque to angular velocity, so the quasi-static solver reads it straight off
// the geometry rather than taking a tuned fudge factor.
function characteristicLengthSquared(parts) {
  let weighted = 0;
  let total = 0;
  for (const part of parts) {
    const width = part.maxX - part.minX;
    const height = part.maxY - part.minY;
    const area = width * height;
    const centerX = (part.minX + part.maxX) / 2;
    const centerY = (part.minY + part.maxY) / 2;
    const own = (width * width + height * height) / 12;
    const offset = centerX * centerX + centerY * centerY;
    weighted += area * (own + offset);
    total += area;
  }
  return weighted / total;
}

// Contact between a circle and an axis-aligned rectangle, both in the same
// frame. The normal points from the rectangle toward the circle centre, i.e.
// the direction that separates them; the caller negates it to get the force
// the circle applies to the rectangle.
export function circleRectContact(centerX, centerY, radius, rect, out) {
  const insideX = centerX > rect.minX && centerX < rect.maxX;
  const insideY = centerY > rect.minY && centerY < rect.maxY;

  if (insideX && insideY) {
    const toMinX = centerX - rect.minX;
    const toMaxX = rect.maxX - centerX;
    const toMinY = centerY - rect.minY;
    const toMaxY = rect.maxY - centerY;
    const smallest = Math.min(toMinX, toMaxX, toMinY, toMaxY);
    if (smallest === toMinX) {
      out.nx = -1; out.ny = 0; out.px = rect.minX; out.py = centerY;
    } else if (smallest === toMaxX) {
      out.nx = 1; out.ny = 0; out.px = rect.maxX; out.py = centerY;
    } else if (smallest === toMinY) {
      out.nx = 0; out.ny = -1; out.px = centerX; out.py = rect.minY;
    } else {
      out.nx = 0; out.ny = 1; out.px = centerX; out.py = rect.maxY;
    }
    out.depth = radius + smallest;
    return true;
  }

  const closestX = clamp(centerX, rect.minX, rect.maxX);
  const closestY = clamp(centerY, rect.minY, rect.maxY);
  const offsetX = centerX - closestX;
  const offsetY = centerY - closestY;
  const distance = Math.hypot(offsetX, offsetY);
  if (distance >= radius || distance === 0) return false;

  out.depth = radius - distance;
  out.nx = offsetX / distance;
  out.ny = offsetY / distance;
  out.px = closestX;
  out.py = closestY;
  return true;
}

export function pointInShape(localX, localY, parts) {
  for (const part of parts) {
    if (localX >= part.minX && localX <= part.maxX && localY >= part.minY && localY <= part.maxY) return true;
  }
  return false;
}

export function toLocal(pose, worldX, worldY, out) {
  const dx = worldX - pose.x;
  const dy = worldY - pose.y;
  const cos = Math.cos(pose.angle);
  const sin = Math.sin(pose.angle);
  out.x = dx * cos + dy * sin;
  out.y = -dx * sin + dy * cos;
  return out;
}

export function toWorld(pose, localX, localY, out) {
  const cos = Math.cos(pose.angle);
  const sin = Math.sin(pose.angle);
  out.x = pose.x + localX * cos - localY * sin;
  out.y = pose.y + localX * sin + localY * cos;
  return out;
}

