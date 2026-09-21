// Canvas rendering for the pushing arena.
//
// The world is a unit square; everything here is the mapping from that square
// into a padded, device-pixel-correct region of the canvas, plus the drawing
// itself. Nothing in this file feeds back into the simulation.

import { SUCCESS_COVERAGE, TEE, WALL_THICKNESS, PUSHER_RADIUS } from "./sim.js";

const INK = "#191c1b";
const BLUE = "#1d66db";
const CORAL = "#ed6b55";

export function createView(canvas) {
  const context = canvas.getContext("2d");
  let frame = { offsetX: 0, offsetY: 0, size: 1, width: 0, height: 0 };

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    const pad = Math.max(16, Math.min(rect.width, rect.height) * 0.04);
    const size = Math.min(rect.width - pad * 2, rect.height - pad * 2);
    frame = {
      offsetX: (rect.width - size) / 2,
      offsetY: (rect.height - size) / 2,
      size,
      width: rect.width,
      height: rect.height,
    };
    return frame;
  }

  const toX = (x) => frame.offsetX + x * frame.size;
  const toY = (y) => frame.offsetY + (1 - y) * frame.size;
  const toLength = (value) => value * frame.size;

  // Canvas point back into world coordinates, for mouse teleoperation.
  function fromClient(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const localX = clientX - rect.left;
    const localY = clientY - rect.top;
    return [
      (localX - frame.offsetX) / frame.size,
      1 - (localY - frame.offsetY) / frame.size,
    ];
  }

  function polygon(points) {
    context.beginPath();
    points.forEach(([x, y], index) => {
      const px = toX(x);
      const py = toY(y);
      if (index === 0) context.moveTo(px, py);
      else context.lineTo(px, py);
    });
    context.closePath();
  }

  function teeOutline(pose) {
    const cos = Math.cos(pose.angle);
    const sin = Math.sin(pose.angle);
    return TEE.outline.map(([x, y]) => [pose.x + x * cos - y * sin, pose.y + x * sin + y * cos]);
  }

  function draw(world, options = {}) {
    resize();
    const { width, height } = frame;
    context.clearRect(0, 0, width, height);

    const background = context.createLinearGradient(0, 0, width, height);
    background.addColorStop(0, "#f8f7f2");
    background.addColorStop(1, "#e7e5de");
    context.fillStyle = background;
    context.fillRect(0, 0, width, height);

    // Grid
    context.strokeStyle = "rgba(25, 28, 27, .05)";
    context.lineWidth = 1;
    for (let index = 0; index <= 10; index++) {
      const amount = index / 10;
      context.beginPath();
      context.moveTo(toX(amount), toY(0));
      context.lineTo(toX(amount), toY(1));
      context.moveTo(toX(0), toY(amount));
      context.lineTo(toX(1), toY(amount));
      context.stroke();
    }

    // Wall band
    context.strokeStyle = "rgba(25, 28, 27, .3)";
    context.lineWidth = Math.max(2, toLength(WALL_THICKNESS));
    const inset = toLength(WALL_THICKNESS / 2);
    context.strokeRect(toX(0) + inset, toY(1) + inset, toLength(1) - inset * 2, toLength(1) - inset * 2);

    // Goal pose
    const goal = teeOutline(world.goal);
    context.save();
    context.setLineDash([toLength(0.012), toLength(0.009)]);
    polygon(goal);
    context.fillStyle = "rgba(29, 102, 219, .10)";
    context.fill();
    context.strokeStyle = "rgba(29, 102, 219, .75)";
    context.lineWidth = 1.6;
    context.stroke();
    context.restore();

    // A second block pose, drawn behind everything: used by the viewer to show
    // where the recorded run had the block at this point in time.
    if (options.ghost) {
      const ghost = teeOutline(options.ghost);
      polygon(ghost);
      context.fillStyle = "rgba(38, 44, 41, .10)";
      context.fill();
      context.setLineDash([toLength(0.01), toLength(0.008)]);
      context.strokeStyle = "rgba(38, 44, 41, .4)";
      context.lineWidth = 1.2;
      context.stroke();
      context.setLineDash([]);
    }

    // Faded peer block poses used by multi-rollout evaluation playback.
    if (options.ghosts) {
      for (const pose of options.ghosts) {
        const ghost = teeOutline(pose);
        polygon(ghost);
        context.fillStyle = "rgba(38, 44, 41, .065)";
        context.fill();
        context.strokeStyle = "rgba(38, 44, 41, .2)";
        context.lineWidth = 0.9;
        context.stroke();
      }
    }

    // Obstacles
    for (const obstacle of world.obstacles) {
      context.beginPath();
      context.arc(toX(obstacle.x), toY(obstacle.y), toLength(obstacle.r), 0, Math.PI * 2);
      context.fillStyle = "rgba(75, 80, 77, .13)";
      context.fill();
      context.strokeStyle = "rgba(75, 80, 77, .5)";
      context.lineWidth = 1.2;
      context.stroke();
    }

    // Flow candidates. Each polyline is one sampled action chunk, drawn while
    // it is still being transported from noise, so the spread at the start and
    // the collapse onto a trajectory are both visible.
    if (options.flow?.lines?.length) {
      const { lines, chosen, progress, executing } = options.flow;
      const converged = progress >= 1;

      // A tick per Euler step, filled as the integration proceeds, so the ten
      // steps are countable rather than a blur.
      if (!executing) {
        const total = options.flow.steps ?? 10;
        const done = options.flow.step ?? 0;
        const tick = Math.max(4, toLength(0.012));
        const gap = tick * 0.5;
        const width = total * tick + (total - 1) * gap;
        const left = frame.offsetX + (frame.size - width) / 2;
        const top = frame.offsetY + 10;
        for (let index = 0; index < total; index++) {
          context.fillStyle = index < done ? "rgba(29, 102, 219, .8)" : "rgba(25, 28, 27, .16)";
          context.fillRect(left + index * (tick + gap), top, tick, 3);
        }
      }

      if (!converged) {
        // Mid-transport the samples are a cloud, not a path: drawing them as
        // points is honest about that, and the collapse from noise into
        // structure is the thing worth watching.
        const radius = Math.max(1.2, toLength(0.004)) * (0.7 + progress * 0.6);
        for (let index = 0; index < lines.length; index++) {
          const points = lines[index];
          const isChosen = index === chosen;
          context.fillStyle = isChosen
            ? `rgba(29, 102, 219, ${0.3 + progress * 0.45})`
            : `rgba(75, 80, 77, ${0.16 + progress * 0.34})`;
          for (let k = 0; k < points.length; k += 2) {
            context.beginPath();
            context.arc(toX(points[k]), toY(points[k + 1]), radius, 0, Math.PI * 2);
            context.fill();
          }
        }
      } else {
        // Converged: the samples are trajectories, so draw them as such. While
        // the committed one is being executed only it is shown.
        for (let index = 0; index < lines.length; index++) {
          const isChosen = index === chosen;
          if (executing && !isChosen) continue;
          const points = lines[index];
          context.beginPath();
          for (let k = 0; k < points.length; k += 2) {
            const px = toX(points[k]);
            const py = toY(points[k + 1]);
            if (k === 0) context.moveTo(px, py);
            else context.lineTo(px, py);
          }
          context.strokeStyle = isChosen ? "rgba(29, 102, 219, .85)" : "rgba(75, 80, 77, .3)";
          context.lineWidth = isChosen ? 2 : 1;
          context.stroke();
          context.beginPath();
          context.arc(toX(points[0]), toY(points[1]), isChosen ? 3.2 : 1.8, 0, Math.PI * 2);
          context.fillStyle = isChosen ? "rgba(29, 102, 219, .9)" : "rgba(75, 80, 77, .35)";
          context.fill();
        }
      }
    }

    // Compared trajectories, drawn up to a cursor so they can be scrubbed.
    if (options.paths) {
      for (const path of options.paths) {
        const points = path.points;
        const upto = path.cursor === undefined ? points.length / 2 : Math.min(points.length / 2, path.cursor);
        if (upto < 2) continue;
        context.beginPath();
        for (let k = 0; k < upto; k++) {
          const px = toX(points[k * 2]);
          const py = toY(points[k * 2 + 1]);
          if (k === 0) context.moveTo(px, py);
          else context.lineTo(px, py);
        }
        context.strokeStyle = path.color;
        context.lineWidth = path.width ?? 1.6;
        if (path.dashed) context.setLineDash([toLength(0.009), toLength(0.007)]);
        context.stroke();
        context.setLineDash([]);
        // Head marker at the cursor.
        if (path.head !== false) {
          context.beginPath();
          context.arc(toX(points[(upto - 1) * 2]), toY(points[(upto - 1) * 2 + 1]), 3.4, 0, Math.PI * 2);
          context.fillStyle = path.color;
          context.fill();
        }
      }
    }

    // Pusher trail
    if (options.trail?.length > 1) {
      context.beginPath();
      options.trail.forEach(([x, y], index) => {
        if (index === 0) context.moveTo(toX(x), toY(y));
        else context.lineTo(toX(x), toY(y));
      });
      context.strokeStyle = "rgba(237, 107, 85, .3)";
      context.lineWidth = 1.4;
      context.stroke();
    }

    // Action distribution for RL rollouts. The translucent shape is the
    // one-standard-deviation contour after applying the algorithm's action
    // transform; the spoke shows the deterministic mean action.
    if (options.actionDistribution) {
      const { meanX, meanY, logStdX, logStdY, squashed = true } = options.actionDistribution;
      const centerX = toX(world.pusher.x);
      const centerY = toY(world.pusher.y);
      const radarRadius = toLength(0.085);
      const stdX = Math.exp(logStdX);
      const stdY = Math.exp(logStdY);
      const actionVector = (rawX, rawY) => {
        const x = squashed ? Math.tanh(rawX) : rawX;
        const y = squashed ? Math.tanh(rawY) : rawY;
        const scale = 1 / Math.max(1, Math.hypot(x, y));
        return [x * scale, y * scale];
      };

      context.save();
      context.strokeStyle = "rgba(237, 107, 85, .18)";
      context.lineWidth = 1;
      context.beginPath();
      context.arc(centerX, centerY, radarRadius, 0, Math.PI * 2);
      context.moveTo(centerX - radarRadius, centerY);
      context.lineTo(centerX + radarRadius, centerY);
      context.moveTo(centerX, centerY - radarRadius);
      context.lineTo(centerX, centerY + radarRadius);
      context.stroke();

      context.beginPath();
      const points = 48;
      for (let index = 0; index <= points; index++) {
        const angle = index / points * Math.PI * 2;
        const [actionX, actionY] = actionVector(
          meanX + stdX * Math.cos(angle),
          meanY + stdY * Math.sin(angle),
        );
        const x = centerX + actionX * radarRadius;
        const y = centerY - actionY * radarRadius;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.closePath();
      context.fillStyle = "rgba(237, 107, 85, .13)";
      context.fill();
      context.strokeStyle = "rgba(237, 107, 85, .55)";
      context.lineWidth = 1.2;
      context.stroke();

      const [actionMeanX, actionMeanY] = actionVector(meanX, meanY);
      const meanPointX = centerX + actionMeanX * radarRadius;
      const meanPointY = centerY - actionMeanY * radarRadius;
      context.beginPath();
      context.moveTo(centerX, centerY);
      context.lineTo(meanPointX, meanPointY);
      context.strokeStyle = "rgba(237, 107, 85, .8)";
      context.lineWidth = 1.5;
      context.stroke();
      context.beginPath();
      context.arc(meanPointX, meanPointY, 2.8, 0, Math.PI * 2);
      context.fillStyle = "rgba(237, 107, 85, .88)";
      context.fill();
      context.restore();
    }

    // Block
    const block = teeOutline(world.block);
    polygon(block);
    context.fillStyle = "rgba(38, 44, 41, .88)";
    context.fill();
    context.strokeStyle = INK;
    context.lineWidth = 1.5;
    context.stroke();

    // Orientation tick from the centroid along the stem.
    const tickCos = Math.cos(world.block.angle - Math.PI / 2);
    const tickSin = Math.sin(world.block.angle - Math.PI / 2);
    context.beginPath();
    context.moveTo(toX(world.block.x), toY(world.block.y));
    context.lineTo(
      toX(world.block.x + tickCos * TEE.unit * 1.8),
      toY(world.block.y + tickSin * TEE.unit * 1.8),
    );
    context.strokeStyle = "rgba(199, 240, 75, .9)";
    context.lineWidth = 2;
    context.stroke();

    // Commanded action
    if (options.showAction && options.action) {
      const [actionX, actionY] = options.action;
      context.beginPath();
      context.arc(toX(actionX), toY(actionY), Math.max(3, toLength(0.009)), 0, Math.PI * 2);
      context.fillStyle = "rgba(29, 102, 219, .55)";
      context.fill();
      context.beginPath();
      context.moveTo(toX(world.pusher.x), toY(world.pusher.y));
      context.lineTo(toX(actionX), toY(actionY));
      context.strokeStyle = "rgba(29, 102, 219, .35)";
      context.lineWidth = 1.2;
      context.stroke();
    }

    // Pusher. Translucent while lifted, since it is off the table and passes
    // over the block without touching it; a dashed outline keeps it legible
    // against the block it is crossing.
    const lifted = world.lifted;
    context.beginPath();
    context.arc(toX(world.pusher.x), toY(world.pusher.y), toLength(PUSHER_RADIUS), 0, Math.PI * 2);
    context.fillStyle = lifted ? "rgba(237, 107, 85, .28)" : CORAL;
    context.fill();
    context.save();
    if (lifted) context.setLineDash([toLength(0.008), toLength(0.006)]);
    context.strokeStyle = lifted ? "rgba(120, 44, 32, .5)" : "rgba(120, 44, 32, .55)";
    context.lineWidth = 1.4;
    context.stroke();
    context.restore();

    if (options.coverage !== undefined) drawCoverage(options.coverage);
  }

  function drawCoverage(coverage) {
    const barWidth = toLength(0.28);
    const barHeight = 6;
    const x = frame.offsetX + frame.size - barWidth;
    const y = frame.offsetY + frame.size + 12;
    if (y + barHeight > frame.height) return;
    context.fillStyle = "rgba(25, 28, 27, .1)";
    context.fillRect(x, y, barWidth, barHeight);
    context.fillStyle = coverage >= SUCCESS_COVERAGE ? "#3f9d5a" : BLUE;
    context.fillRect(x, y, barWidth * Math.min(1, coverage), barHeight);
  }

  return { draw, resize, fromClient };
}
