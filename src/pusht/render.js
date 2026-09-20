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

    // Pusher
    context.beginPath();
    context.arc(toX(world.pusher.x), toY(world.pusher.y), toLength(PUSHER_RADIUS), 0, Math.PI * 2);
    context.fillStyle = CORAL;
    context.fill();
    context.strokeStyle = "rgba(120, 44, 32, .55)";
    context.lineWidth = 1.4;
    context.stroke();

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
