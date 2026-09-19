import "./style.css";

const INPUT_WIDTH = 5;
const WIDTH = 64;
const BATCH = 96;
const LR = 0.002;
const PARTICLE_COUNT = 260;
const WORLD = 3.35;

const $ = (selector) => document.querySelector(selector);
const flowCanvas = $("#flowCanvas");
const lossCanvas = $("#lossCanvas");
const flowCtx = flowCanvas.getContext("2d");
const lossCtx = lossCanvas.getContext("2d");

let seed = (Date.now() ^ 0x9e3779b9) >>> 0;
function random() {
  seed += 0x6d2b79f5;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function gaussian() {
  const u = Math.max(1e-7, random());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

function sampleSource() {
  return [gaussian() * 0.82, gaussian() * 0.82];
}

function sampleCheckerboard() {
  const active = Math.floor(random() * 8);
  const row = Math.floor(active / 2);
  const col = (active % 2) * 2 + (row % 2);
  const cell = 1.18;
  const inset = 0.06;
  return [
    (col - 1.5) * cell + (random() - 0.5) * (cell - inset),
    (row - 1.5) * cell + (random() - 0.5) * (cell - inset),
  ];
}

function normalCDF(value) {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * x);
  const polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const erf = sign * (1 - polynomial * Math.exp(-x * x));
  return Math.min(1 - 1e-7, Math.max(1e-7, 0.5 * (1 + erf)));
}

// For the built-in checkerboard we can use an exact, stable coupling instead
// of relearning a different minibatch assignment every step. Gaussian x picks
// one of the two occupied columns; Gaussian y picks the row. Fractional
// quantiles remain uniform coordinates inside the selected square.
function coupleSourceToCheckerboard(source) {
  const u = normalCDF(source[0] / 0.82);
  const v = normalCDF(source[1] / 0.82);
  const side = Math.min(1, Math.floor(u * 2));
  const row = Math.min(3, Math.floor(v * 4));
  const col = side * 2 + (row % 2);
  const localX = u * 2 - side;
  const localY = v * 4 - row;
  const cell = 1.18;
  const span = cell - 0.06;
  return [
    (col - 1.5) * cell + (localX - 0.5) * span,
    (row - 1.5) * cell + (localY - 0.5) * span,
  ];
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
    this.grads = this.params.map((p) => new Float32Array(p.length));
    this.moments = this.params.map((p) => new Float32Array(p.length));
    this.velocities = this.params.map((p) => new Float32Array(p.length));
    this.step = 0;
    this.h1 = new Float32Array(WIDTH);
    this.h2 = new Float32Array(WIDTH);
    this.dh1 = new Float32Array(WIDTH);
    this.dh2 = new Float32Array(WIDTH);
  }

  forward(x, y, time) {
    const h1 = this.h1;
    const h2 = this.h2;
    const timeSin = Math.sin(Math.PI * 2 * time);
    const timeCos = Math.cos(Math.PI * 2 * time);
    for (let j = 0; j < WIDTH; j++) {
      const k = j * INPUT_WIDTH;
      h1[j] = Math.tanh(
        this.w1[k] * x +
        this.w1[k + 1] * y +
        this.w1[k + 2] * time +
        this.w1[k + 3] * timeSin +
        this.w1[k + 4] * timeCos +
        this.b1[j],
      );
    }
    for (let j = 0; j < WIDTH; j++) {
      let sum = this.b2[j];
      const offset = j * WIDTH;
      for (let i = 0; i < WIDTH; i++) sum += this.w2[offset + i] * h1[i];
      h2[j] = Math.tanh(sum);
    }
    let out0 = this.b3[0];
    let out1 = this.b3[1];
    for (let i = 0; i < WIDTH; i++) {
      out0 += this.w3[i] * h2[i];
      out1 += this.w3[WIDTH + i] * h2[i];
    }
    return [out0, out1];
  }

  trainBatch() {
    const grads = this.grads;
    for (const grad of grads) grad.fill(0);
    const sources = Array.from({ length: BATCH }, sampleSource);
    const targets = sources.map(coupleSourceToCheckerboard);
    let loss = 0;
    for (let n = 0; n < BATCH; n++) {
      const x0 = sources[n];
      const x1 = targets[n];
      // Keep full-path coverage while spending half the updates near the target,
      // where the checkerboard's sharp empty-cell boundaries are hardest.
      const timeSample = random() < 0.5 ? random() : 1 - random() ** 2;
      const time = 0.02 + timeSample * 0.96;
      const x = x0[0] * (1 - time) + x1[0] * time;
      const y = x0[1] * (1 - time) + x1[1] * time;
      const target0 = x1[0] - x0[0];
      const target1 = x1[1] - x0[1];
      const output = this.forward(x, y, time);
      const d0 = (output[0] - target0) / BATCH;
      const d1 = (output[1] - target1) / BATCH;
      loss += 0.5 * ((output[0] - target0) ** 2 + (output[1] - target1) ** 2);

      for (let j = 0; j < WIDTH; j++) {
        grads[4][j] += d0 * this.h2[j];
        grads[4][WIDTH + j] += d1 * this.h2[j];
        this.dh2[j] = (this.w3[j] * d0 + this.w3[WIDTH + j] * d1) * (1 - this.h2[j] ** 2);
      }
      grads[5][0] += d0;
      grads[5][1] += d1;

      this.dh1.fill(0);
      for (let j = 0; j < WIDTH; j++) {
        const offset = j * WIDTH;
        const delta = this.dh2[j];
        grads[3][j] += delta;
        for (let i = 0; i < WIDTH; i++) {
          grads[2][offset + i] += delta * this.h1[i];
          this.dh1[i] += this.w2[offset + i] * delta;
        }
      }

      for (let j = 0; j < WIDTH; j++) {
        const delta = this.dh1[j] * (1 - this.h1[j] ** 2);
        const offset = j * INPUT_WIDTH;
        grads[1][j] += delta;
        grads[0][offset] += delta * x;
        grads[0][offset + 1] += delta * y;
        grads[0][offset + 2] += delta * time;
        grads[0][offset + 3] += delta * Math.sin(Math.PI * 2 * time);
        grads[0][offset + 4] += delta * Math.cos(Math.PI * 2 * time);
      }
    }

    this.step++;
    const b1Correction = 1 - 0.9 ** this.step;
    const b2Correction = 1 - 0.999 ** this.step;
    for (let p = 0; p < this.params.length; p++) {
      const values = this.params[p];
      const m = this.moments[p];
      const v = this.velocities[p];
      const g = grads[p];
      for (let i = 0; i < values.length; i++) {
        m[i] = 0.9 * m[i] + 0.1 * g[i];
        v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
        values[i] -= LR * (m[i] / b1Correction) / (Math.sqrt(v[i] / b2Correction) + 1e-8);
      }
    }
    return loss / BATCH;
  }
}

let model = new TinyMLP();
let training = true;
let flowPlaying = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let simTime = 0;
let holdUntil = 0;
let particles = [];
let lossHistory = [];
let lossRecordEvery = 6;
let smoothedLoss = null;
let lastFrame = performance.now();
let telemetryAt = performance.now();
let telemetrySteps = 0;
let measuredSpeed = 0;
let latestHitRate = null;

function isInTargetCell(x, y) {
  const edge = 2.36;
  const cell = 1.18;
  if (x < -edge || x >= edge || y < -edge || y >= edge) return false;
  const col = Math.floor((x + edge) / cell);
  const row = Math.floor((y + edge) / cell);
  return (row + col) % 2 === 0;
}

function resetParticles() {
  particles = Array.from({ length: PARTICLE_COUNT }, () => {
    const [x, y] = sampleSource();
    return { x, y, x0: x, y0: y, tail: [] };
  });
  simTime = 0;
  holdUntil = 0;
  syncTimeUI();
}

function setParticlesTo(time) {
  for (const particle of particles) {
    particle.x = particle.x0;
    particle.y = particle.y0;
    particle.tail = [];
  }
  simTime = 0;
  const steps = Math.max(1, Math.ceil(time * 84));
  const dt = time / steps;
  for (let s = 0; s < steps; s++) advanceParticles(dt, false);
  simTime = time;
  syncTimeUI();
}

function advanceParticles(dt, keepTails = true) {
  if (dt <= 0) return;
  for (const p of particles) {
    if (keepTails && model.step > 50 && Math.floor(simTime * 1000) % 4 === 0) {
      p.tail.push([p.x, p.y]);
      if (p.tail.length > 7) p.tail.shift();
    }
    const [vx, vy] = model.forward(p.x, p.y, simTime);
    p.x += vx * dt;
    p.y += vy * dt;
  }
  simTime = Math.min(1, simTime + dt);
}

function syncTimeUI() {
  $("#timeSlider").value = simTime;
  $("#timeOutput").value = simTime.toFixed(2);
  const phase = simTime < 0.04 ? "Source noise" : simTime > 0.96 ? "Generated samples" : "Transporting";
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

function drawArrow(ctx, x, y, dx, dy, alpha) {
  const length = Math.hypot(dx, dy);
  if (length < 0.01) return;
  const scale = Math.min(15, length * 5.3) / length;
  const ex = x + dx * scale;
  const ey = y + dy * scale;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(ex, ey);
  ctx.strokeStyle = `rgba(29, 102, 219, ${alpha})`;
  ctx.lineWidth = 1;
  ctx.stroke();
  const angle = Math.atan2(ey - y, ex - x);
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - 3.5 * Math.cos(angle - 0.55), ey - 3.5 * Math.sin(angle - 0.55));
  ctx.lineTo(ex - 3.5 * Math.cos(angle + 0.55), ey - 3.5 * Math.sin(angle + 0.55));
  ctx.closePath();
  ctx.fillStyle = `rgba(29, 102, 219, ${alpha})`;
  ctx.fill();
}

function drawFlow() {
  const rect = resizeCanvas(flowCanvas, flowCtx);
  const w = rect.width;
  const h = rect.height;
  const scale = Math.min(w, h) / (WORLD * 2);
  const ox = w / 2;
  const oy = h / 2;
  const sx = (x) => ox + x * scale;
  const sy = (y) => oy - y * scale;
  flowCtx.clearRect(0, 0, w, h);

  const gradient = flowCtx.createRadialGradient(ox, oy, 5, ox, oy, Math.min(w, h) * 0.52);
  gradient.addColorStop(0, "#f8f7f2");
  gradient.addColorStop(1, "#e6e4dd");
  flowCtx.fillStyle = gradient;
  flowCtx.fillRect(0, 0, w, h);

  const cell = 1.18;
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      if ((row + col) % 2 !== 0) continue;
      const x = (col - 2) * cell;
      const y = (row - 2) * cell;
      flowCtx.fillStyle = `rgba(29, 102, 219, ${0.025 + simTime * 0.065})`;
      flowCtx.fillRect(sx(x), sy(y + cell), cell * scale, cell * scale);
      flowCtx.strokeStyle = `rgba(29, 102, 219, ${0.11 + simTime * 0.06})`;
      flowCtx.lineWidth = 1;
      flowCtx.strokeRect(sx(x), sy(y + cell), cell * scale, cell * scale);
    }
  }

  flowCtx.strokeStyle = "rgba(25, 28, 27, .055)";
  flowCtx.lineWidth = 1;
  for (let i = -3; i <= 3; i++) {
    flowCtx.beginPath(); flowCtx.moveTo(sx(i), 0); flowCtx.lineTo(sx(i), h); flowCtx.stroke();
    flowCtx.beginPath(); flowCtx.moveTo(0, sy(i)); flowCtx.lineTo(w, sy(i)); flowCtx.stroke();
  }

  if (model.step > 8) {
    const gap = w < 520 ? 58 : 64;
    for (let px = gap / 2; px < w; px += gap) {
      for (let py = gap / 2; py < h; py += gap) {
        const x = (px - ox) / scale;
        const y = (oy - py) / scale;
        const [vx, vy] = model.forward(x, y, simTime);
        drawArrow(flowCtx, px, py, vx, -vy, 0.24);
      }
    }
  }

  for (const p of particles) {
    if (p.tail.length > 1) {
      flowCtx.beginPath();
      for (let i = 0; i < p.tail.length; i++) {
        const [tx, ty] = p.tail[i];
        if (i === 0) flowCtx.moveTo(sx(tx), sy(ty)); else flowCtx.lineTo(sx(tx), sy(ty));
      }
      flowCtx.lineTo(sx(p.x), sy(p.y));
      flowCtx.strokeStyle = "rgba(237, 107, 85, .15)";
      flowCtx.lineWidth = 1;
      flowCtx.stroke();
    }
    flowCtx.beginPath();
    flowCtx.arc(sx(p.x), sy(p.y), w < 520 ? 2.1 : 2.5, 0, Math.PI * 2);
    flowCtx.fillStyle = "rgba(226, 79, 58, .79)";
    flowCtx.fill();
  }

  flowCtx.fillStyle = "rgba(25, 28, 27, .44)";
  flowCtx.font = "10px DM Mono, monospace";
  flowCtx.fillText(`MLP FIELD · STEP ${model.step.toLocaleString()}`, 15, h - 16);
}

function drawLoss() {
  const rect = resizeCanvas(lossCanvas, lossCtx);
  const w = rect.width;
  const h = rect.height;
  lossCtx.clearRect(0, 0, w, h);
  lossCtx.strokeStyle = "rgba(25, 28, 27, .07)";
  lossCtx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (h * i) / 4;
    lossCtx.beginPath(); lossCtx.moveTo(0, y); lossCtx.lineTo(w, y); lossCtx.stroke();
  }
  const plotTop = 6;
  const plotBottom = h - 18;
  if (lossHistory.length < 2) return;
  const logs = lossHistory.map((point) => Math.log(Math.max(point.loss, 1e-5)));
  let min = Math.min(...logs);
  let max = Math.max(...logs);
  if (max - min < 0.25) { max += 0.125; min -= 0.125; }
  lossCtx.beginPath();
  lossHistory.forEach((point, i) => {
    const x = (point.step / Math.max(1, model.step)) * w;
    const y = plotBottom - ((logs[i] - min) / (max - min)) * (plotBottom - plotTop);
    if (i === 0) lossCtx.moveTo(x, y); else lossCtx.lineTo(x, y);
  });
  lossCtx.strokeStyle = "#1d66db";
  lossCtx.lineWidth = 1.8;
  lossCtx.stroke();
  lossCtx.fillStyle = "rgba(25, 28, 27, .48)";
  lossCtx.font = "9px DM Mono, monospace";
  lossCtx.textAlign = "left";
  lossCtx.fillText("0", 0, h - 3);
  lossCtx.textAlign = "right";
  lossCtx.fillText(model.step.toLocaleString(), w, h - 3);
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
  $("#hitMetric").textContent = latestHitRate === null ? "—" : `${Math.round(latestHitRate * 100)}%`;
  if (lossHistory.length > 12) {
    const old = lossHistory[Math.max(0, lossHistory.length - 12)].loss;
    const newest = lossHistory[lossHistory.length - 1].loss;
    const change = ((newest - old) / old) * 100;
    $("#lossTrend").textContent = change < -1 ? `${Math.abs(change).toFixed(0)}% ↓ recent` : "Stabilizing";
  }
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
        latestHitRate = particles.filter((particle) => isInTargetCell(particle.x, particle.y)).length / particles.length;
        holdUntil = now + 1400;
        updateTelemetry();
      }
      else if (now >= holdUntil) resetParticles();
    } else {
      advanceParticles(Math.min(0.012, elapsed / 6800));
      syncTimeUI();
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

$("#trainingToggle").addEventListener("click", () => {
  training = !training;
  $("#trainingToggle").innerHTML = training ? '<span aria-hidden="true">Ⅱ</span> Pause' : '<span aria-hidden="true">▶</span> Train';
  $("#trainingStatus").textContent = training ? "Running on the main thread" : "Model weights are frozen";
});

$("#flowToggle").addEventListener("click", () => {
  flowPlaying = !flowPlaying;
  $("#flowToggle").innerHTML = flowPlaying ? '<span aria-hidden="true">Ⅱ</span>' : '<span aria-hidden="true">▶</span>';
  $("#flowToggle").setAttribute("aria-label", flowPlaying ? "Pause particle animation" : "Play particle animation");
});

$("#timeSlider").addEventListener("input", (event) => {
  flowPlaying = false;
  $("#flowToggle").innerHTML = '<span aria-hidden="true">▶</span>';
  setParticlesTo(Number(event.target.value));
});

$("#resampleButton").addEventListener("click", resetParticles);

$("#resetButton").addEventListener("click", () => {
  seed = (Date.now() ^ 0x85ebca6b) >>> 0;
  model = new TinyMLP();
  lossHistory = [];
  lossRecordEvery = 6;
  smoothedLoss = null;
  latestHitRate = null;
  training = true;
  $("#trainingToggle").innerHTML = '<span aria-hidden="true">Ⅱ</span> Pause';
  $("#trainingStatus").textContent = "Running on the main thread";
  $("#lossTrend").textContent = "Collecting data";
  resetParticles();
  updateTelemetry();
  drawLoss();
});

window.addEventListener("resize", () => { drawFlow(); drawLoss(); });

resetParticles();
updateTelemetry();
if (!flowPlaying) {
  $("#flowToggle").innerHTML = '<span aria-hidden="true">▶</span>';
  $("#flowToggle").setAttribute("aria-label", "Play particle animation");
}
requestAnimationFrame(animate);
