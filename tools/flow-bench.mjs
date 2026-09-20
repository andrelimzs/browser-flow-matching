// Batched forward+backward against a faithful reimplementation of the
// per-sample loop the planner demo uses, at the same network shape.
import { MLP } from "../src/flow/mlp.js";
import { Adam } from "../src/flow/adam.js";

function makeRandom(seed) {
  let state = seed >>> 0;
  return () => { state += 0x6d2b79f5; let v = state; v = Math.imul(v ^ (v >>> 15), v | 1); v ^= v + Math.imul(v ^ (v >>> 7), v | 61); return ((v ^ (v >>> 14)) >>> 0) / 4294967296; };
}

const IN = 20, WIDTH = 64, OUT = 2, BATCH = 96, ITERS = 3000;

// ---- per-sample baseline, structured exactly as src/main.js does it ----
function baseline() {
  const random = makeRandom(5);
  const xavier = (size, fanIn, fanOut) => { const v = new Float32Array(size); const l = Math.sqrt(6/(fanIn+fanOut)); for (let i=0;i<size;i++) v[i]=(random()*2-1)*l; return v; };
  const w1 = xavier(WIDTH*IN, IN, WIDTH), b1 = new Float32Array(WIDTH);
  const w2 = xavier(WIDTH*WIDTH, WIDTH, WIDTH), b2 = new Float32Array(WIDTH);
  const w3 = xavier(OUT*WIDTH, WIDTH, OUT), b3 = new Float32Array(OUT);
  const params = [w1,b1,w2,b2,w3,b3];
  const grads = params.map(p => new Float32Array(p.length));
  const moments = params.map(p => new Float32Array(p.length));
  const velocities = params.map(p => new Float32Array(p.length));
  const input = new Float32Array(IN), h1 = new Float32Array(WIDTH), h2 = new Float32Array(WIDTH);
  const dh1 = new Float32Array(WIDTH), dh2 = new Float32Array(WIDTH);
  let step = 0;
  const started = process.hrtime.bigint();
  for (let iter = 0; iter < ITERS; iter++) {
    for (const g of grads) g.fill(0);
    for (let s = 0; s < BATCH; s++) {
      for (let i = 0; i < IN; i++) input[i] = random() * 2 - 1;
      for (let j = 0; j < WIDTH; j++) { const o=j*IN; let sum=b1[j]; for (let i=0;i<IN;i++) sum+=w1[o+i]*input[i]; h1[j]=Math.tanh(sum); }
      for (let j = 0; j < WIDTH; j++) { const o=j*WIDTH; let sum=b2[j]; for (let i=0;i<WIDTH;i++) sum+=w2[o+i]*h1[i]; h2[j]=Math.tanh(sum); }
      let ox=b3[0], oy=b3[1];
      for (let i=0;i<WIDTH;i++){ ox+=w3[i]*h2[i]; oy+=w3[WIDTH+i]*h2[i]; }
      const dx=(ox-0.3)/BATCH, dy=(oy+0.2)/BATCH;
      for (let j=0;j<WIDTH;j++){ grads[4][j]+=dx*h2[j]; grads[4][WIDTH+j]+=dy*h2[j]; dh2[j]=(w3[j]*dx+w3[WIDTH+j]*dy)*(1-h2[j]**2); }
      grads[5][0]+=dx; grads[5][1]+=dy;
      dh1.fill(0);
      for (let j=0;j<WIDTH;j++){ const o=j*WIDTH, d=dh2[j]; grads[3][j]+=d; for (let i=0;i<WIDTH;i++){ grads[2][o+i]+=d*h1[i]; dh1[i]+=w2[o+i]*d; } }
      for (let j=0;j<WIDTH;j++){ const d=dh1[j]*(1-h1[j]**2), o=j*IN; grads[1][j]+=d; for (let i=0;i<IN;i++) grads[0][o+i]+=d*input[i]; }
    }
    step++;
    const c1=1-0.9**step, c2=1-0.999**step;
    for (let p=0;p<params.length;p++){ const v=params[p],m=moments[p],vel=velocities[p],g=grads[p];
      for (let i=0;i<v.length;i++){ m[i]=0.9*m[i]+0.1*g[i]; vel[i]=0.999*vel[i]+0.001*g[i]*g[i]; v[i]-=0.002*(m[i]/c1)/(Math.sqrt(vel[i]/c2)+1e-8); } }
  }
  return Number(process.hrtime.bigint() - started) / 1e9;
}

// ---- batched ----
function batched(batchSize) {
  const random = makeRandom(5);
  const iters = Math.round(ITERS * BATCH / batchSize);
  const model = new MLP({ sizes: [IN, WIDTH, WIDTH, OUT], maxBatch: batchSize, random });
  const optimizer = new Adam(model.params, { learningRate: 0.002 });
  const input = model.inputBuffer();
  const gradOutput = new Float32Array(batchSize * OUT);
  const started = process.hrtime.bigint();
  for (let iter = 0; iter < iters; iter++) {
    for (let i = 0; i < batchSize * IN; i++) input[i] = random() * 2 - 1;
    model.zeroGrad();
    const out = model.forward(batchSize);
    for (let s = 0; s < batchSize; s++) {
      gradOutput[s*OUT] = (out[s*OUT] - 0.3) / batchSize;
      gradOutput[s*OUT+1] = (out[s*OUT+1] + 0.2) / batchSize;
    }
    model.backward(gradOutput, batchSize);
    optimizer.step(model.grads);
  }
  return Number(process.hrtime.bigint() - started) / 1e9;
}

const samples = ITERS * BATCH;
const base = baseline();
console.log(`network ${IN}->${WIDTH}->${WIDTH}->${OUT}, ${samples} sample passes\n`);
console.log(`per-sample loop (as in src/main.js)  ${base.toFixed(2)}s  ${Math.round(samples/base).toLocaleString()} samples/s  1.00x`);
for (const size of [96, 256, 1024]) {
  const t = batched(size);
  console.log(`batched, batch ${String(size).padStart(4)}                 ${t.toFixed(2)}s  ${Math.round(samples/t).toLocaleString()} samples/s  ${(base/t).toFixed(2)}x`);
}
