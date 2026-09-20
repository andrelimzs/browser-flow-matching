// How different are the two orbit branches from an identical state?
// If the answer is "barely", the task does not justify a multimodal policy.
import { PushWorld, createRandom } from "../src/pusht/sim.js";
import { ScriptedExpert } from "../src/pusht/expert.js";

const random = createRandom(2024);
const probe = new PushWorld({ random, obstacleCount: Number(process.env.OBSTACLES ?? 1) });
const horizon = 60;
let divergent = 0;
let split = 0;
let gapSum = 0;
const episodes = 300;

for (let episode = 0; episode < episodes; episode++) {
  probe.reset();
  // Advance a little so the pusher is somewhere non-trivial.
  const warm = new ScriptedExpert({ random, tieBreak: "cw" });
  for (let step = 0; step < 40; step++) { const [x, y, lift] = warm.act(probe); probe.step(x, y, lift); }
  const state = probe.snapshot();

  const traces = [];
  for (const tieBreak of ["cw", "ccw"]) {
    probe.restore(state);
    const expert = new ScriptedExpert({ random, tieBreak });
    const actions = [];
    for (let step = 0; step < horizon; step++) {
      const [x, y, lift] = expert.act(probe);
      actions.push([x, y, lift]);
      probe.step(x, y, lift);
    }
    traces.push(actions);
  }

  // Peak separation between the two action sequences.
  let peak = 0;
  for (let step = 0; step < horizon; step++) {
    peak = Math.max(peak, Math.hypot(traces[0][step][0] - traces[1][step][0], traces[0][step][1] - traces[1][step][1]));
  }
  gapSum += peak;
  if (peak > 0.05) divergent += 1;
  if (peak > 0.2) split += 1;
}

console.log(`episodes ${episodes}, ${horizon}-step branch comparison from identical states (obstacles=${process.env.OBSTACLES ?? 1})`);
console.log(`mean peak action separation  ${(gapSum / episodes).toFixed(3)} world units`);
console.log(`states where branches differ at all (>0.05): ${divergent} (${(100 * divergent / episodes).toFixed(0)}%)`);
console.log(`states with a wide split      (>0.20): ${split} (${(100 * split / episodes).toFixed(0)}%)`);
console.log(`for scale: block circumradius 0.134, arena 1.0 wide`);
