# Flow Planner

An interactive, dependency-light robotics flow-matching demo that trains a tiny MLP entirely in the browser. A clearance-aware A* planner generates a safe reference trajectory through a walled obstacle field, and the model learns to transform a uniform free-space distribution into the complete path distribution.

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:5173`.

## How it works

- World: fixed bottom-left and top-right route endpoints, a perimeter wall, and 2–6 obstacles placed sequentially by solving a path, blocking it, and confirming a replacement path remains feasible; the default is 4
- Planner: 8-connected A* with a logarithmic clearance barrier around obstacles and walls, followed by barrier-aware path shortcutting; the centerline reserves the target ribbon’s width beyond the displayed safety envelope
- Source distribution: uniform samples over collision-free workspace positions
- Target distribution: the ordered curve `p(s) = (x(s), y(s))`, with `s` sampled uniformly along the complete A* route
- Coupling: independent free-space/path samples by default; optional progress mode ranks sources from start-nearest to goal-nearest and couples them to increasing `s`
- Probability path: `x_τ = (1 - τ)x_0 + τp(s)`
- Target velocity: `p(s) - x_0`
- Conditioning: path progress `s` is enabled by default and held fixed while flow time `τ` advances from 0 to 1
- Model: `20 → 64 → 64 → 2` tanh MLP with multiscale spatial, flow-time, and path-progress Fourier features
- Time sampling: half uniform, half biased toward the thin terminal path distribution
- Optimizer: Adam, implemented with typed arrays
- Sampling: Euler integration with shared, configurable 1, 2, 4, 10, or 20-step paths for animation and scrubbing, defaulting to 10; playback shows only real solver states

There are no ML runtime dependencies. Planning, training, backpropagation, optimization, and inference are implemented directly in `src/main.js`.

---

# Push-T (`/pusht`)

A second, self-contained site in this repo: quasi-static planar pushing, where a circular pusher has to
align a T-shaped block with a goal pose among circular obstacles. It shares no code with the planner above.

```bash
npm run dev            # http://localhost:5173/browser-flow-matching/pusht/
npm run check:pusht    # stability, replay determinism, recording pipeline
npm run bench:pusht    # scripted-expert success rate
npm run check:rl       # PPO/SAC environment and optimizer smoke checks
npm run train:ppo -- 200000
npm run train:sac -- 200000
```

## Physics

Inertia is dropped. Under the ellipsoidal limit-surface approximation the block's motion is a linear
function of the applied wrench — velocity proportional to force, angular rate proportional to torque
over `c²` — where `c² = (1/A)∫r²dA` is read directly off the geometry rather than tuned. Contacts are
solved as relaxed Gauss-Seidel projections, so large timesteps stay stable because there is no momentum
to integrate. Pusher contact carries Coulomb friction, which is what lets a sideways sweep turn the tee
instead of sliding off it.

- Pusher: position-controlled circle, radius `0.022`, capped at `0.017` world units per step
- Block: 4u × 1u bar with a 1u × 3u stem, `u = 0.05`, area `0.0175`, `c² = 0.005366`
- Action space: absolute pusher target position, matching the diffusion-policy Push-T convention; the
  lift flag is temporarily forced off for the current policy ablation
- Observation: 11 values (pusher, block pose as cos/sin, goal pose, inactive lift) plus 3 per obstacle — 11 currently
- Success: 90% coverage of the goal footprint, by fixed-sample membership test
- The block starts in the lower-left and the goal in the upper-right, both fixed at 0° orientation. The
  expert requests translation directly toward the goal without an orient-first phase.
- The simulator supports obstacles in the corridor between start and goal, but the current UI fixes their
  count at zero while the learned policy is brought up on the simpler task.

Throughput is roughly 220k sim steps per second in Node. A ten-episode collection costs about 180 ms
of blocking main-thread work (279 ms if every episode runs to the step cap) — a visible hitch, not a freeze.
Snapshot-and-replay is bit-exact, so recorded episodes are reproducible.

## Scripted expert

A push at surface point `p` with inward direction `f` moves the block proportionally to `f` and spins it
proportionally to `(r × f) / c²`. That is a linear map from contact choice to block motion, so the expert
scores every candidate contact against the motion it currently wants, minus the cost of travelling there.
The travel term matters more than it looks: without it the pusher spends two thirds of the episode
orbiting the block to chase marginally better contacts.

Two routes are planned on a grid (`plan.js`), because obstacles sitting in the corridor break purely local
behaviour. The **block** follows a planned path to the goal — without one it wedges against an obstacle
while the straight line to the goal points back through it, and the expert oscillates indefinitely. The
**pusher** falls back to a planned route when it cannot walk its orbit around the block; with one obstacle
the orbit is fully clear only 36% of the time. Orientation is down-weighted until the block is near the end
of its route, since correcting it mid-transit drives the tee back into whatever it is going around.

The route around the next obstacle is a free choice, so it is planned both ways and picked at random when
the two cost within 1.25x of each other — see below. The committed side is latched until that obstacle is
passed; resampling it on every replan would make the block dither between the two routes and produce
neither cleanly.

The current zero-obstacle diagonal task keeps lift disabled and removes the expert's orient-first phase.

## Reinforcement learning baselines

The `rl-policy` branch includes two dependency-free continuous-control baselines. PPO is on-policy with
GAE and a clipped objective; SAC uses a replay buffer, twin Q-functions, target networks, and a squashed
Gaussian actor. Both deliberately use the same minimal task definition:

- observation: one normalized 11-value simulator frame, with no history or frame stack
- action: normalized `dx, dy` in `[-1, 1]`, scaled to one `0.017` pusher step
- reward: change in block-to-goal center distance (`previous - current`), `+1` on completion, or `-1` and termination when the pusher touches the outer wall
- episode horizon: 600 steps by default

Training is seeded and runs in Node. Set `OUT=policy.json` to save the actor and its evaluation metadata.
Distance shaping is potential-based: progress earns a small positive reward, moving away earns the matching
negative reward, and waiting earns zero. Completion counts remain explicit in the CLI output.

## Multimodality

To move the block one way the pusher must stand on the opposite side, so relocating means travelling
around the block — clockwise or counter-clockwise, equally valid whenever both arcs are clear. The expert
flips a coin there. From an identical observation the two branches then diverge by more than the block's
own circumradius, and a unimodal regression policy averages those two modes and drives straight into the
block.

An obstacle rules out one way *around the block*, so on its own it suppresses that choice: the orbit
coin flip fires 76% of the time in an empty arena but only 35% with one obstacle. What restores it, and
more, is branching the **block's** route — which side to push the tee past the obstacle. That decision is
coarser and lasts hundreds of steps rather than tens.

| obstacles | 0 | 1 | 2 |
|---|---|---|---|
| states with a wide branch split | 52% | **65%** | 61% |
| mean peak action separation | 0.167 | **0.214** | 0.195 |

Obstacles now *increase* multimodality rather than suppressing it. This depends entirely on the obstacle
sitting near the corridor centre line, because that is what makes the two ways round equally good:

| obstacle lateral offset | 0.00 | 0.04 (current) | 0.09 | 0.14 (previous) |
|---|---|---|---|---|
| median cost ratio between the two routes | 1.01 | ~1.15 | 1.37 | 1.57 |
| layouts with both routes within 1.10x | 100% | 56% | 0% | 0% |

Position *along* the corridor does not matter — jittering it leaves the ratios unchanged — so layout
variety costs nothing. Wall proximity turns out to be a red herring: room-to-wall stays asymmetric at
~1.45 regardless, but the detour around the obstacle dominates path cost.

`node tools/multimodality.mjs` reproduces the measurement (`OBSTACLES=n` to vary the count). The **Orbit
tie-break** control on the page pins the expert to one branch, for collecting a deliberately unimodal
comparison set.

## Demonstrations

Both sources write the same format. **Collect** runs the scripted expert headlessly and resamples any
episode it fails, so every collected episode is a solved one; **Teleop** mode lets
you steer the pusher with the cursor and records what you do, so the multimodality comes from your own
inconsistency rather than a coin flip. Episodes persist to `localStorage` at 4-decimal precision
(~110 bytes per transition, so roughly 45 episodes fit the quota) and export as JSON. `toDataset()`
flattens the set into observation and action matrices with episode boundaries, ready for action chunking.

Observation width is `10 + 3 x obstacles`, so episodes recorded at different slider settings are different
shapes. `toDataset()` selects one width — the most common, or whatever `observationSize` is passed — and
reports the rest as `excluded` rather than running them together, which would otherwise pad the short rows
with zeros and shift every field past the pusher position into the wrong column. Data restored from
storage is validated on the way in, so one malformed episode cannot take the page down on first render.

## Demo viewer (`/pusht/viewer/`)

Replays any recorded demonstration with the cached policy rolled out from the
identical starting pose, so the step where the two part company is visible.
Expert pusher path solid, policy dashed, the expert's block pose behind the
policy's as a ghost, with a scrubber over the longer of the two.

It reads the demonstrations and the policy out of the same browser storage the
main page writes, so it is a view onto that data rather than a second copy of
the training machinery. Recorded poses come straight from the stored
observations, which reconstruct the simulator's state to 7e-5 -- the four-decimal
rounding applied on record. **Re-run policy** draws a fresh sample from the same
distribution, which is worth doing a few times: the spread between draws is the
multimodality, and a policy that lands somewhere different each time is behaving
as intended rather than malfunctioning.

## Tools

```
tools/sim-check.mjs        solver stability and throughput
tools/replay-check.mjs     snapshot/restore determinism
tools/demo-check.mjs       recording pipeline and dataset shape
tools/expert-bench.mjs     success rate      [episodes] [obstacles] [horizon], SEED=n
tools/layout-check.mjs     obstacle placement relative to the corridor
tools/failure-check.mjs    breakdown of how failed episodes failed
tools/multimodality.mjs    branch divergence  OBSTACLES=n
tools/tune.mjs             weight grid search
```

In Policy mode the sampling itself is animated. The pusher holds still while a
batch of twelve candidate action chunks is transported from Gaussian noise to
trajectories over ten Euler steps. Mid-transport they are drawn as point clouds,
because that is what they are; once converged they become polylines, and the
committed one is the only trajectory shown while it executes for sixteen steps.
Then the cycle repeats. Watching the spread is the point — it
is the distribution the method exists to represent, which a regression policy
could not show at all.

A trained policy is cached in the browser alongside the demonstrations, so a
reload does not mean retraining. Weights are stored as base64 of the raw
Float32Array buffer -- 225 KB for 43k parameters, against roughly 850 KB written
as decimal JSON, and they share a quota with the demos. Restoring reproduces
sampled trajectories exactly. **Clear policy** discards it.

Training runs in a Web Worker and streams weight snapshots back about 25 times
over a run, so Policy mode is live from the first snapshot and the policy can be
watched improving while it trains. The snapshot is a flat `Float32Array` handed
over as a transferable: 0.019 ms against a ~20 ms training step, where the
`toJSON` path would have cost 2.87 ms. Inference is on the main thread and costs
ten single-sample forward passes every 16 simulation steps, so neither side
slows the other.

Within each chunk, xy commands are learned as first differences: the first is
relative to the current pusher and each later command is relative to the one
before it. Sampling cumulatively reconstructs the simulator's absolute targets.
The reserved lift channel is held off during the current ablation; keeping its
slot makes the experiment easy to reverse without another action-schema change.

`ScriptedExpert` takes `routeCostGate: 0` to disable route branching (always take the shorter way round),
which is how the branching cost above was isolated.

## Known limitations

Deliberate, and measured rather than assumed. Each would change the tuned numbers above, so none is a
quiet fix:

- **The pusher penetrates the tee.** `advancePusher` collides the pusher with obstacles and walls but not
  with the block, and `pushTarget` aims deliberately past the contact, so the pusher centre is strictly
  inside a tee part on ~7% of expert steps. Penetration depth *is* the force model here, so this cannot
  simply be clamped out; but once the centre is inside, `circleRectContact` ejects along the nearest face
  with no memory of the entry face, and the normal can flip. Measured: max depth 0.046, and the normal
  opposes the pusher's travel on 6.8% of deep contacts against 3.5% of shallow ones.
- **Obstacle friction uses a step-lagged slip.** `this.previous` refreshes once per `step()`, not per
  substep, so the first solver iteration of each step sees zero slip and later ones see the whole step's
  motion. The Coulomb clamp bounds it, so it cannot diverge, but it ties obstacle friction to
  `SUBSTEPS` and `SOLVER_ITERATIONS`.
- **Contacts within one solver call are detected at a stale pose.** The pusher is transformed into the
  block frame once, before the loop over parts, while `applyWrench` moves the block inside it — Jacobi
  detection with Gauss-Seidel application. At the tee's inner corners (~4.5% of steps touch both parts)
  this roughly doubles local stiffness. Bounded and deterministic; replay stays bit-exact.
- **Path shortcutting is skipped at force-opened endpoints.** When the block starts inside inflated
  geometry, `clear()` re-tests the anchor and fails, so the grid staircase survives for that one leg.
  Costs an extra waypoint or two, never an invalid path.

---

# Flow core (`src/flow/`)

Shared, batched flow-matching primitives, extracted from the planner demo so the
pushing policy does not end up with a second copy that drifts.

```bash
npm run check:flow   # gradient check, then a bimodal end-to-end test
npm run bench:flow   # batched against the per-sample loop
```

- `mlp.js` — batched MLP, arbitrary depth, tanh hidden layers and a linear output,
  row-major typed arrays, no allocation after construction, JSON save/load
- `adam.js` — Adam with bias correction over a list of parameter buffers
- `features.js` — Fourier feature encoding, expressed as data rather than inlined
- `flow.js` — the probability path, the time schedule, the training step, Euler sampling

Correctness is two independent checks (`tools/flow-gradcheck.mjs`). The primary one
compares every gradient against a naive Float64 reference — no unrolling, no shared
code — over seven shapes including no hidden layer, width-1 layers, deeper nets and
widths that are not multiples of the unroll factor; all land within 1.2% of a
combined `1e-6 + 1e-4|g|` tolerance. Finite differences then validate the reference
itself on the shallow shapes, where float32 FD noise is small enough to trust; it
grows with depth, which is why it cannot be the primary check.
End to end (`tools/flow-check.mjs`) the core transports
uniform 2D samples onto two separated clusters and populates **both** — a 48/52
split with 3% strays — which is the property a mean-regressing model cannot show.

## Batching is not a speedup in JavaScript

Worth stating plainly, because the opposite is widely assumed and it was the
premise this extraction started from. At `20->64->64->2`:

| | samples/s | relative |
|---|---|---|
| per-sample loop | 92,600 | 1.00x |
| batched, batch 96 / 256 / 1024 | ~83,700 | **0.90x** |

The batched version is ~10% *slower*, flat across batch size. The per-sample code
it replaced was already allocation-free over flat typed arrays with a good loop
order, so there was no overhead to amortise, and batching only adds offset
arithmetic and activation traffic. Four-accumulator unrolling of the inner dot
product recovers about 1.25x on the matmul in isolation (0.96 -> 1.19 Gmac/s) and
is applied, but does not close the gap.

The speedup people associate with batching comes from dispatching to BLAS — cache
blocking and SIMD. Plain JS loops get neither. Batching here buys a shared API and
the precondition for WebGPU, not throughput.

## Training budget

Measured per optimizer step, batch 256:

| shape | ms/step | params | steps in 10s |
|---|---|---|---|
| `20->64->64->2` (planner, batch 96) | 1.14 | 5.6k | 8,700 |
| `48->128->128->16` | 12.0 | 25k | 830 |
| `48->256->256->16` | 40.1 | 82k | 250 |
| `48->512->512->16` | 146.6 | 296k | 68 |

A policy-sized network trains at hundreds of steps per ten seconds, not thousands.
Behaviour cloning wants thousands, so live in-browser training of the pushing
policy is a one-to-two minute job at `256` wide — which is a constraint on how the
demo is framed, not a detail.
