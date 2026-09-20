# Flow Planner

An interactive, dependency-light robotics flow-matching demo that trains a tiny MLP entirely in the browser. A clearance-aware A* planner generates a safe reference trajectory through a walled obstacle field, and the model learns to transform a uniform free-space distribution into the complete path distribution.

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:5173`.

## How it works

- World: fixed bottom-left and top-right route endpoints, a perimeter wall, and exactly 3 widely spaced circular obstacles
- Planner: 8-connected A* with a logarithmic clearance barrier around obstacles and walls, followed by barrier-aware path shortcutting; the centerline reserves the target ribbon’s width beyond the displayed safety envelope
- Source distribution: uniform samples over collision-free workspace positions
- Target distribution: the ordered curve `p(s) = (x(s), y(s))`, with `s` sampled uniformly along the complete A* route
- Coupling: independent free-space/path samples by default; optional progress mode ranks sources from start-nearest to goal-nearest and couples them to increasing `s`
- Probability path: `x_τ = (1 - τ)x_0 + τp(s)`
- Target velocity: `p(s) - x_0`
- Conditioning: flow time `τ` by default; optional path progress `s` is held fixed while `τ` advances from 0 to 1
- Model: `20 → 64 → 64 → 2` tanh MLP with multiscale spatial, flow-time, and path-progress Fourier features
- Time sampling: half uniform, half biased toward the thin terminal path distribution
- Optimizer: Adam, implemented with typed arrays
- Sampling: Euler integration with shared, configurable 1, 2, 4, 10, or 20-step paths for animation and scrubbing; playback shows only real solver states

There are no ML runtime dependencies. Planning, training, backpropagation, optimization, and inference are implemented directly in `src/main.js`.
