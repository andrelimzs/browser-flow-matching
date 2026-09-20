# Flow Planner

An interactive, dependency-light robotics flow-matching demo that trains a tiny MLP entirely in the browser. A clearance-aware A* planner generates a safe reference trajectory through a walled obstacle field, and the model learns to transform a uniform free-space distribution into the complete path distribution.

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:5173`.

## How it works

- World: fixed bottom-left and top-right route endpoints, a perimeter wall, and 3–4 widely spaced circular obstacles
- Planner: 8-connected A* on an obstacle-inflated grid, followed by collision-checked path shortcutting
- Source distribution: uniform samples over collision-free workspace positions
- Target distribution: uniform arc-length samples along the complete A* route
- Pairing: exact distribution-agnostic minibatch optimal transport
- Probability path: `x_t = (1 - t)x_uniform + tx_path`
- Target velocity: `x_path - x_uniform`
- Model: `17 → 64 → 64 → 2` tanh MLP with general multiscale spatial and temporal Fourier features
- Time sampling: half uniform, half biased toward the thin terminal path distribution
- Optimizer: Adam, implemented with typed arrays
- Sampling: Euler integration with shared, configurable 1, 2, 4, 10, or 20-step paths for animation and scrubbing; playback shows only real solver states

There are no ML runtime dependencies. Planning, training, backpropagation, optimization, and inference are implemented directly in `src/main.js`.
