# Flow Planner

An interactive, dependency-light robotics flow-matching demo that trains a tiny MLP entirely in the browser. A clearance-aware A* planner generates a safe reference trajectory through random obstacle fields, and the model learns its time-dependent 2D velocity field live.

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:5173`.

## How it works

- World: fixed bottom-left start and top-right goal regions with 3–4 random circular obstacles
- Expert: 8-connected A* on an obstacle-inflated grid, followed by collision-checked path shortcutting
- Probability path: the safe reference route plus interpolated start/goal offsets
- Target velocity: the local route derivative plus the offset derivative
- Model: `17 → 64 → 64 → 2` tanh MLP with general multiscale spatial and temporal Fourier features
- Time sampling: uniform along the reference trajectory
- Optimizer: Adam, implemented with typed arrays
- Sampling: Euler integration with shared, configurable 1, 2, 4, 10, or 20-step paths for animation and scrubbing; playback shows only real solver states

There are no ML runtime dependencies. Planning, training, backpropagation, optimization, and inference are implemented directly in `src/main.js`.
