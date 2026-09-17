# Flow Playground

An interactive, dependency-light flow-matching demo that trains a tiny MLP entirely in the browser. The model learns a time-dependent 2D velocity field from Gaussian noise to a checkerboard distribution, while the UI visualizes particles, the learned field, and training loss live.

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:5173`.

## How it works

- Source: 2D Gaussian samples
- Target: samples from alternating cells of a 4 × 4 checkerboard
- Interpolant: `x_t = (1 - t)x_0 + tx_1`
- Target velocity: `x_1 - x_0`
- Model: `3 → 64 → 64 → 2` tanh MLP
- Optimizer: Adam, implemented with typed arrays
- Sampling: Euler integration through the learned velocity field

There are no ML runtime dependencies. Training, backpropagation, optimization, and inference are implemented directly in `src/main.js`, which makes the target sampler easy to replace with a drawn point distribution later.
