// Conditional flow matching.
//
// The model learns a velocity field that transports a source distribution onto
// a target one. Training pairs an independent sample from each, picks a time
// along the straight path between them, and regresses the model's velocity at
// that point onto the displacement:
//
//   x_t = (1 - t) * x0 + t * x1        target velocity = x1 - x0
//
// Sampling then integrates that field from a fresh source sample. Encoding of
// (x_t, t, conditioning) into model inputs belongs to the caller, because what
// conditions the field differs per task; everything here is shared.

// Half the samples uniform, half pushed toward t = 1. The terminal
// distribution is the thin one, and it is where error is most visible, so it
// gets the extra supervision.
export function sampleTime(random) {
  const value = random() < 0.5 ? random() : 1 - random() ** 2;
  return 0.01 + value * 0.98;
}

export function interpolate(source, target, time, out, dimension) {
  for (let index = 0; index < dimension; index++) {
    out[index] = source[index] * (1 - time) + target[index] * time;
  }
  return out;
}

export function velocityTarget(source, target, out, dimension) {
  for (let index = 0; index < dimension; index++) out[index] = target[index] - source[index];
  return out;
}

// Owns the loss, the backward pass and the optimizer step. The caller fills
// model.inputBuffer() with the encoded batch and supplies matching velocity
// targets; this keeps the trainer indifferent to how a task is conditioned.
export class FlowTrainer {
  constructor({ model, optimizer }) {
    this.model = model;
    this.optimizer = optimizer;
    this.gradOutput = new Float32Array(model.maxBatch * model.outputWidth);
  }

  // targets: [sample * outputWidth + component]. Returns mean loss per sample.
  step(targets, batch) {
    const model = this.model;
    const width = model.outputWidth;
    model.zeroGrad();
    const out = model.forward(batch);

    let loss = 0;
    for (let index = 0; index < batch * width; index++) {
      const error = out[index] - targets[index];
      loss += 0.5 * error * error;
      this.gradOutput[index] = error / batch;
    }

    model.backward(this.gradOutput, batch);
    this.optimizer.step(model.grads);
    return loss / batch;
  }
}

// Integrates the velocity field from `state` over `steps` uniform Euler steps.
// `velocityAt(state, time, out)` is supplied by the caller so this works for a
// single sample or a batch encoded however the task needs.
export function eulerIntegrate(velocityAt, state, steps, dimension, scratch) {
  const velocity = scratch ?? new Float32Array(dimension);
  const delta = 1 / steps;
  for (let step = 0; step < steps; step++) {
    velocityAt(state, step * delta, velocity);
    for (let index = 0; index < dimension; index++) state[index] += velocity[index] * delta;
  }
  return state;
}
