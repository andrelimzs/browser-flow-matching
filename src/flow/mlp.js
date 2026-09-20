// Batched multilayer perceptron over typed arrays.
//
// Extracted from the per-sample network the planner demo grew up with. The
// shape is the same — tanh hidden layers, a linear output — but every pass
// handles a whole batch, which is what makes the cost per sample reasonable
// and is a precondition for ever moving this to WebGPU.
//
// Layout is row-major throughout: weights[l][j * inputs + i] is the weight from
// input i to unit j, and activations[l][sample * size + unit]. There are no
// allocations after construction.

export function xavier(values, fanIn, fanOut, random) {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  for (let index = 0; index < values.length; index++) values[index] = (random() * 2 - 1) * limit;
  return values;
}

export class MLP {
  // sizes: [inputWidth, hidden..., outputWidth]. Hidden layers are tanh, the
  // output layer is linear.
  constructor({ sizes, maxBatch, random = Math.random }) {
    if (sizes.length < 2) throw new Error("MLP needs at least an input and an output size");
    this.sizes = sizes.slice();
    this.maxBatch = maxBatch;
    this.layers = sizes.length - 1;

    this.weights = [];
    this.biases = [];
    this.weightGrads = [];
    this.biasGrads = [];
    for (let layer = 0; layer < this.layers; layer++) {
      const inputs = sizes[layer];
      const outputs = sizes[layer + 1];
      this.weights.push(xavier(new Float32Array(outputs * inputs), inputs, outputs, random));
      this.biases.push(new Float32Array(outputs));
      this.weightGrads.push(new Float32Array(outputs * inputs));
      this.biasGrads.push(new Float32Array(outputs));
    }

    // activations[0] is the input; activations[l] is the output of layer l-1.
    this.activations = sizes.map((size) => new Float32Array(maxBatch * size));
    // deltas[l] is dLoss/dz for layer l.
    this.deltas = [];
    for (let layer = 0; layer < this.layers; layer++) {
      this.deltas.push(new Float32Array(maxBatch * sizes[layer + 1]));
    }

    this.batch = 0;
    this.outputViews = new Map();
    this.params = [];
    this.grads = [];
    for (let layer = 0; layer < this.layers; layer++) {
      this.params.push(this.weights[layer], this.biases[layer]);
      this.grads.push(this.weightGrads[layer], this.biasGrads[layer]);
    }
  }

  get inputWidth() {
    return this.sizes[0];
  }

  get outputWidth() {
    return this.sizes[this.sizes.length - 1];
  }

  // The buffer callers write a batch of inputs into.
  inputBuffer() {
    return this.activations[0];
  }

  // Runs the batch and returns the output activations, laid out
  // [sample * outputWidth + unit]. Valid until the next forward pass.
  //
  // The returned buffer is the full maxBatch-sized one, so anything past
  // `batch * outputWidth` is left over from an earlier, larger pass. Callers
  // that iterate must bound by the batch, or use outputs() below.
  forward(batch) {
    if (!Number.isInteger(batch) || batch < 1) throw new Error(`batch must be a positive integer, got ${batch}`);
    if (batch > this.maxBatch) throw new Error(`batch ${batch} exceeds maxBatch ${this.maxBatch}`);
    this.batch = batch;
    for (let layer = 0; layer < this.layers; layer++) {
      const inputs = this.sizes[layer];
      const outputs = this.sizes[layer + 1];
      const weights = this.weights[layer];
      const biases = this.biases[layer];
      const source = this.activations[layer];
      const destination = this.activations[layer + 1];
      const isOutput = layer === this.layers - 1;

      for (let sample = 0; sample < batch; sample++) {
        const sourceOffset = sample * inputs;
        const destinationOffset = sample * outputs;
        for (let unit = 0; unit < outputs; unit++) {
          const weightOffset = unit * inputs;
          // Four accumulators: this is the one shape of unrolling the engine
          // does not do itself, and it is worth about 1.25x on this loop.
          let a = biases[unit];
          let b = 0;
          let c = 0;
          let d = 0;
          let index = 0;
          for (; index + 3 < inputs; index += 4) {
            a += weights[weightOffset + index] * source[sourceOffset + index];
            b += weights[weightOffset + index + 1] * source[sourceOffset + index + 1];
            c += weights[weightOffset + index + 2] * source[sourceOffset + index + 2];
            d += weights[weightOffset + index + 3] * source[sourceOffset + index + 3];
          }
          let sum = a + b + c + d;
          for (; index < inputs; index++) sum += weights[weightOffset + index] * source[sourceOffset + index];
          destination[destinationOffset + unit] = isOutput ? sum : Math.tanh(sum);
        }
      }
    }
    return this.activations[this.layers];
  }

  // The valid region of the last forward pass. Views are cached per batch size,
  // so this does not allocate in a steady-state loop.
  outputs(batch = this.batch) {
    let view = this.outputViews.get(batch);
    if (!view) {
      view = this.activations[this.layers].subarray(0, batch * this.outputWidth);
      this.outputViews.set(batch, view);
    }
    return view;
  }

  zeroGrad() {
    for (const gradient of this.grads) gradient.fill(0);
  }

  // gradOutput is dLoss/dOutput for the batch, same layout as forward's result.
  // Gradients accumulate, so callers zeroGrad() when they mean to start fresh.
  backward(gradOutput, batch) {
    const last = this.layers - 1;
    const needed = batch * this.sizes[this.layers];
    // subarray clamps rather than throwing, so an undersized gradOutput would
    // otherwise write a prefix and leave the rest of deltas holding whatever
    // the previous backward pass put there — wrong gradients, no error.
    if (gradOutput.length < needed) {
      throw new Error(`gradOutput has ${gradOutput.length} entries, need ${needed} for batch ${batch}`);
    }
    if (batch !== this.batch) {
      throw new Error(`backward batch ${batch} does not match the last forward batch ${this.batch}`);
    }
    this.deltas[last].set(gradOutput.subarray(0, needed));

    for (let layer = last; layer >= 0; layer--) {
      const inputs = this.sizes[layer];
      const outputs = this.sizes[layer + 1];
      const delta = this.deltas[layer];
      const source = this.activations[layer];
      const weightGrad = this.weightGrads[layer];
      const biasGrad = this.biasGrads[layer];

      for (let sample = 0; sample < batch; sample++) {
        const deltaOffset = sample * outputs;
        const sourceOffset = sample * inputs;
        for (let unit = 0; unit < outputs; unit++) {
          const value = delta[deltaOffset + unit];
          if (value === 0) continue;
          biasGrad[unit] += value;
          const weightOffset = unit * inputs;
          let index = 0;
          for (; index + 3 < inputs; index += 4) {
            weightGrad[weightOffset + index] += value * source[sourceOffset + index];
            weightGrad[weightOffset + index + 1] += value * source[sourceOffset + index + 1];
            weightGrad[weightOffset + index + 2] += value * source[sourceOffset + index + 2];
            weightGrad[weightOffset + index + 3] += value * source[sourceOffset + index + 3];
          }
          for (; index < inputs; index++) weightGrad[weightOffset + index] += value * source[sourceOffset + index];
        }
      }

      if (layer === 0) break;

      // Propagate into the previous layer and through its tanh.
      const weights = this.weights[layer];
      const previous = this.deltas[layer - 1];
      previous.fill(0, 0, batch * inputs);
      for (let sample = 0; sample < batch; sample++) {
        const deltaOffset = sample * outputs;
        const previousOffset = sample * inputs;
        for (let unit = 0; unit < outputs; unit++) {
          const value = delta[deltaOffset + unit];
          if (value === 0) continue;
          const weightOffset = unit * inputs;
          let index = 0;
          for (; index + 3 < inputs; index += 4) {
            previous[previousOffset + index] += weights[weightOffset + index] * value;
            previous[previousOffset + index + 1] += weights[weightOffset + index + 1] * value;
            previous[previousOffset + index + 2] += weights[weightOffset + index + 2] * value;
            previous[previousOffset + index + 3] += weights[weightOffset + index + 3] * value;
          }
          for (; index < inputs; index++) previous[previousOffset + index] += weights[weightOffset + index] * value;
        }
        for (let index = 0; index < inputs; index++) {
          const activation = source[previousOffset + index];
          previous[previousOffset + index] *= 1 - activation * activation;
        }
      }
    }
  }

  toJSON() {
    return {
      sizes: this.sizes,
      weights: this.weights.map((values) => Array.from(values)),
      biases: this.biases.map((values) => Array.from(values)),
    };
  }

  loadJSON(state) {
    if (String(state.sizes) !== String(this.sizes)) {
      throw new Error(`shape mismatch: have ${this.sizes}, got ${state.sizes}`);
    }
    state.weights.forEach((values, layer) => this.weights[layer].set(values));
    state.biases.forEach((values, layer) => this.biases[layer].set(values));
    return this;
  }
}
