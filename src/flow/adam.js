// Adam over a list of typed-array parameter buffers.
//
// Bias correction is applied to both moments, matching the reference algorithm;
// the planner demo's inline version did the same thing with the constants
// spelled out.

export class Adam {
  constructor(params, options = {}) {
    this.params = params;
    this.learningRate = options.learningRate ?? 0.002;
    this.beta1 = options.beta1 ?? 0.9;
    this.beta2 = options.beta2 ?? 0.999;
    this.epsilon = options.epsilon ?? 1e-8;
    this.moments = params.map((values) => new Float32Array(values.length));
    this.velocities = params.map((values) => new Float32Array(values.length));
    this.steps = 0;
  }

  step(grads) {
    this.steps += 1;
    const firstCorrection = 1 - this.beta1 ** this.steps;
    const secondCorrection = 1 - this.beta2 ** this.steps;
    for (let index = 0; index < this.params.length; index++) {
      const values = this.params[index];
      const moments = this.moments[index];
      const velocities = this.velocities[index];
      const gradients = grads[index];
      // A short gradient buffer reads undefined, which puts NaN into the moment
      // and velocity for that slot permanently — later correct gradients never
      // recover it.
      if (gradients.length !== values.length) {
        throw new Error(`gradient buffer ${index} has ${gradients.length} entries, expected ${values.length}`);
      }
      for (let i = 0; i < values.length; i++) {
        const gradient = gradients[i];
        moments[i] = this.beta1 * moments[i] + (1 - this.beta1) * gradient;
        velocities[i] = this.beta2 * velocities[i] + (1 - this.beta2) * gradient * gradient;
        const first = moments[i] / firstCorrection;
        const second = velocities[i] / secondCorrection;
        values[i] -= this.learningRate * first / (Math.sqrt(second) + this.epsilon);
      }
    }
  }

  reset() {
    for (const buffer of this.moments) buffer.fill(0);
    for (const buffer of this.velocities) buffer.fill(0);
    this.steps = 0;
  }
}
