// Fourier feature encoding.
//
// A tanh MLP of this size cannot resolve sharp spatial structure from raw
// coordinates alone, so selected inputs are lifted onto sin/cos pairs at
// several frequencies. The planner demo hard-coded one such layout inline;
// this expresses the same idea as data, because the pushing policy needs a
// different one over a different number of inputs.

// bands: [{ index, frequencies: [1, 2, 4] }], where a frequency f contributes
// sin(pi * f * value) and cos(pi * f * value). Raw values are passed through
// first, so the encoding is strictly additive.
export function makeFourierEncoder({ inputWidth, bands = [] }) {
  for (const band of bands) {
    if (band.index < 0 || band.index >= inputWidth) {
      throw new Error(`band index ${band.index} outside input width ${inputWidth}`);
    }
  }
  const extra = bands.reduce((total, band) => total + band.frequencies.length * 2, 0);
  const size = inputWidth + extra;

  function encode(raw, out, offset = 0) {
    for (let index = 0; index < inputWidth; index++) out[offset + index] = raw[index];
    let cursor = offset + inputWidth;
    for (const band of bands) {
      const value = raw[band.index];
      for (const frequency of band.frequencies) {
        const angle = Math.PI * frequency * value;
        out[cursor++] = Math.sin(angle);
        out[cursor++] = Math.cos(angle);
      }
    }
    return out;
  }

  function encodeBatch(raw, out, batch) {
    for (let sample = 0; sample < batch; sample++) {
      encode(raw.subarray(sample * inputWidth, (sample + 1) * inputWidth), out, sample * size);
    }
    return out;
  }

  return { size, inputWidth, bands, encode, encodeBatch };
}
