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
    // A misspelled key leaves index undefined, which passes both range
    // comparisons, reads undefined out of the raw vector and writes NaN into
    // the model input. Nothing downstream throws: the NaN reaches Adam's
    // moments and every weight is NaN one step later, reported as a NaN loss
    // rather than an error. So the type is checked, not just the range.
    if (!Number.isInteger(band.index) || band.index < 0 || band.index >= inputWidth) {
      throw new Error(`band index ${band.index} is not an integer in [0, ${inputWidth})`);
    }
    if (!Array.isArray(band.frequencies) || band.frequencies.length === 0) {
      throw new Error(`band at index ${band.index} needs a non-empty frequencies array`);
    }
    if (!band.frequencies.every(Number.isFinite)) {
      throw new Error(`band at index ${band.index} has a non-finite frequency`);
    }
  }
  const extra = bands.reduce((total, band) => total + band.frequencies.length * 2, 0);
  const size = inputWidth + extra;

  function encode(raw, out, offset = 0, rawOffset = 0) {
    for (let index = 0; index < inputWidth; index++) out[offset + index] = raw[rawOffset + index];
    let cursor = offset + inputWidth;
    for (const band of bands) {
      const value = raw[rawOffset + band.index];
      for (const frequency of band.frequencies) {
        const angle = Math.PI * frequency * value;
        out[cursor++] = Math.sin(angle);
        out[cursor++] = Math.cos(angle);
      }
    }
    return out;
  }

  // Takes a raw offset rather than slicing: subarray allocates a view per call,
  // which in a render loop is exactly the jank the no-allocation rule exists to
  // avoid.
  function encodeBatch(raw, out, batch) {
    for (let sample = 0; sample < batch; sample++) {
      encode(raw, out, sample * size, sample * inputWidth);
    }
    return out;
  }

  return { size, inputWidth, bands, encode, encodeBatch };
}
