// Persists a trained policy in the browser, so a reload does not mean retraining.
//
// Weights go in as base64 of the raw Float32Array buffer rather than as JSON
// numbers: 43k parameters are 172 KB binary and about 230 KB base64, against
// roughly 390 KB written as decimal text, and they share a storage quota with
// the demonstrations.

const STORAGE_KEY = "pusht-policy-v1";
// Version 2 changed xy chunks to deltas, version 3 fixed the task distribution,
// version 4 adopted smooth targets, and version 5 adopts the staged expert.
// Earlier weights are not a valid cached policy for the current data.
const VERSION = 5;

function toBase64(values) {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let binary = "";
  const CHUNK = 0x8000; // chunked, or fromCharCode.apply blows the stack
  for (let index = 0; index < bytes.length; index += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(index, index + CHUNK));
  }
  return btoa(binary);
}

function fromBase64(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
  return new Float32Array(bytes.buffer);
}

// Returns null on success, or a reason it could not be saved.
export function savePolicy({ weights, sizes, scales, observationSize, obstacleCount, loss, steps }) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
      version: VERSION,
      sizes,
      scales: Array.from(scales),
      observationSize,
      obstacleCount,
      loss,
      steps,
      savedAt: Date.now(),
      weights: toBase64(weights),
    }));
    return null;
  } catch (error) {
    const quota = error instanceof DOMException &&
      (error.name === "QuotaExceededError" || error.name === "NS_ERROR_DOM_QUOTA_REACHED");
    return quota
      ? "Browser storage is full, so the trained policy was not cached — clearing demos frees room."
      : "Browser storage is unavailable, so the trained policy was not cached.";
  }
}

// Validated on the way out: a malformed entry must not take the page down
// during its first render.
export function loadPolicy() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed?.version !== VERSION) return null;
    if (!Array.isArray(parsed.sizes) || !Array.isArray(parsed.scales)) return null;
    if (typeof parsed.weights !== "string" || typeof parsed.observationSize !== "number") return null;
    const weights = fromBase64(parsed.weights);
    const expected = parsed.sizes.slice(0, -1).reduce(
      (total, size, index) => total + size * parsed.sizes[index + 1] + parsed.sizes[index + 1],
      0,
    );
    if (weights.length !== expected) return null;
    if (!weights.every(Number.isFinite)) return null;
    return {
      weights,
      sizes: parsed.sizes,
      scales: Float32Array.from(parsed.scales),
      observationSize: parsed.observationSize,
      obstacleCount: parsed.obstacleCount ?? 0,
      loss: parsed.loss,
      steps: parsed.steps,
      savedAt: parsed.savedAt,
    };
  } catch {
    return null;
  }
}

export function clearPolicy() {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Nothing to do: it was not readable either.
  }
}
