// Tiny deterministic PRNG so the mock trail is identical on every reload.
// (No real randomness — this is a design demo, and a stable map is nicer to design against.)

export function mulberry32(seed) {
  let a = seed >>> 0
  return function () {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Cheap string hash → 32-bit seed, so a track's features follow from its title.
export function hashString(str) {
  let h = 2166136261 >>> 0
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

export const lerp = (a, b, t) => a + (b - a) * t
export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))
export const clamp01 = (v) => clamp(v, 0, 1)
