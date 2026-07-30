// ─────────────────────────────────────────────────────────────────────────────
// Size-independent terrain data, computed ONCE in normalized 0..1 map space and
// cached. TerrainCanvas just projects these into pixels and paints — so resizing
// the window is cheap (no field recompute). Everything here is deterministic.
// ─────────────────────────────────────────────────────────────────────────────
import { altitudeAt } from '../utils/metaphor.js'
import { generateContours } from '../utils/terrain.js'
import { TRACKS, forestCount } from '../data/tracks.js'
import { mulberry32, lerp } from '../utils/rng.js'

const PAD = 0.06

// Relief field: altitude sampled on a grid, plus min/max for normalization.
function buildField(gx, gy) {
  const values = new Float32Array(gx * gy)
  let min = Infinity
  let max = -Infinity
  for (let j = 0; j < gy; j++) {
    for (let i = 0; i < gx; i++) {
      const x = lerp(-PAD, 1 + PAD, i / (gx - 1))
      const y = lerp(-PAD, 1 + PAD, j / (gy - 1))
      const v = altitudeAt(x, y)
      values[j * gx + i] = v
      if (v < min) min = v
      if (v > max) max = v
    }
  }
  return { gx, gy, values, min, max }
}

// Scatter groove-forest glyphs around each place. A percussive festival banger
// (dense grooveGrid) sits in thick woods; a sparse vocal cut in open meadow.
function buildForests() {
  const glyphs = []
  for (const t of TRACKS) {
    const n = forestCount(t)
    const rnd = mulberry32(t.seed ^ 0x51ed270b)
    for (let i = 0; i < n; i++) {
      const ang = rnd() * Math.PI * 2
      const rad = 0.022 + rnd() * 0.05
      glyphs.push({
        x: t.x + Math.cos(ang) * rad,
        y: t.y + Math.sin(ang) * rad * 0.82,
        type: Math.floor(rnd() * 3), // 0 fir, 1 round bush, 2 scrub
        size: 0.7 + rnd() * 0.7,
        // higher-energy places lean coniferous & dark; lowlands softer
        dark: t.energy > 6 ? rnd() > 0.3 : rnd() > 0.7,
      })
    }
  }
  // Paint far-then-near so overlaps look layered.
  glyphs.sort((a, b) => a.y - b.y)
  return glyphs
}

// A couple of organic wax-crayon lakes tucked into low ground (bottom, away from
// the high-energy NE peaks). Hand-authored blob outlines in normalized space.
const LAKES = [
  {
    pts: [
      [0.12, 0.9], [0.2, 0.86], [0.29, 0.88], [0.33, 0.93],
      [0.3, 0.98], [0.2, 1.0], [0.12, 0.97],
    ],
  },
  {
    pts: [
      [0.52, 0.93], [0.6, 0.9], [0.68, 0.92], [0.7, 0.96],
      [0.64, 1.0], [0.55, 1.0], [0.5, 0.97],
    ],
  },
]

// Fold seams across the paper (fixed positions so the sheet never changes).
const FOLDS = [
  { x: 0.33, wob: 6 },
  { x: 0.67, wob: -5 },
  { y: 0.5, wob: 4 },
]

// Lazy singletons.
let _field = null
let _contours = null
let _forests = null

export function getField() {
  if (!_field) _field = buildField(150, 96)
  return _field
}
export function getContours() {
  if (!_contours) _contours = generateContours({ res: 88, nLevels: 13, pad: PAD })
  return _contours
}
export function getForests() {
  if (!_forests) _forests = buildForests()
  return _forests
}
export { LAKES, FOLDS, PAD }
