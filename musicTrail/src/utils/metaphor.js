// ─────────────────────────────────────────────────────────────────────────────
// The metaphor engine. Turns two tracks (A = current camp, B = candidate place)
// into a hiking route. Every number the UI shows in km / m / bars / BPM comes
// from here, derived from the mock features in data/tracks.js.
//
//   style distance   → trail distance   (how far the walk is)
//   harmonic distance → climb / ascent  (how steep)
//   groove distance  → vegetation       (how dense the scrub is)
//   tempo delta      → wind / weather    (head- or tail-wind)
//   ETA (Naismith)   → mix duration in bars
// ─────────────────────────────────────────────────────────────────────────────
import { TRACKS } from '../data/tracks.js'
import { clamp01, clamp, lerp } from './rng.js'

// ── primitive distances ──────────────────────────────────────────────────────

// Style distance: plain 2D separation on the PaCMAP-style map, 0..1.
const STYLE_SPAN = 1.05 // roughly the widest gap between placed points
export function styleDistance01(a, b) {
  const d = Math.hypot(a.x - b.x, a.y - b.y)
  return clamp01(d / STYLE_SPAN)
}

// Harmonic distance: circle-of-fifths gap on the Camelot wheel, 0..1.
function parseCamelot(code) {
  const m = /^(\d{1,2})([AB])$/.exec(code)
  if (!m) return { ring: 1, letter: 'A' }
  return { ring: parseInt(m[1], 10), letter: m[2] }
}
export function harmonicDistance01(a, b) {
  const ka = parseCamelot(a.camelot)
  const kb = parseCamelot(b.camelot)
  const raw = Math.abs(ka.ring - kb.ring)
  const ringDist = Math.min(raw, 12 - raw) // 0..6 around the wheel
  const letterDiff = ka.letter === kb.letter ? 0 : 1
  return clamp01((ringDist / 6) * 0.85 + letterDiff * 0.15)
}

// Groove distance: mean absolute difference between the two 16×3 onset grids, 0..1.
export function grooveDistance01(a, b) {
  let sum = 0
  let n = 0
  for (let r = 0; r < a.grooveGrid.length; r++) {
    for (let c = 0; c < a.grooveGrid[r].length; c++) {
      sum += Math.abs(a.grooveGrid[r][c] - b.grooveGrid[r][c])
      n++
    }
  }
  return clamp01((sum / n) * 1.8) // scale up; raw diffs are small
}

// ── derived, human-facing quantities ─────────────────────────────────────────

export function distanceKm(style01) {
  return Math.round((0.4 + style01 * 7.6) * 10) / 10 // ~0.4–8.0 km
}

// Climb: harmonic friction becomes metres of ascent (always ≥ 0). Separate from
// the *net* altitude change, which can be a descent.
export function ascentM(harm01) {
  return Math.round(harm01 * 620) // 0–620 m of harmonic "climbing"
}

// Vegetation 0..5 bars from groove distance.
export function vegetationLevel(groove01) {
  return Math.round(groove01 * 5)
}

// Wind from tempo delta. Tailwind = destination is same/slower BPM (easy);
// headwind = you have to push up in tempo.
export function windFrom(a, b) {
  const delta = b.bpm - a.bpm
  const mag = Math.abs(delta)
  let kind = 'calm'
  if (delta > 2) kind = 'headwind'
  else if (delta < -2) kind = 'tailwind'
  return { delta, mag, kind }
}

// Naismith-style rule → mix duration in bars (whole phrases), quantised to 4.
export function naismithBars(style01, harm01, groove01) {
  const base = 8,
    kDist = 8,
    kClimb = 12,
    kBush = 4
  const raw = base + kDist * style01 + kClimb * harm01 + kBush * groove01
  const clamped = clamp(raw, 4, 32)
  return Math.round(clamped / 4) * 4
}

export function barsToSeconds(bars, bpm) {
  return Math.round((bars * 4 * 60) / bpm) // 4 beats per bar
}

// Difficulty score 0..1 and its SAC-style tier.
export function difficulty01(style01, harm01, groove01) {
  return clamp01(0.3 * style01 + 0.45 * harm01 + 0.25 * groove01)
}

export const TIERS = [
  { grade: 'T1', name: 'Easy Path', max: 0.2, note: 'A gentle crossfade — barely any work.' },
  { grade: 'T2', name: 'Mountain Trail', max: 0.4, note: 'A standard blend, steady footing.' },
  { grade: 'T3', name: 'Rugged Trail', max: 0.6, note: 'Long blend with an EQ hand-off.' },
  { grade: 'T4', name: 'Alpine Route', max: 0.8, note: 'A very long transition under filter cover.' },
  { grade: 'T5', name: 'Ropes Needed', max: 1.01, note: 'A tough line — but you can still try it.' },
]
export function tierFor(diff01) {
  return TIERS.find((t) => diff01 < t.max) || TIERS[TIERS.length - 1]
}

// ── the altitude field (for the elevation profile) ───────────────────────────

// Smooth altitude at any map point: inverse-gaussian blend of every track's
// altitude. This is the terrain the elevation profile slices through.
const SIGMA = 0.16
export function altitudeAt(x, y) {
  let num = 0
  let den = 0
  for (const t of TRACKS) {
    const d2 = (t.x - x) ** 2 + (t.y - y) ** 2
    const w = Math.exp(-d2 / (2 * SIGMA * SIGMA))
    num += t.altitude * w
    den += w
  }
  return den > 0 ? num / den : 0
}

// Perpendicular distance from point P to segment AB, plus the projection param t.
function projectToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax
  const dy = by - ay
  const len2 = dx * dx + dy * dy || 1e-9
  let t = ((px - ax) * dx + (py - ay) * dy) / len2
  t = clamp01(t)
  const cx = ax + t * dx
  const cy = ay + t * dy
  return { t, dist: Math.hypot(px - cx, py - cy) }
}

// ── the whole route ──────────────────────────────────────────────────────────

const N_PROFILE = 24

export function buildRoute(a, b) {
  if (!a || !b || a.id === b.id) return null

  const style01 = styleDistance01(a, b)
  const harm01 = harmonicDistance01(a, b)
  const groove01 = grooveDistance01(a, b)
  const wind = windFrom(a, b)
  const diff01 = difficulty01(style01, harm01, groove01)
  const tier = tierFor(diff01)
  const bars = naismithBars(style01, harm01, groove01)

  // Elevation profile: sample the altitude field along A→B.
  const profile = []
  for (let i = 0; i < N_PROFILE; i++) {
    const f = i / (N_PROFILE - 1)
    const px = lerp(a.x, b.x, f)
    const py = lerp(a.y, b.y, f)
    profile.push(altitudeAt(px, py))
  }
  const minAlt = Math.min(...profile)
  const maxAlt = Math.max(...profile)

  // Waypoints: other tracks hugging the corridor between A and B.
  const waypoints = TRACKS.filter((t) => t.id !== a.id && t.id !== b.id)
    .map((t) => {
      const { t: proj, dist } = projectToSegment(t.x, t.y, a.x, a.y, b.x, b.y)
      return { track: t, proj, dist }
    })
    .filter((w) => w.dist < 0.11 && w.proj > 0.08 && w.proj < 0.92)
    .sort((p, q) => p.proj - q.proj)

  return {
    from: a,
    to: b,
    style01,
    harm01,
    groove01,
    diff01,
    tier,
    bars,
    seconds: barsToSeconds(bars, a.bpm),
    distanceKm: distanceKm(style01),
    ascentM: ascentM(harm01),
    netAltitude: b.altitude - a.altitude,
    vegetation: vegetationLevel(groove01),
    wind,
    profile,
    minAlt,
    maxAlt,
    waypoints,
    bearing: bearingDeg(a, b),
  }
}

// ── compass bearing ──────────────────────────────────────────────────────────

// Bearing A→B in screen space (north = up = decreasing y). 0°=N, 90°=E …
export function bearingDeg(a, b) {
  const dx = b.x - a.x
  const dy = b.y - a.y
  let deg = (Math.atan2(dx, -dy) * 180) / Math.PI
  if (deg < 0) deg += 360
  return deg
}

const COMPASS_POINTS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
export function compassLabel(deg) {
  return COMPASS_POINTS[Math.round(deg / 45) % 8]
}
