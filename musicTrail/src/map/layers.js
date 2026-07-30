// ─────────────────────────────────────────────────────────────────────────────
// The three scope CHANNELS. One instrument, three probes: the same 20 tracks at
// the same coordinates, measured through a different feature dimension. Nothing
// here is decorative — each channel derives a real field, a real node payload
// and a real connection rule, so patching the beam genuinely rewires the picture.
//
//   CH1 STYLE   ← MAEST timbre / PaCMAP position   → density waterfall + kNN mesh
//   CH2 HARM    ← key + energy + altitude          → contour relief + Camelot edges
//   CH3 GROOVE  ← 16×3 onset flux + BPM            → warped lattice + tempo rails
// ─────────────────────────────────────────────────────────────────────────────
import { TRACKS } from '../data/tracks.js'
import {
  altitudeAt,
  styleDistance01,
  harmonicDistance01,
  grooveDistance01,
} from '../utils/metaphor.js'
import { generateContours } from '../utils/terrain.js'
import { clamp01 } from '../utils/rng.js'

export const CHANNELS = ['style', 'harm', 'groove']

export const CH_META = {
  style: {
    key: 'style',
    ch: 'CH1',
    name: 'STYLE',
    sub: 'Timbre density',
    axis: 'MAEST timbre distance',
    reveals:
      'Where tracks pile up by how they sound. Ridges are crowded style families; the mesh is the nearest-neighbour graph a recommender would actually walk.',
    varName: '--ch1-style',
  },
  harm: {
    key: 'harm',
    ch: 'CH2',
    name: 'HARM',
    sub: 'Key & relief',
    axis: 'Camelot geometry + energy relief',
    reveals:
      'How things are tuned and how high they sit. Contours are real energy relief; edges only connect keys within one step on the wheel.',
    varName: '--ch2-harm',
  },
  groove: {
    key: 'groove',
    ch: 'CH3',
    name: 'GROOVE',
    sub: 'Rhythm lattice',
    axis: '16×3 onset flux + BPM',
    reveals:
      'How things move. The lattice pinches where percussive material clusters; rails link only the pairs you could phrase-lock without touching pitch.',
    varName: '--ch3-groove',
  },
}

// ── per-track scalars ────────────────────────────────────────────────────────

// Total onset flux of a track's groove grid → how busy its rhythm is.
export function grooveFlux(track) {
  let sum = 0
  for (const row of track.grooveGrid) for (const v of row) sum += v
  return sum / (track.grooveGrid.length * track.grooveGrid[0].length)
}

// ── scalar fields (normalized 0..1 map space → number) ───────────────────────

const S_STYLE = 0.13
const S_GROOVE = 0.16

// CH1: kernel density of tracks in embedding space, normalized 0..1.
let _styleMax = null
export function styleDensityRaw(x, y) {
  let sum = 0
  for (const t of TRACKS) {
    const d2 = (t.x - x) ** 2 + (t.y - y) ** 2
    sum += Math.exp(-d2 / (2 * S_STYLE * S_STYLE))
  }
  return sum
}
export function styleDensityAt(x, y) {
  if (_styleMax == null) {
    let m = 0
    for (let j = 0; j <= 40; j++) {
      for (let i = 0; i <= 40; i++) {
        const v = styleDensityRaw(i / 40, j / 40)
        if (v > m) m = v
      }
    }
    _styleMax = m || 1
  }
  return clamp01(styleDensityRaw(x, y) / _styleMax)
}

// CH2: the existing altitude/energy field.
export { altitudeAt as harmAt }

// CH3: distance-weighted groove flux, normalized 0..1.
export function grooveFieldAt(x, y) {
  let num = 0
  let den = 0
  for (const t of TRACKS) {
    const d2 = (t.x - x) ** 2 + (t.y - y) ** 2
    const w = Math.exp(-d2 / (2 * S_GROOVE * S_GROOVE))
    num += grooveFlux(t) * w
    den += w
  }
  return den > 0 ? num / den : 0
}

// ── connection rules (one per channel — genuinely different graphs) ──────────

function parseCamelot(code) {
  const m = /^(\d{1,2})([AB])$/.exec(code || '')
  return m ? { ring: parseInt(m[1], 10), letter: m[2] } : { ring: 1, letter: 'A' }
}

// Within one step on the Camelot wheel: ±1 ring same letter, or the A/B relative.
export function camelotWithinOneStep(a, b) {
  const ka = parseCamelot(a.camelot)
  const kb = parseCamelot(b.camelot)
  const raw = Math.abs(ka.ring - kb.ring)
  const ringDist = Math.min(raw, 12 - raw)
  if (ka.letter === kb.letter) return ringDist <= 1
  return ringDist === 0
}

// CH1: k-nearest-neighbour mesh in style space.
function styleMesh(k = 3) {
  const seen = new Set()
  const edges = []
  for (const a of TRACKS) {
    const near = TRACKS.filter((b) => b.id !== a.id)
      .map((b) => ({ b, d: styleDistance01(a, b) }))
      .sort((p, q) => p.d - q.d)
      .slice(0, k)
    for (const { b, d } of near) {
      const key = a.id < b.id ? `${a.id}|${b.id}` : `${b.id}|${a.id}`
      if (seen.has(key)) continue
      seen.add(key)
      edges.push({ a: a.id, b: b.id, d, w: clamp01(1 - d) })
    }
  }
  return edges
}

// CH2: only harmonically compatible pairs.
function camelotEdges() {
  const edges = []
  for (let i = 0; i < TRACKS.length; i++) {
    for (let j = i + 1; j < TRACKS.length; j++) {
      const a = TRACKS[i]
      const b = TRACKS[j]
      if (!camelotWithinOneStep(a, b)) continue
      const d = harmonicDistance01(a, b)
      edges.push({ a: a.id, b: b.id, d, w: clamp01(1 - d) })
    }
  }
  return edges
}

// CH3: tempo rails — phrase-lockable without touching pitch.
function tempoRails() {
  const edges = []
  for (let i = 0; i < TRACKS.length; i++) {
    for (let j = i + 1; j < TRACKS.length; j++) {
      const a = TRACKS[i]
      const b = TRACKS[j]
      if (Math.abs(a.bpm - b.bpm) > 3) continue
      const gd = grooveDistance01(a, b)
      if (gd >= 0.35) continue
      edges.push({ a: a.id, b: b.id, d: gd, w: clamp01(1 - gd) })
    }
  }
  return edges
}

const EDGES = { style: styleMesh, harm: camelotEdges, groove: tempoRails }
const DIST = { style: styleDistance01, harm: harmonicDistance01, groove: grooveDistance01 }
const FIELD = { style: styleDensityAt, harm: altitudeAt, groove: grooveFieldAt }

// ── per-node values (0..1) ───────────────────────────────────────────────────

function nodeValues(channel) {
  const raw = {}
  for (const t of TRACKS) {
    if (channel === 'style') raw[t.id] = styleDensityAt(t.x, t.y)
    else if (channel === 'harm') raw[t.id] = t.energy
    else raw[t.id] = grooveFlux(t)
  }
  const vals = Object.values(raw)
  const lo = Math.min(...vals)
  const hi = Math.max(...vals)
  const span = hi - lo || 1
  const out = {}
  for (const id of Object.keys(raw)) out[id] = (raw[id] - lo) / span
  return out
}

// ── lazily built + cached per channel ────────────────────────────────────────
const cache = {}

export function getChannel(channel) {
  if (cache[channel]) return cache[channel]
  const data = {
    ...CH_META[channel],
    edges: EDGES[channel](),
    nodeValue: nodeValues(channel),
    fieldAt: FIELD[channel],
    // CH2 is the only channel that traces real isolines
    contours: channel === 'harm' ? generateContours({ res: 88, nLevels: 13, pad: 0.06 }) : null,
  }
  cache[channel] = data
  return data
}

export function channelDistance(channel, a, b) {
  return DIST[channel](a, b)
}
