// ─────────────────────────────────────────────────────────────────────────────
// Mock trail data — the 20 real "warren" track titles, dressed up as places on a
// hiking map. All feature values are invented (this is a front-end design demo),
// but they are internally consistent: every route metric on screen is derived
// from these numbers, so the map behaves like the real thing would.
// ─────────────────────────────────────────────────────────────────────────────
import { mulberry32, hashString, clamp01 } from '../utils/rng.js'

// Camelot code → friendly musical key name (for the compass / key readouts).
const CAMELOT_NAMES = {
  '1A': 'Ab minor', '1B': 'B major', '2A': 'Eb minor', '2B': 'F# major',
  '3A': 'Bb minor', '3B': 'Db major', '4A': 'F minor', '4B': 'Ab major',
  '5A': 'C minor', '5B': 'Eb major', '6A': 'G minor', '6B': 'Bb major',
  '7A': 'D minor', '7B': 'F major', '8A': 'A minor', '8B': 'C major',
  '9A': 'E minor', '9B': 'G major', '10A': 'B minor', '10B': 'D major',
  '11A': 'F# minor', '11B': 'A major', '12A': 'Db minor', '12B': 'E major',
}

// Duotone "album cover" swatches (no real art available) — warm topographic tones,
// picked so markers stay legible on the paper map.
const SWATCHES = [
  ['#f0b46a', '#b45c2b'], ['#8fd1c4', '#2f6d6b'], ['#e88f8f', '#a83a4a'],
  ['#b7d98a', '#4f7a34'], ['#f2c76b', '#c78a2e'], ['#9db9e8', '#3d5a94'],
  ['#e6a3cf', '#8a3f78'], ['#7fc7a8', '#2f7d5c'], ['#f0a58a', '#b0492f'],
  ['#c9b18a', '#7a5a34'], ['#93c9e0', '#2f6a8f'], ['#d7b6e0', '#6a3f8a'],
  ['#f0cf8a', '#b08a2e'], ['#a3d18f', '#3f7a34'], ['#e89ca8', '#a8395a'],
  ['#8fbfd1', '#2f5f7d'], ['#e0b48f', '#9a5a2f'], ['#b0d98f', '#57832f'],
  ['#f0a86b', '#b0532b'], ['#93c7c4', '#2f6d6b'],
]

// Base table: title / artist / cluster region / coords / energy / bpm / camelot.
// Coordinates are in a normalized 0..1 PaCMAP-style space; the map scales them.
const BASE = [
  // Coastal Lowlands — mellow vocal cuts, low altitude
  { n: 1, title: 'Used To Love', artist: 'Martin Garrix, Dean Lewis', region: 'Coastal Lowlands', x: 0.14, y: 0.72, energy: 3, bpm: 120, key: '9B' },
  { n: 12, title: 'Ocean', artist: 'Martin Garrix, Khalid', region: 'Coastal Lowlands', x: 0.10, y: 0.60, energy: 3, bpm: 118, key: '4A' },
  { n: 2, title: 'Ocean (Don Diablo Remix)', artist: 'Martin Garrix, Don Diablo, Khalid', region: 'Coastal Lowlands', x: 0.22, y: 0.66, energy: 5, bpm: 124, key: '4B' },
  { n: 11, title: 'Hero', artist: 'Martin Garrix, JVKE', region: 'Coastal Lowlands', x: 0.18, y: 0.83, energy: 4, bpm: 122, key: '11A' },
  { n: 18, title: 'Together', artist: 'Martin Garrix, Matisse & Sadko', region: 'Coastal Lowlands', x: 0.30, y: 0.78, energy: 4, bpm: 123, key: '9A' },
  { n: 17, title: 'Mistaken', artist: 'Martin Garrix, Matisse & Sadko, Alex Aris', region: 'Coastal Lowlands', x: 0.27, y: 0.55, energy: 5, bpm: 122, key: '1A' },

  // Summer Ridge — progressive, mid altitude, the connective spine
  { n: 14, title: 'Summer Days', artist: 'Martin Garrix, Macklemore, Patrick Stump', region: 'Summer Ridge', x: 0.46, y: 0.50, energy: 6, bpm: 114, key: '5B' },
  { n: 15, title: 'Summer Days (VC & Bruno Be Remix)', artist: 'Vintage Culture, Bruno Be', region: 'Summer Ridge', x: 0.55, y: 0.42, energy: 7, bpm: 123, key: '5A' },
  { n: 16, title: 'Forever', artist: 'Martin Garrix, Matisse & Sadko', region: 'Summer Ridge', x: 0.40, y: 0.35, energy: 6, bpm: 125, key: '6B' },
  { n: 7, title: 'These Are The Times', artist: 'Martin Garrix, JRM', region: 'Summer Ridge', x: 0.50, y: 0.64, energy: 5, bpm: 120, key: '10A' },
  { n: 8, title: 'These Are The Times (Dyro Remix)', artist: 'Martin Garrix, JRM, Dyro', region: 'Summer Ridge', x: 0.59, y: 0.71, energy: 7, bpm: 128, key: '10B' },
  { n: 3, title: 'Empty', artist: 'Martin Garrix, DubVision, Jaimes', region: 'Summer Ridge', x: 0.44, y: 0.24, energy: 7, bpm: 126, key: '2B' },
  { n: 5, title: 'Love Runs Out', artist: 'Martin Garrix, G-Eazy, Sasha Alex Sloan', region: 'Summer Ridge', x: 0.36, y: 0.60, energy: 6, bpm: 122, key: '7A' },
  { n: 9, title: 'Burn Out', artist: 'Martin Garrix, Justin Mylo, Dewain Whitmore', region: 'Summer Ridge', x: 0.63, y: 0.56, energy: 7, bpm: 124, key: '3B' },

  // Festival Peaks — big-room / high energy, the summits, NE
  { n: 6, title: 'Wizard', artist: 'Martin Garrix, Jay Hardway', region: 'Festival Peaks', x: 0.80, y: 0.28, energy: 9, bpm: 128, key: '8B' },
  { n: 10, title: 'Bouncybob', artist: 'Martin Garrix, Justin Mylo, Mesto', region: 'Festival Peaks', x: 0.88, y: 0.39, energy: 9, bpm: 130, key: '12B' },
  { n: 13, title: 'Game Over', artist: 'Martin Garrix, LOOPERS', region: 'Festival Peaks', x: 0.74, y: 0.17, energy: 10, bpm: 150, key: '8A' },
  { n: 4, title: 'Latency', artist: 'Martin Garrix, Dyro', region: 'Festival Peaks', x: 0.70, y: 0.41, energy: 8, bpm: 128, key: '6A' },
  { n: 19, title: 'Limitless', artist: 'Martin Garrix, Mesto', region: 'Festival Peaks', x: 0.85, y: 0.53, energy: 8, bpm: 126, key: '3A' },
  { n: 20, title: 'WIEE', artist: 'Martin Garrix, Mesto', region: 'Festival Peaks', x: 0.93, y: 0.23, energy: 10, bpm: 132, key: '12A' },
]

// Build a 16×3 onset-flux groove grid; grooviness rises with energy so festival
// peaks read as dense scrub and lowland vocal cuts read as open meadow.
function buildGrooveGrid(seed, energy) {
  const rnd = mulberry32(seed)
  const density = 0.28 + (energy / 10) * 0.55
  const grid = []
  for (let row = 0; row < 3; row++) {
    const line = []
    for (let col = 0; col < 16; col++) {
      // Emphasize the downbeats (col % 4 === 0) like a real kick pattern.
      const beat = col % 4 === 0 ? 0.35 : 0
      const v = clamp01(density * rnd() + beat + (rnd() - 0.5) * 0.25)
      line.push(Number(v.toFixed(3)))
    }
    grid.push(line)
  }
  return grid
}

// Altitude label (metres): energy 1..10 → ~780..2460 m, with a little seeded drift.
function altitudeFor(seed, energy) {
  const rnd = mulberry32(seed ^ 0x9e3779b9)
  return Math.round(600 + energy * 180 + (rnd() - 0.5) * 90)
}

export const TRACKS = BASE.map((t, i) => {
  const seed = hashString(t.title)
  const swatch = SWATCHES[i % SWATCHES.length]
  return {
    id: `t${t.n}`,
    trackNo: t.n,
    title: t.title,
    artist: t.artist,
    region: t.region,
    x: t.x,
    y: t.y,
    energy: t.energy,
    bpm: t.bpm,
    camelot: t.key,
    keyName: CAMELOT_NAMES[t.key] || t.key,
    altitude: altitudeFor(seed, t.energy),
    grooveGrid: buildGrooveGrid(seed, t.energy),
    cover: swatch, // [light, dark] duotone stand-in for album art
    seed,
  }
})

// Default "current camp" — start in the mellow lowlands, like the warm-up of a set.
export const START_ID = 't12' // Ocean
export const getTrack = (id) => TRACKS.find((t) => t.id === id)

// A narrative order for the trail log / route ribbon: warm up in the lowlands,
// cross the summer ridge, summit at the festival peaks. (Purely for storytelling;
// the map itself lets you walk anywhere.)
export const SET_ORDER = [
  't12', 't1', 't2', 't17', 't11', 't18', // Coastal Lowlands
  't5', 't7', 't14', 't15', 't16', 't3', 't8', 't9', // Summer Ridge
  't4', 't19', 't6', 't10', 't20', 't13', // Festival Peaks → summit at Game Over
]

// How many hand-drawn bush glyphs to scatter around a place: a busy festival
// banger sits in thick woods, a sparse vocal cut in open meadow.
export function forestCount(track) {
  let sum = 0
  for (const row of track.grooveGrid) for (const v of row) sum += v
  const mean = sum / (track.grooveGrid.length * track.grooveGrid[0].length)
  return Math.max(2, Math.round(mean * 16))
}
