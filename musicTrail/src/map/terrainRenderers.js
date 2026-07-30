// ─────────────────────────────────────────────────────────────────────────────
// One canvas painter per channel. These are the reason switching channels feels
// like re-patching a probe rather than recolouring a chart — the terrain is drawn
// by a completely different instrument in each case:
//
//   CH1 STYLE  → a spectrum-analyser WATERFALL of the timbre-density field
//   CH2 HARM   → CONTOUR relief traced from the real altitude field
//   CH3 GROOVE → a warped 16-step LATTICE that pinches around percussive mass
//
// Each paints ONCE per (channel, size) into a cached canvas. No per-frame cost.
// ─────────────────────────────────────────────────────────────────────────────
import { neonStroke, alpha } from './neon.js'
import { styleDensityAt, grooveFieldAt } from './layers.js'

// ── CH1: timbre-density waterfall ────────────────────────────────────────────
export function drawStyleWaterfall(ctx, w, h, project, hue) {
  const ROWS = 46
  const SAMPLES = 150
  // generous displacement so a dense timbre cluster visibly lifts its scan row
  // clear of its neighbours — that ridge IS the information
  const AMP = Math.max(52, h * 0.13)

  for (let r = 0; r < ROWS; r++) {
    const my = r / (ROWS - 1)
    const baseY = project(0.5, my)[1]

    // sample the row once so we can both draw it and judge its brightness
    const pts = new Array(SAMPLES)
    let peak = 0
    for (let i = 0; i < SAMPLES; i++) {
      const mx = i / (SAMPLES - 1)
      const d = styleDensityAt(mx, my)
      if (d > peak) peak = d
      pts[i] = [project(mx, my)[0], baseY - d * AMP]
    }

    const draw = (c) => {
      c.moveTo(pts[0][0], pts[0][1])
      for (let i = 1; i < SAMPLES; i++) c.lineTo(pts[i][0], pts[i][1])
    }
    // quiet rows stay as faint scanlines; ridges bloom hard
    neonStroke(ctx, draw, {
      color: hue,
      width: 0.6 + peak * 0.7,
      glow: 4 + peak * 10,
      bloomAlpha: 0.02 + peak * 0.16,
      coreAlpha: 0.07 + peak * 0.78,
    })
  }
}

// ── CH2: contour relief from the real altitude field ─────────────────────────
export function drawHarmContours(ctx, w, h, project, hue, contours) {
  const { levels } = contours
  levels.forEach((lvl, li) => {
    const t = li / Math.max(1, levels.length - 1)
    const index = li % 3 === 0 // every 3rd is an index contour
    const draw = (c) => {
      for (const [ax, ay, bx, by] of lvl.segments) {
        const [x1, y1] = project(ax, ay)
        const [x2, y2] = project(bx, by)
        c.moveTo(x1, y1)
        c.lineTo(x2, y2)
      }
    }
    neonStroke(ctx, draw, {
      color: hue,
      width: index ? 1.1 : 0.7,
      glow: index ? 9 : 5,
      bloomAlpha: 0.03 + t * 0.07,
      coreAlpha: 0.14 + t * 0.41 + (index ? 0.1 : 0),
    })
  })
}

// ── CH3: warped step lattice ─────────────────────────────────────────────────
export function drawGrooveLattice(ctx, w, h, project, hue) {
  const V = 16 // one hairline per 16th of a bar
  const H = 12
  const WARP = 22
  const SAMPLES = 70

  // normalise the groove field over the well so the warp uses its full range
  let lo = Infinity
  let hi = -Infinity
  const grid = []
  for (let j = 0; j <= 24; j++) {
    const row = []
    for (let i = 0; i <= 24; i++) {
      const v = grooveFieldAt(i / 24, j / 24)
      row.push(v)
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
    grid.push(row)
  }
  const span = hi - lo || 1
  const norm = (mx, my) => (grooveFieldAt(mx, my) - lo) / span - 0.5 // −0.5..0.5

  // vertical hairlines, displaced horizontally where percussion clusters
  for (let i = 0; i < V; i++) {
    const mx = (i + 0.5) / V
    const beat = i % 4 === 0 // downbeats read stronger
    const pts = []
    for (let s = 0; s < SAMPLES; s++) {
      const my = s / (SAMPLES - 1)
      const [px, py] = project(mx, my)
      pts.push([px + norm(mx, my) * WARP * 2, py])
    }
    const draw = (c) => {
      c.moveTo(pts[0][0], pts[0][1])
      for (let s = 1; s < SAMPLES; s++) c.lineTo(pts[s][0], pts[s][1])
    }
    neonStroke(ctx, draw, {
      color: hue,
      width: beat ? 1 : 0.6,
      glow: beat ? 7 : 4,
      bloomAlpha: beat ? 0.08 : 0.04,
      coreAlpha: beat ? 0.5 : 0.24,
    })
  }

  // horizontal hairlines, displaced vertically by the same field
  for (let j = 0; j < H; j++) {
    const my = (j + 0.5) / H
    const pts = []
    for (let s = 0; s < SAMPLES; s++) {
      const mx = s / (SAMPLES - 1)
      const [px, py] = project(mx, my)
      pts.push([px, py + norm(mx, my) * WARP])
    }
    const draw = (c) => {
      c.moveTo(pts[0][0], pts[0][1])
      for (let s = 1; s < SAMPLES; s++) c.lineTo(pts[s][0], pts[s][1])
    }
    neonStroke(ctx, draw, {
      color: hue,
      width: 0.6,
      glow: 4,
      bloomAlpha: 0.03,
      coreAlpha: 0.18,
    })
  }
}

// ── the shared scope graticule (channel-independent) ─────────────────────────
export function drawGraticule(ctx, w, h, minor, major) {
  const COLS = 10
  const ROWS = 8
  ctx.save()
  // minor divisions
  ctx.strokeStyle = minor
  ctx.lineWidth = 0.5
  ctx.beginPath()
  for (let i = 1; i < COLS; i++) {
    const x = Math.round((i / COLS) * w) + 0.25
    ctx.moveTo(x, 0)
    ctx.lineTo(x, h)
  }
  for (let j = 1; j < ROWS; j++) {
    const y = Math.round((j / ROWS) * h) + 0.25
    ctx.moveTo(0, y)
    ctx.lineTo(w, y)
  }
  ctx.stroke()

  // centre axes + 0.2-division tick marks (the classic scope centre scale)
  ctx.strokeStyle = major
  ctx.lineWidth = 1
  const cx = Math.round(w / 2) + 0.5
  const cy = Math.round(h / 2) + 0.5
  ctx.beginPath()
  ctx.moveTo(cx, 0)
  ctx.lineTo(cx, h)
  ctx.moveTo(0, cy)
  ctx.lineTo(w, cy)
  const stepX = w / COLS / 5
  for (let k = 0; k * stepX < w; k++) {
    const x = Math.round(k * stepX) + 0.5
    ctx.moveTo(x, cy - 3)
    ctx.lineTo(x, cy + 3)
  }
  const stepY = h / ROWS / 5
  for (let k = 0; k * stepY < h; k++) {
    const y = Math.round(k * stepY) + 0.5
    ctx.moveTo(cx - 3, y)
    ctx.lineTo(cx + 3, y)
  }
  ctx.stroke()
  ctx.restore()
}

export { alpha }
