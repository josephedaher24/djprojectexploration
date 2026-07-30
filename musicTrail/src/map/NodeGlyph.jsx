// One line-art glyph per channel. Strokes only — no fills, no bezels. Each glyph
// carries that channel's real payload, so a node literally looks different
// depending on which probe is patched in.
import { grooveFlux } from './layers.js'

export default function NodeGlyph({ channel, track, v, isCamp }) {
  if (isCamp) return <CampGlyph />
  if (channel === 'harm') return <Rosette track={track} v={v} />
  if (channel === 'groove') return <Sequencer track={track} />
  return <Blip track={track} v={v} />
}

// ── camp / trigger: open diamond + hot core, breathing on the beat ────────────
function CampGlyph() {
  const r = 11
  return (
    <g className="g g--camp">
      <path className="g__dia" d={`M0 ${-r} L${r} 0 L0 ${r} L${-r} 0 Z`} />
      <circle className="g__core" r="2" />
    </g>
  )
}

// ── CH1: a blip — cross + open diamond sized by energy ───────────────────────
function Blip({ track, v }) {
  const r = 5 + track.energy * 0.45
  const arm = 4.5
  return (
    <g className="g">
      <path d={`M${-arm} 0 H${arm}`} />
      <path d={`M0 ${-arm} V${arm}`} />
      <path className="g__dia" d={`M0 ${-r} L${r} 0 L0 ${r} L${-r} 0 Z`} />
    </g>
  )
}

// ── CH2: a Camelot rosette — 12 ticks, the track's own key extended ───────────
function Rosette({ track }) {
  const R = 9
  const m = /^(\d{1,2})([AB])$/.exec(track.camelot || '')
  const ring = m ? parseInt(m[1], 10) : 1
  const minor = m ? m[2] === 'A' : false
  const ticks = []
  for (let i = 0; i < 12; i++) {
    const a = (i / 12) * Math.PI * 2 - Math.PI / 2
    const own = i === (ring - 1) % 12
    const r2 = own ? 14 : R + 3
    ticks.push(
      <line
        key={i}
        className={own ? 'g__own' : 'g__tick'}
        x1={Math.cos(a) * R}
        y1={Math.sin(a) * R}
        x2={Math.cos(a) * r2}
        y2={Math.sin(a) * r2}
      />,
    )
  }
  return (
    <g className="g">
      <circle className="g__hoop" r={R} />
      {/* minor keys carry an inner hoop so A/B is readable at a glance */}
      {minor && <circle className="g__hoop g__hoop--in" r={R - 3.4} />}
      {ticks}
    </g>
  )
}

// ── CH3: a micro-sequencer — the real 16×3 onset grid as line ticks ──────────
function Sequencer({ track }) {
  const PITCH = 1.7
  const W = 16 * PITCH
  const x0 = -W / 2
  const rows = track.grooveGrid
  const out = []
  for (let r = 0; r < rows.length; r++) {
    const base = -3 + r * 5 // three stacked baselines
    out.push(
      <line key={`b${r}`} className="g__base" x1={x0} y1={base} x2={x0 + W} y2={base} />,
    )
    for (let c = 0; c < rows[r].length; c++) {
      const hgt = 1 + rows[r][c] * 4.2
      const x = x0 + c * PITCH + PITCH / 2
      out.push(
        <line
          key={`${r}-${c}`}
          className={c % 4 === 0 ? 'g__step g__step--beat' : 'g__step'}
          x1={x}
          y1={base}
          x2={x}
          y2={base - hgt}
        />,
      )
    }
  }
  return <g className="g g--seq">{out}</g>
}

export { grooveFlux }
