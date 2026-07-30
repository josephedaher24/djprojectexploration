import { useMemo } from 'react'
import { Mod } from './Rack.jsx'
import { heartEnvelope } from '../clock/ClockProvider.jsx'

const W = 200
const H = 40

// Two beats of the clock's actual envelope, drawn twice end-to-end and scrolled
// by exactly one trace width. ZERO per-frame JS: the CSS animation duration is
// var(--beat) * 2, so it retimes itself whenever the camp BPM changes.
export default function CadenceTrace({ camp, route }) {
  const d = useMemo(() => {
    const N = 96
    const pts = []
    for (let i = 0; i < N; i++) {
      const p = (i / (N - 1)) * 2 // two beats
      const v = heartEnvelope(p % 1)
      pts.push([(i / (N - 1)) * W, H - 6 - v * (H - 14)])
    }
    return pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ')
  }, [])

  const delta = route ? route.wind.delta : 0

  return (
    <Mod name="Cadence" sub="2 BEATS" className="mod--cad">
      <svg className="cad" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <g className="cad__grat">
          {Array.from({ length: 4 }).map((_, i) => (
            <line key={i} x1={((i + 1) / 5) * W} y1="0" x2={((i + 1) / 5) * W} y2={H} />
          ))}
          <line className="cad__base" x1="0" y1={H - 6} x2={W} y2={H - 6} />
        </g>
        {/* the group holds the trace twice and slides left by half its width */}
        <g className="cad__scroll">
          <path className="cad__line" d={d} />
          <g transform={`translate(${W} 0)`}>
            <path className="cad__line" d={d} />
          </g>
        </g>
      </svg>
      <div className="mod__row">
        <span className="n n--hero">
          {camp.bpm}
          <span className="u">bpm</span>
        </span>
        {route && (
          <span className={`cad__d n n--11 ${delta > 0 ? 'is-up' : delta < 0 ? 'is-dn' : ''}`}>
            {delta > 0 ? '▲' : delta < 0 ? '▼' : '·'} {delta > 0 ? '+' : ''}
            {delta}
          </span>
        )}
      </div>
    </Mod>
  )
}
