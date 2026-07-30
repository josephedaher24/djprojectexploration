import { Mod } from './Rack.jsx'

const W = 200
const H = 86
const PAD_T = 14
const PAD_B = 12

// The route's elevation profile as a scope trace: no fill under the curve ever —
// density comes from hairline drops. Min/max get measurement cursors and the
// climb delta gets a square bracket, like a real instrument's cursor readout.
export default function AltScope({ camp, route }) {
  // Scale to the profile's OWN range so the trace uses the full well; a flat
  // route would otherwise collapse to a line at one edge.
  const p = route ? route.profile : null
  const rawLo = p ? route.minAlt : 0
  const rawHi = p ? route.maxAlt : 1
  const pad = Math.max(8, (rawHi - rawLo) * 0.18)
  const lo = rawLo - pad
  const hi = rawHi + pad
  const span = hi - lo || 1
  const yOf = (v) => H - PAD_B - ((v - lo) / span) * (H - PAD_T - PAD_B)
  const xOf = (i) => (i / ((p ? p.length : 2) - 1)) * W

  const pts = p ? p.map((v, i) => [xOf(i), yOf(v)]) : []
  const d = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ')
  const iMin = p ? p.indexOf(Math.min(...p)) : 0
  const iMax = p ? p.indexOf(Math.max(...p)) : 0
  // Anchor the reference to the corridor's OWN first sample — that is the camp's
  // altitude in field space, and is in-domain by construction. (The track's
  // nominal `altitude` comes from a different, un-smoothed metric, so using it
  // here pinned the line to the frame on flat routes.)
  const campField = p ? p[0] : null
  const campY = p ? yOf(campField) : H / 2
  const dh = route ? route.netAltitude : 0

  return (
    <Mod name="Alt Scope" sub={route ? `${route.profile.length} SMP` : '—'} className="mod--alt">
      <svg className="alt" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        {/* mini graticule 10×6 */}
        <g className="alt__grat">
          {Array.from({ length: 9 }).map((_, i) => (
            <line key={`v${i}`} x1={((i + 1) / 10) * W} y1="0" x2={((i + 1) / 10) * W} y2={H} />
          ))}
          {Array.from({ length: 5 }).map((_, i) => (
            <line key={`h${i}`} x1="0" y1={((i + 1) / 6) * H} x2={W} y2={((i + 1) / 6) * H} />
          ))}
        </g>

        {p && (
          <>
            {/* hairline drops every 4th sample — density without a fill */}
            <g className="alt__drop">
              {pts.map(([x, y], i) => (i % 4 === 0 ? <line key={i} x1={x} y1={y} x2={x} y2={H - PAD_B} /> : null))}
            </g>

            {/* camp altitude reference */}
            <line className="alt__ref" x1="0" y1={campY} x2={W} y2={campY} />
            <text className="alt__reflbl n n--10" x="2" y={campY - 3}>
              START {Math.round(campField)}
            </text>

            {/* min / max cursors, each capped with an open triangle */}
            {[iMin, iMax].map((idx, k) => {
              const [x, y] = pts[idx]
              return (
                <g className="alt__cur" key={k}>
                  <line x1={x} y1={y} x2={x} y2={H - PAD_B} />
                  <path d={`M${x - 3.5} ${y - 5} L${x + 3.5} ${y - 5} L${x} ${y - 0.5} Z`} />
                </g>
              )
            })}

            {/* ΔH measurement bracket */}
            <g className="alt__brk">
              <path
                d={`M${Math.min(pts[iMin][0], pts[iMax][0])} 9 V4 H${Math.max(
                  pts[iMin][0],
                  pts[iMax][0],
                )} V9`}
              />
            </g>

            {/* the trace: knock-out beneath, hairline core on top */}
            <path className="alt__knock" d={d} />
            <path className="alt__line" d={d} />
          </>
        )}
      </svg>

      <div className="mod__row">
        <span className="n n--hero">
          {camp.altitude}
          <span className="u">m</span>
        </span>
        <span className="alt__dh n n--11">
          ΔH {dh >= 0 ? '+' : ''}
          {dh} m
        </span>
      </div>
    </Mod>
  )
}
