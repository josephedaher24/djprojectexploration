import { Mod } from './Rack.jsx'

const W = 180
const H = 30
const CX = W / 2
const CY = H / 2
const PAD = 6
const FULL = 12 // BPM at the labelled end ticks
const SPAN = CX - PAD // px of available deflection
const SCALE = SPAN / FULL // px per BPM — derived, so geometry can't drift from the labels
const STRONG = FULL / 2

const LABEL = { headwind: 'HEADWIND', tailwind: 'TAILWIND', calm: 'CALM AIR' }

// Three staggered streak lines blowing off a zero tick — moving air, not a bar
// chart. Tailwind blows left (with you), headwind right (against you).
export default function WindVector({ route }) {
  const wind = route ? route.wind : { delta: 0, mag: 0, kind: 'calm' }
  // direction follows the SIGN, not the category: |delta| <= 2 is reported as
  // "calm" but a -2 must still blow the opposite way from a +2
  const dir = wind.delta < 0 ? -1 : 1
  const len = Math.min(SPAN, wind.mag * SCALE)
  const over = wind.mag > FULL // pegged: deltas in this set reach ~36 BPM
  const strong = wind.mag > STRONG
  const rows = [
    { dy: -6, k: 1.0 },
    { dy: 0, k: 0.82 },
    { dy: 6, k: 0.9 },
  ]

  return (
    <Mod name="Wind" sub="TEMPO Δ" className={`mod--wind${strong ? ' is-warn' : ''}`}>
      <svg className="wind" viewBox={`0 0 ${W} ${H}`}>
        <line className="wind__axis" x1={PAD} y1={CY} x2={W - PAD} y2={CY} />
        <line className="wind__zero" x1={CX} y1={CY - 8} x2={CX} y2={CY + 8} />
        <text className="wind__end n n--10" x={PAD} y={H - 1}>−{FULL}</text>
        <text className="wind__end n n--10" x={W - PAD} y={H - 1} textAnchor="end">+{FULL}</text>

        {/* pegged at full scale — say so rather than let it read as in-range */}
        {over && (
          <g className="wind__over">
            <path
              d={`M${CX + dir * (SPAN - 4)} ${CY - 9} L${CX + dir * SPAN} ${CY - 5} L${
                CX + dir * (SPAN - 4)
              } ${CY - 1}`}
            />
            <text
              className="n n--10"
              x={CX + dir * SPAN}
              y={CY - 12}
              textAnchor={dir > 0 ? 'end' : 'start'}
            >
              OVR
            </text>
          </g>
        )}

        {len > 1 &&
          rows.map((r, i) => {
            const L = len * r.k
            const ex = CX + dir * L
            const ch = 5
            return (
              <g className="wind__streak" key={i} style={{ '--len': `${L}px` }}>
                <line x1={CX} y1={CY + r.dy} x2={ex} y2={CY + r.dy} />
                <path
                  d={`M${ex - dir * ch} ${CY + r.dy - ch * 0.7} L${ex} ${CY + r.dy} L${
                    ex - dir * ch
                  } ${CY + r.dy + ch * 0.7}`}
                />
                {strong && (
                  <path
                    d={`M${ex - dir * ch * 2} ${CY + r.dy - ch * 0.7} L${ex - dir * ch} ${
                      CY + r.dy
                    } L${ex - dir * ch * 2} ${CY + r.dy + ch * 0.7}`}
                  />
                )}
              </g>
            )
          })}
      </svg>
      <div className="mod__row">
        <span className="w w--11">{LABEL[wind.kind]}</span>
        <span className="n n--16">
          {wind.delta > 0 ? '+' : ''}
          {wind.delta}
          <span className="u">bpm</span>
        </span>
      </div>
    </Mod>
  )
}
