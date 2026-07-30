import { Mod } from './Rack.jsx'
import { CHANNELS, CH_META } from '../map/layers.js'

const RAIL = 132
const ROW_H = 20

// All three raw feature distances side by side — the quiet confirmation of what
// the big channel rail just did. The active channel's row lifts to its hue; the
// other two sit at phosphor-ghost. 0.7→1.0 is a hatched danger zone.
export default function LockStrip({ channel, route }) {
  const vals = route
    ? { style: route.style01, harm: route.harm01, groove: route.groove01 }
    : { style: 0, harm: 0, groove: 0 }

  return (
    <Mod name="Channel Lock" sub={route ? 'Δ 0–1' : '—'} className="mod--lock">
      <svg className="lock" viewBox={`0 0 200 ${CHANNELS.length * ROW_H + 6}`}>
        <defs>
          <pattern id="lh" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="7" strokeWidth="0.7" stroke="var(--warn)" strokeOpacity="0.35" />
          </pattern>
        </defs>
        {CHANNELS.map((k, i) => {
          const y = i * ROW_H + 12
          const v = vals[k]
          const on = k === channel
          const danger = v >= 0.7
          const x0 = 46
          return (
            <g key={k} className={`lock__row${on ? ' is-on' : ''}${danger ? ' is-danger' : ''}`} data-ch={k}>
              <text className="lock__lbl w" x="0" y={y + 3}>
                {CH_META[k].name}
              </text>
              <line className="lock__rail" x1={x0} y1={y} x2={x0 + RAIL} y2={y} />
              {/* danger zone 0.7→1 */}
              <rect
                className="lock__danger"
                x={x0 + RAIL * 0.7}
                y={y - 3}
                width={RAIL * 0.3}
                height="6"
                fill="url(#lh)"
              />
              {[0, 0.5, 1].map((t) => (
                <line key={t} className="lock__tick" x1={x0 + RAIL * t} y1={y + 2} x2={x0 + RAIL * t} y2={y + 5} />
              ))}
              <rect
                className="lock__mk"
                x={x0 + RAIL * v - 3}
                y={y - 3}
                width="6"
                height="6"
              />
              <text className="lock__v n n--10" x="200" y={y + 3} textAnchor="end">
                {v.toFixed(2)}
              </text>
            </g>
          )
        })}
      </svg>
    </Mod>
  )
}
