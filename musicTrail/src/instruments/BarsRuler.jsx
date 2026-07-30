import { Mod } from './Rack.jsx'

const STEPS = 32
const PITCH = 6
const W = STEPS * PITCH
const H = 26

const mmss = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`

// A 32-step measurement ruler with a square bracket spanning exactly `bars`.
// Every 4th tick is a phrase boundary and gets labelled.
export default function BarsRuler({ route }) {
  const bars = route ? route.bars : 0
  return (
    <Mod name="Bars" sub={route ? `${STEPS} MAX` : '—'} className="mod--bars">
      <svg className="bars" viewBox={`0 0 ${W} ${H}`}>
        {/* bracket over the covered span */}
        {bars > 0 && (
          <path className="bars__brk" d={`M0.5 3 V9 H${bars * PITCH - 0.5} V3`} />
        )}
        <line className="bars__base" x1="0" y1={H - 10} x2={W} y2={H - 10} />
        {Array.from({ length: STEPS }).map((_, i) => {
          const x = i * PITCH + 0.5
          const phrase = i % 4 === 0
          const covered = i < bars
          return (
            <g key={i} className={`bars__t${covered ? ' is-on' : ''}`}>
              <line x1={x} y1={H - 10} x2={x} y2={H - 10 - (phrase ? 11 : 6)} />
              {phrase && i > 0 && (
                <text className="bars__lbl n n--10" x={x} y={H - 1} textAnchor="middle">
                  {i}
                </text>
              )}
            </g>
          )
        })}
      </svg>
      <div className="mod__row">
        <span className="n n--hero">
          {bars || '--'}
          <span className="u">bars</span>
        </span>
        <span className="n n--11 n--mid">{route ? `≈ ${mmss(route.seconds)}` : '—'}</span>
      </div>
    </Mod>
  )
}
