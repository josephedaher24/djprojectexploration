import { Mod } from './Rack.jsx'
import { useSpring } from '../clock/useSpring.js'
import { TIERS } from '../utils/metaphor.js'

const W = 200
const CELL_H = 14
const GAP = 3
const LADDER_W = 148

// Five stacked open rectangles. The active tier is never *filled* — it is
// hatched, so the whole instrument stays line-art. A continuous 0..1 scale runs
// down the right with an open pointer at diff01.
export default function GradeLadder({ route }) {
  const grade = route ? route.tier.grade : null
  const diff = route ? route.diff01 : 0
  const H = TIERS.length * (CELL_H + GAP)
  // Map diff01 into the ACTIVE tier's own cell, so the pointer always sits beside
  // the row that is hatched rather than drifting into the neighbouring band.
  const tierIdx = Math.min(TIERS.length - 1, Math.floor(diff * TIERS.length))
  const frac = Math.min(1, Math.max(0, diff * TIERS.length - tierIdx))
  const rowTop = (TIERS.length - 1 - tierIdx) * (CELL_H + GAP)
  const pointerY = useSpring(rowTop + (1 - frac) * CELL_H, { stiffness: 120, damping: 16 })
  const ptrY = Math.max(4, Math.min(H - 4, pointerY)) // never clip the arrow
  const tone = grade === 'T5' ? 'is-alert' : grade === 'T4' ? 'is-warn' : ''
  const hatch =
    grade === 'T5' ? 'var(--alert)' : grade === 'T4' ? 'var(--warn)' : 'var(--hue)'

  return (
    <Mod name="Grade" sub={route ? `${(diff * 100).toFixed(0)}%` : '—'} className={`mod--grade ${tone}`}>
      <svg className="grade" viewBox={`0 0 ${W} ${H}`}>
        <defs>
          {/* the active tier is hatched, never filled — keeps it line-art */}
          <pattern id="gh" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" strokeWidth="0.8" stroke={hatch} />
          </pattern>
        </defs>

        {/* T5 at the top → T1 at the bottom, like a real difficulty ladder */}
        {TIERS.map((t, i) => {
          const y = (TIERS.length - 1 - i) * (CELL_H + GAP)
          const on = t.grade === grade
          return (
            <g key={t.grade} className={`grade__cell${on ? ' is-on' : ''}`}>
              <text className="grade__t n n--10" x="0" y={y + CELL_H - 4}>
                {t.grade}
              </text>
              <rect x="22" y={y} width={LADDER_W} height={CELL_H} />
              {on && <rect className="grade__hatch" x="22" y={y} width={LADDER_W} height={CELL_H} fill="url(#gh)" />}
            </g>
          )
        })}

        {/* continuous scale + open pointer */}
        <g className="grade__scale">
          <line x1={W - 8} y1="0" x2={W - 8} y2={H} />
          {Array.from({ length: 11 }).map((_, i) => (
            <line key={i} x1={W - 8} y1={(i / 10) * H} x2={W - 4} y2={(i / 10) * H} />
          ))}
          <path
            className="grade__ptr"
            d={`M${W - 10} ${ptrY} L${W - 17} ${ptrY - 3.5} L${W - 17} ${ptrY + 3.5} Z`}
          />
        </g>
      </svg>
      <div className="mod__row">
        <span className="n n--hero">{grade || '--'}</span>
        <span className="w w--11">{route ? route.tier.name : 'no route'}</span>
      </div>
    </Mod>
  )
}
