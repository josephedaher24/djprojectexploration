import { useMemo } from 'react'
import { Mod } from './Rack.jsx'
import { useSpring } from '../clock/useSpring.js'
import { compassLabel } from '../utils/metaphor.js'

const S = 76
const C = S / 2
const R = 30

// No dial face, no bezel: the ring exists only as 36 radial ticks. The needle is
// one line and an open V. The ticks nearest the needle light up.
export default function BearingRose({ route }) {
  const bearing = route ? route.bearing : 0
  const angle = useSpring(bearing, { stiffness: 110, damping: 14, wrap: true })

  const ticks = useMemo(() => {
    const out = []
    for (let i = 0; i < 36; i++) {
      const deg = i * 10
      const a = (deg - 90) * (Math.PI / 180)
      const card = deg % 90 === 0
      const len = card ? 7 : 4
      out.push({
        deg,
        card,
        x1: C + Math.cos(a) * (R - len),
        y1: C + Math.sin(a) * (R - len),
        x2: C + Math.cos(a) * R,
        y2: C + Math.sin(a) * R,
      })
    }
    return out
  }, [])

  const cards = [
    ['N', 0],
    ['E', 90],
    ['S', 180],
    ['W', 270],
  ]
  const rad = (angle - 90) * (Math.PI / 180)
  const nx = C + Math.cos(rad) * 26
  const ny = C + Math.sin(rad) * 26
  const sx = C - Math.cos(rad) * 8
  const sy = C - Math.sin(rad) * 8
  // open V arrowhead at the needle tip
  const vA = rad + 2.5
  const vB = rad - 2.5

  return (
    <Mod name="Bearing" sub={route ? 'CAMP→TGT' : '—'} className="mod--rose">
      <div className="rose">
        <svg className="rose__svg" viewBox={`0 0 ${S} ${S}`}>
          <g className="rose__ticks">
            {ticks.map((t) => {
              // angular distance from the needle; light the tick under it plus
              // its two neighbours (the spring can overshoot outside 0..360, so
              // the +540 wrap keeps this correct)
              const dist = Math.abs(((t.deg - angle + 540) % 360) - 180)
              const lit = dist < 15
              return (
                <line
                  key={t.deg}
                  x1={t.x1}
                  y1={t.y1}
                  x2={t.x2}
                  y2={t.y2}
                  className={`${t.card ? 'is-card' : ''} ${lit ? 'is-lit' : ''}`}
                />
              )
            })}
          </g>
          {cards.map(([c, deg]) => {
            const a = (deg - 90) * (Math.PI / 180)
            return (
              <text
                key={c}
                className="rose__card w w--lo"
                x={C + Math.cos(a) * (R + 7)}
                y={C + Math.sin(a) * (R + 7) + 3}
                textAnchor="middle"
              >
                {c}
              </text>
            )
          })}
          <g className="rose__needle">
            <line x1={sx} y1={sy} x2={nx} y2={ny} />
            <path
              d={`M${nx + Math.cos(vA) * -7} ${ny + Math.sin(vA) * -7} L${nx} ${ny} L${
                nx + Math.cos(vB) * -7
              } ${ny + Math.sin(vB) * -7}`}
            />
          </g>
        </svg>
        <div className="rose__read">
          <span className="n n--hero">{route ? String(Math.round(bearing)).padStart(3, '0') : '---'}°</span>
          <span className="w w--lo">{route ? compassLabel(bearing) : 'no target'}</span>
        </div>
      </div>
    </Mod>
  )
}
