import { Mod } from './Rack.jsx'

const CELL_W = 26
const CELL_H = 24
const CELLS = 5
const W = CELLS * CELL_W + 24
const STRIP_H = 18
const H = CELL_H + STRIP_H + 8

// Density drawn as density: cell n holds n+1 hairlines, so the comb literally
// gets denser left to right. Below it, the target's real 16-step kick row.
export default function GrooveComb({ route, target }) {
  const level = route ? route.vegetation : 0
  const kick = target ? target.grooveGrid[0] : null

  return (
    <Mod name="Groove" sub={route ? `${level}/5` : '—'} className="mod--comb">
      <svg className="comb" viewBox={`0 0 ${W} ${H}`}>
        <line className="comb__base" x1="0" y1={CELL_H} x2={CELLS * CELL_W} y2={CELL_H} />
        {Array.from({ length: CELLS }).map((_, c) => {
          const on = c < level
          const n = c + 1
          const hgt = 8 + c * 3
          return (
            <g key={c} className={`comb__cell${on ? ' is-on' : ''}`}>
              {Array.from({ length: n }).map((_, k) => {
                const x = c * CELL_W + ((k + 1) / (n + 1)) * (CELL_W - 4) + 2
                return <line key={k} x1={x} y1={CELL_H} x2={x} y2={CELL_H - hgt} />
              })}
            </g>
          )
        })}

        {/* the target's actual kick pattern, 16 steps */}
        {kick && (
          <g className="comb__strip">
            <line x1="0" y1={H - 2} x2={16 * 6} y2={H - 2} />
            {kick.map((v, i) => {
              const x = i * 6 + 1.5
              const hgt = 1 + v * 7
              return (
                <g key={i}>
                  {i % 4 === 0 && (
                    <line className="comb__sep" x1={x - 1.5} y1={H - 2} x2={x - 1.5} y2={H - 12} />
                  )}
                  <line className="comb__step" x1={x} y1={H - 2} x2={x} y2={H - 2 - hgt} />
                </g>
              )
            })}
          </g>
        )}
      </svg>
    </Mod>
  )
}
