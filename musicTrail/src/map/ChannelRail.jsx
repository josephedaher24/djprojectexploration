import { CHANNELS, CH_META } from './layers.js'

// A channel rail, not tabs: three open brackets down the left edge of the well.
// Each carries a live hairline micro-preview of what that probe draws, so you can
// see the shape of the channel before you patch the beam to it.
const PREVIEW = {
  // a mini waterfall ridge
  style: (
    <g>
      {[0, 1, 2, 3, 4].map((r) => {
        const y = 3 + r * 2.2
        const amp = 3.2 - r * 0.5
        return (
          <path
            key={r}
            d={`M0 ${y} Q10 ${y - amp} 20 ${y - amp * 0.4} T40 ${y - amp * 0.8} T60 ${y}`}
          />
        )
      })}
    </g>
  ),
  // three nested isolines
  harm: (
    <g>
      <path d="M6 12 Q18 2 32 7 T56 4" />
      <path d="M10 12 Q20 6 32 10 T52 8" />
      <path d="M14 13 Q22 9 31 12 T48 11" />
    </g>
  ),
  // a 16-tick comb
  groove: (
    <g>
      <path d="M2 13 H58" />
      {Array.from({ length: 16 }).map((_, i) => {
        const x = 3 + i * 3.6
        const h = i % 4 === 0 ? 9 : 3 + ((i * 5) % 4)
        return <path key={i} d={`M${x} 13 V${13 - h}`} />
      })}
    </g>
  ),
}

const KEYNUM = { style: '1', harm: '2', groove: '3' }

export default function ChannelRail({ channel, onChannel }) {
  return (
    <div className="chrail" role="radiogroup" aria-label="Scope channel">
      {CHANNELS.map((k) => {
        const m = CH_META[k]
        const on = k === channel
        return (
          <button
            key={k}
            type="button"
            role="radio"
            aria-checked={on}
            tabIndex={on ? 0 : -1}
            className={`chrail__item${on ? ' is-on' : ''}`}
            data-ch={k}
            onClick={() => onChannel(k)}
          >
            <span className="chrail__bar" aria-hidden="true" />
            <span className="chrail__body">
              <span className="chrail__code w">
                {m.ch} <b>{m.name}</b>
              </span>
              <svg className="chrail__prev" viewBox="0 0 60 14" fill="none" aria-hidden="true">
                {PREVIEW[k]}
              </svg>
            </span>
            <span className="chrail__key n n--10">{KEYNUM[k]}</span>
            <span className="chrail__probe" aria-hidden="true" />
          </button>
        )
      })}
    </div>
  )
}
