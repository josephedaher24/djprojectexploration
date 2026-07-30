import { useEffect, useMemo, useState } from 'react'
import { ClockProvider } from './clock/ClockProvider.jsx'
import { START_ID, SET_ORDER, getTrack, TRACKS } from './data/tracks.js'
import { buildRoute, compassLabel } from './utils/metaphor.js'
import { getChannel, CH_META } from './map/layers.js'
import ScopeMap from './map/ScopeMap.jsx'
import Rack from './instruments/Rack.jsx'

// The next suggested leg, so the rig always has something to measure.
function nextInSet(campId, visited) {
  const start = SET_ORDER.indexOf(campId)
  for (let k = 1; k <= SET_ORDER.length; k++) {
    const id = SET_ORDER[(start + k) % SET_ORDER.length]
    if (!visited.has(id)) return id
  }
  return null
}

export default function App() {
  const [channel, setChannel] = useState('style')
  const [campId, setCampId] = useState(START_ID)
  const [hoverTargetId, setHoverTargetId] = useState(null)
  const [visitedIds, setVisitedIds] = useState(() => new Set([START_ID]))

  const camp = getTrack(campId)
  const suggestedId = useMemo(() => nextInSet(campId, visitedIds), [campId, visitedIds])
  const targetId = hoverTargetId || suggestedId
  const target = targetId ? getTrack(targetId) : null
  const route = useMemo(() => buildRoute(camp, target), [camp, target])
  const C = useMemo(() => getChannel(channel), [channel])

  // the active channel re-hues the whole rig through one attribute
  useEffect(() => {
    document.documentElement.setAttribute('data-ch', channel)
  }, [channel])

  const onSelect = (id) => {
    if (id === campId) return
    setCampId(id)
    setVisitedIds((v) => new Set(v).add(id))
    setHoverTargetId(null)
  }

  const tgt = target && target.id !== camp.id ? target : null

  return (
    <ClockProvider bpm={camp.bpm}>
      <div className="rig">
        <header className="rig__head">
          <div className="rig__brand">
            <h1>TRAILSCOPE</h1>
            <span className="ch">
              {C.ch} · {C.name}
            </span>
          </div>
          <div className="rig__stat">
            <span className="n n--10 n--lo">
              {visitedIds.size}/{TRACKS.length} ACQ
            </span>
            <span className="rig__acq">acquiring</span>
          </div>
        </header>

        <div className="rig__well">
          <ScopeMap
            channel={channel}
            onChannel={setChannel}
            camp={camp}
            target={target}
            route={route}
            hoverTargetId={hoverTargetId}
            onHover={setHoverTargetId}
            onSelect={onSelect}
          />
        </div>

        <div className="rig__rack">
          <Rack channel={channel} camp={camp} target={target} route={route} />
        </div>

        <footer className="rig__foot">
          <span className="k">probe</span>
          <span className="hue">
            {C.ch} {C.name}
          </span>
          <span className="sep">│</span>
          <span className="k">camp</span>
          <span className="v">{camp.title}</span>
          <span className="sep">│</span>
          <span className="k">tgt</span>
          <span className="tgt">{tgt ? tgt.title : '—'}</span>
          {route && (
            <>
              <span className="sep">│</span>
              <span className="v">
                {route.distanceKm}km {route.ascentM}m {route.bars}b
              </span>
              <span className="sep">│</span>
              <span className="v">
                {compassLabel(route.bearing)} {Math.round(route.bearing)}°
              </span>
              <span className="sep">│</span>
              <span className="tgt">{route.tier.grade}</span>
            </>
          )}
          <span className="keys">
            <b>1</b>
            <b>2</b>
            <b>3</b> patch · <b>[</b>
            <b>]</b> cycle
          </span>
        </footer>
      </div>
    </ClockProvider>
  )
}
