import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { setupCanvas, canvasDpr } from './palette.js'
import { makeProject } from './projection.js'
import { cssVar } from './neon.js'
import { getChannel, channelDistance, CH_META } from './layers.js'
import {
  drawStyleWaterfall,
  drawHarmContours,
  drawGrooveLattice,
  drawGraticule,
} from './terrainRenderers.js'
import { TRACKS } from '../data/tracks.js'
import { altitudeAt } from '../utils/metaphor.js'
import NodeGlyph from './NodeGlyph.jsx'
import RouteTrace from './RouteTrace.jsx'
import ChannelRail from './ChannelRail.jsx'

// ─────────────────────────────────────────────────────────────────────────────
// The scope screen. Graticule + one of three terrain paints + an SVG layer of
// nodes, edges, cursors and the route trace. Terrain canvases are cached per
// (channel, size) and only ever blitted, so switching channels repaints nothing.
// ─────────────────────────────────────────────────────────────────────────────
export default function ScopeMap({
  channel,
  onChannel,
  camp,
  target,
  route,
  hoverTargetId,
  onHover,
  onSelect,
}) {
  const wellRef = useRef(null)
  const gratRef = useRef(null)
  const terrRef = useRef(null)
  const cacheRef = useRef({})
  const [size, setSize] = useState({ w: 0, h: 0 })

  useLayoutEffect(() => {
    const el = wellRef.current
    if (!el) return undefined
    const ro = new ResizeObserver((e) => {
      const { width, height } = e[0].contentRect
      setSize({ w: Math.round(width), h: Math.round(height) })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const { w, h } = size
  const project = useMemo(() => makeProject(w, h), [w, h])
  const C = useMemo(() => getChannel(channel), [channel])

  // graticule: painted once per size
  useLayoutEffect(() => {
    if (!w || !h) return
    const ctx = setupCanvas(gratRef.current, w, h)
    ctx.clearRect(0, 0, w, h)
    drawGraticule(ctx, w, h, cssVar('--graticule', '#0e1f26'), cssVar('--graticule-major', '#16323c'))
  }, [w, h])

  // Drop cached paints when the well is resized — they're size-specific, so
  // keeping them would grow memory without ever being reused again.
  useLayoutEffect(() => {
    cacheRef.current = {}
  }, [w, h])

  // terrain: painted once per (channel, size) into an offscreen cache, then blitted
  useLayoutEffect(() => {
    if (!w || !h) return
    // key on DPR too: dragging the window between a 2× and a 1× display keeps
    // w/h identical, so a DPR-blind key would blit a wrong-scale bitmap
    const key = `${channel}:${w}x${h}@${canvasDpr()}`
    let off = cacheRef.current[key]
    if (!off) {
      off = document.createElement('canvas')
      const octx = setupCanvas(off, w, h)
      const hue = cssVar(CH_META[channel].varName, '#2bf5ff')
      if (channel === 'style') drawStyleWaterfall(octx, w, h, project, hue)
      else if (channel === 'harm') drawHarmContours(octx, w, h, project, hue, C.contours)
      else drawGrooveLattice(octx, w, h, project, hue)
      cacheRef.current[key] = off
    }
    // Blit the cached paint — switching channels costs one drawImage, not
    // thousands of strokes. Drawn in CSS-pixel space with an explicit
    // destination size so geometry stays aligned with the SVG layer even if the
    // source bitmap's scale ever differs.
    const ctx = setupCanvas(terrRef.current, w, h)
    ctx.clearRect(0, 0, w, h)
    ctx.drawImage(off, 0, 0, w, h)
  }, [channel, C, w, h, project])

  const pos = useMemo(() => {
    const m = {}
    for (const t of TRACKS) m[t.id] = project(t.x, t.y)
    return m
  }, [project])

  const campXY = pos[camp.id]
  const tgt = target && target.id !== camp.id ? target : null
  const hovered = hoverTargetId ? TRACKS.find((t) => t.id === hoverTargetId) : null
  const chDelta = tgt ? channelDistance(channel, camp, tgt) : null

  // keyboard: 1/2/3 patch the beam, [ ] cycle
  useEffect(() => {
    const onKey = (e) => {
      if (e.target && /^(INPUT|TEXTAREA)$/.test(e.target.tagName)) return
      const map = { 1: 'style', 2: 'harm', 3: 'groove' }
      if (map[e.key]) {
        onChannel(map[e.key])
        return
      }
      if (e.key === '[' || e.key === ']') {
        const order = ['style', 'harm', 'groove']
        const i = order.indexOf(channel)
        const next = e.key === ']' ? (i + 1) % 3 : (i + 2) % 3
        onChannel(order[next])
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [channel, onChannel])

  return (
    <div className="well" ref={wellRef} data-ch={channel}>
      <canvas ref={gratRef} className="well__c" aria-hidden="true" />
      {/* remounted on channel change so the retrace clip animation replays */}
      <canvas ref={terrRef} className="well__c well__c--terr" key={channel} aria-hidden="true" />
      <div className="well__beam" key={`beam-${channel}`} aria-hidden="true" />

      {w > 0 && (
        <svg className="well__svg" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
          {/* this channel's connection graph */}
          <g className={`ed ed--${channel}`}>
            {C.edges.map((e) => {
              const [x1, y1] = pos[e.a]
              const [x2, y2] = pos[e.b]
              if (channel === 'groove') {
                // tempo rails: two parallel hairlines + rungs = a ladder
                const dx = x2 - x1
                const dy = y2 - y1
                const len = Math.hypot(dx, dy) || 1
                const nx = (-dy / len) * 1.6
                const ny = (dx / len) * 1.6
                const rungs = []
                for (let d = 12; d < len - 6; d += 12) {
                  const t = d / len
                  const bx = x1 + dx * t
                  const by = y1 + dy * t
                  rungs.push(
                    <line key={d} x1={bx - nx} y1={by - ny} x2={bx + nx} y2={by + ny} />,
                  )
                }
                return (
                  <g key={`${e.a}-${e.b}`} className="ed__rail">
                    <line x1={x1 - nx} y1={y1 - ny} x2={x2 - nx} y2={y2 - ny} />
                    <line x1={x1 + nx} y1={y1 + ny} x2={x2 + nx} y2={y2 + ny} />
                    {rungs}
                  </g>
                )
              }
              return (
                <line
                  key={`${e.a}-${e.b}`}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  strokeOpacity={channel === 'style' ? 0.34 : 0.16 + e.w * 0.34}
                />
              )
            })}
          </g>

          {/* scope measurement cursors on hover */}
          {hovered && hovered.id !== camp.id && (
            <g className="cur">
              <line x1={pos[hovered.id][0]} y1={0} x2={pos[hovered.id][0]} y2={h} />
              <line x1={0} y1={pos[hovered.id][1]} x2={w} y2={pos[hovered.id][1]} />
              <text className="n n--10 cur__v" x={pos[hovered.id][0] + 5} y={13}>
                Δ{channelDistance(channel, camp, hovered).toFixed(2)}
              </text>
              <text className="n n--10 cur__v" x={6} y={pos[hovered.id][1] - 5}>
                {Math.round(altitudeAt(hovered.x, hovered.y))} m
              </text>
            </g>
          )}

          {/* trigger marker: the camp's x, on the top edge, like a scope trigger */}
          <g className="trig">
            <path
              d={`M${campXY[0] - 5} 2 L${campXY[0] + 5} 2 L${campXY[0]} 9 Z`}
              className="trig__tri"
            />
            <line x1={campXY[0]} y1={9} x2={campXY[0]} y2={campXY[1] - 13} className="trig__drop" />
            <text className="w w--lo trig__lbl" x={campXY[0] + 9} y={9}>
              TRIG · CAMP
            </text>
          </g>

          {tgt && route && <RouteTrace from={campXY} to={pos[tgt.id]} route={route} />}

          {/* nodes */}
          {TRACKS.map((t) => {
            const [x, y] = pos[t.id]
            const isCamp = t.id === camp.id
            const isTgt = tgt && t.id === tgt.id
            const named = isCamp || isTgt || t.id === hoverTargetId
            return (
              <g
                key={t.id}
                className={`nd${isCamp ? ' is-camp' : ''}${isTgt ? ' is-target' : ''}`}
                transform={`translate(${x} ${y})`}
                onMouseEnter={() => !isCamp && onHover(t.id)}
                onMouseLeave={() => onHover(null)}
                onClick={() => onSelect(t.id)}
                role="button"
                tabIndex={0}
                aria-label={t.title}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    onSelect(t.id)
                  }
                }}
              >
                <circle r="17" fill="transparent" />
                <NodeGlyph channel={channel} track={t} v={C.nodeValue[t.id]} isCamp={isCamp} />
                {named ? (
                  <text className="nd__name w w--hi" y="-17" textAnchor="middle">
                    {t.title}
                  </text>
                ) : (
                  <text className="nd__no n n--10" y="19" textAnchor="middle">
                    {String(t.trackNo).padStart(2, '0')}
                  </text>
                )}
              </g>
            )
          })}
        </svg>
      )}

      <ChannelRail channel={channel} onChannel={onChannel} />

      {/* channel identity, typed in on switch */}
      <div className="well__ch" key={`lbl-${channel}`}>
        <span className="w w--13 w--hi">
          {C.ch} · {C.name}
        </span>
        <span className="w w--lo">{C.axis}</span>
      </div>

      {/* camp identity, bottom-left */}
      <div className="well__camp">
        <span className="w w--13 w--hi">{camp.title}</span>
        <span className="w w--lo">{camp.region}</span>
        {tgt && (
          <span className="well__camp-d n n--10">
            → {tgt.title} · Δ{chDelta.toFixed(2)} {C.name}
          </span>
        )}
      </div>

      <div className="well__reveal w w--lo">{C.reveals}</div>

      <div className="well__scan" aria-hidden="true" />
      <div className="well__vig" aria-hidden="true" />
      <span className="well__L tl" aria-hidden="true" />
      <span className="well__L tr" aria-hidden="true" />
      <span className="well__L bl" aria-hidden="true" />
      <span className="well__L br" aria-hidden="true" />
    </div>
  )
}
