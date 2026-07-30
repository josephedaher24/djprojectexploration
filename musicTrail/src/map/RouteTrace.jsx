// The connection between two tracks IS a waveform: the route's own 24-sample
// elevation profile, displaced perpendicular to the camp→target axis. So a hard
// climb doesn't just *say* 420 m — the line between the two points visibly
// buckles. A beam-white packet rides it in tempo, leaving a phosphor tail.
import { useMemo } from 'react'

const AMP = 26

export default function RouteTrace({ from, to, route }) {
  const geom = useMemo(() => {
    const [fx, fy] = from
    const [tx, ty] = to
    const dx = tx - fx
    const dy = ty - fy
    const len = Math.hypot(dx, dy) || 1
    const ux = dx / len
    const uy = dy / len
    const nx = -uy // perpendicular
    const ny = ux

    // Detrend against the direct A→B climb: what bulges is the terrain you'd
    // actually have to cross, not the fact that B is higher than A. A steady
    // climb therefore reads straight; a route over a ridge visibly buckles.
    const p = route.profile
    const last = p.length - 1
    const dev = p.map((v, i) => v - (p[0] + (p[last] - p[0]) * (i / last)))
    // AUTO vertical gain, like a scope: normalise to the deviation's own peak so
    // the shape is always legible, and report the gain (below) so the scale is
    // never implied. Genuinely flat corridors stay flat.
    const peak = Math.max(...dev.map(Math.abs))
    const flat = peak < 2 // metres — nothing to show
    const scale = flat ? 0 : 1 / peak

    const offsetOf = (v) => Math.max(-1, Math.min(1, v * scale))
    const at = (t, offsetNorm) => {
      const bx = fx + dx * t
      const by = fy + dy * t
      const o = offsetNorm * AMP
      return [bx + nx * o, by + ny * o]
    }

    const pts = dev.map((v, i) => at(i / last, offsetOf(v)))
    const d = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ')

    // sample the trace at an arbitrary t for labels / ticks
    const sampleAt = (t) => {
      const idx = Math.min(last, Math.max(0, t * last))
      const i0 = Math.floor(idx)
      const i1 = Math.min(last, i0 + 1)
      const f = idx - i0
      const v = dev[i0] + (dev[i1] - dev[i0]) * f
      return { pt: at(t, offsetOf(v)), nx, ny }
    }

    return { d, pts, sampleAt, peak, flat }
  }, [from, to, route])

  const label = geom.sampleAt(0.6)

  return (
    <g className="rt">
      {/* knock-out beneath so the trace stays readable over terrain */}
      <path className="rt__knock" d={geom.d} />
      <path className="rt__line" d={geom.d} />

      {/* waypoints on the corridor get a perpendicular tick + their number */}
      {route.waypoints.map((wpt) => {
        const s = geom.sampleAt(wpt.proj)
        const [x, y] = s.pt
        return (
          <g key={wpt.track.id} className="rt__wpt">
            <line
              x1={x - s.nx * 4}
              y1={y - s.ny * 4}
              x2={x + s.nx * 4}
              y2={y + s.ny * 4}
            />
            <text className="n n--10" x={x + s.nx * 9} y={y + s.ny * 9 + 3} textAnchor="middle">
              {wpt.track.trackNo}
            </text>
          </g>
        )
      })}

      {/* the packet: one bar per traverse, tempo-locked, with a phosphor tail */}
      <g className="rt__packet" style={{ offsetPath: `path("${geom.d}")` }}>
        <rect x="-1.4" y="-1.4" width="2.8" height="2.8" />
      </g>
      {[0.5, 0.3, 0.18, 0.09].map((a, i) => (
        <g
          key={i}
          className="rt__packet rt__packet--tail"
          style={{
            offsetPath: `path("${geom.d}")`,
            animationDelay: `calc(var(--beat) * -${(i + 1) * 0.06})`,
            opacity: a,
          }}
        >
          <rect x="-1" y="-1" width="2" height="2" />
        </g>
      ))}

      {/* measured distance + the trace's vertical gain — no pill, no plate */}
      <text
        className="rt__dist n n--11"
        x={label.pt[0]}
        y={label.pt[1] - 7}
        textAnchor="middle"
      >
        {route.distanceKm} km
      </text>
      <text
        className="rt__gain n n--10"
        x={label.pt[0]}
        y={label.pt[1] + 12}
        textAnchor="middle"
      >
        {geom.flat ? 'FLAT' : `±${Math.round(geom.peak)} m/div`}
      </text>
    </g>
  )
}
