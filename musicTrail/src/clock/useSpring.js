import { useEffect, useRef, useState } from 'react'

// A tiny velocity spring so gauge needles/hands *swing and settle* (with a little
// overshoot) instead of snapping — the thing that makes a skeuomorphic instrument
// feel physical. `wrap` takes the short way around a 0..360 dial.
export function useSpring(target, { stiffness = 120, damping = 14, wrap = false } = {}) {
  const [value, setValue] = useState(target)
  const pos = useRef(target)
  const vel = useRef(0)
  const raf = useRef(0)
  const [reduced, setReduced] = useState(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  )

  // React to a live prefers-reduced-motion toggle (snaps any in-flight swing).
  useEffect(() => {
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)')
    const onChange = () => setReduced(mql.matches)
    mql.addEventListener('change', onChange)
    return () => mql.removeEventListener('change', onChange)
  }, [])

  useEffect(() => {
    cancelAnimationFrame(raf.current)
    // Honor reduced-motion the same way the heartbeat clock does: snap, don't swing.
    if (reduced) {
      pos.current = target
      vel.current = 0
      setValue(target)
      return undefined
    }
    let last = null
    const step = (now) => {
      if (last == null) last = now
      const dt = Math.min(0.032, (now - last) / 1000)
      last = now

      let delta = target - pos.current
      if (wrap) delta = ((delta + 540) % 360) - 180

      const accel = stiffness * delta - damping * vel.current
      vel.current += accel * dt
      pos.current += vel.current * dt

      if (Math.abs(delta) < 0.05 && Math.abs(vel.current) < 0.05) {
        pos.current = target
        vel.current = 0
        setValue(target)
        return
      }
      setValue(pos.current)
      raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
  }, [target, stiffness, damping, wrap, reduced])

  return value
}
