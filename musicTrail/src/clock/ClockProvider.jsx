import { createContext, useContext, useEffect, useRef, useState } from 'react'

// ─────────────────────────────────────────────────────────────────────────────
// The one heartbeat / beat-grid of the whole terminal. A SINGLE requestAnimationFrame
// loop computes a beat phase (0..1 per beat) and a "pulse" envelope from the
// current Deck A BPM, then:
//   • writes --beat / --beat-phase / --pulse to :root  (CSS-driven consumers:
//     deck waveform playhead, VU meters, focal contour breathe, Deck-A rivet flash
//     — zero React re-renders)
//   • pushes the same numbers to registered frame callbacks (JS-driven consumers).
// So the map and the console pulse as one instrument at the track's tempo.
// ─────────────────────────────────────────────────────────────────────────────

const ClockContext = createContext(null)

// A realistic two-bump beat envelope over a single beat (0..1). Exported so the
// cadence trace can draw exactly the curve the clock is running.
export function heartEnvelope(p) {
  const lub = Math.exp(-((p - 0.02) ** 2) / (2 * 0.045 ** 2))
  const dub = 0.55 * Math.exp(-((p - 0.2) ** 2) / (2 * 0.05 ** 2))
  return Math.min(1, lub + dub)
}

export function ClockProvider({ bpm = 120, children }) {
  const bpmRef = useRef(bpm)
  const state = useRef({ t: 0, beatPhase: 0, pulse: 0, bpm })
  const subs = useRef(new Set())
  const [reduced, setReduced] = useState(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  )

  // Retime smoothly when the camp BPM changes (phase carries over, no jump).
  useEffect(() => {
    bpmRef.current = bpm
    document.documentElement.style.setProperty('--beat', `${(60 / (bpm || 120)).toFixed(4)}s`)
  }, [bpm])

  // Honor a mid-session prefers-reduced-motion toggle (re-runs the loop effect).
  useEffect(() => {
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)')
    const onChange = () => setReduced(mql.matches)
    mql.addEventListener('change', onChange)
    return () => mql.removeEventListener('change', onChange)
  }, [])

  useEffect(() => {
    const root = document.documentElement

    if (reduced) {
      // A single still beat marker — never fully dead, but no motion.
      const frozen = { t: 0, beatPhase: 0, pulse: 0.4, bpm: bpmRef.current }
      state.current = frozen
      root.style.setProperty('--beat-phase', '0')
      root.style.setProperty('--pulse', '0.4')
      subs.current.forEach((cb) => cb(frozen))
      return
    }

    let raf = 0
    let last = null
    let phase = 0
    const loop = (now) => {
      if (last == null) last = now
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      phase = (phase + dt * ((bpmRef.current || 120) / 60)) % 1
      const pulse = heartEnvelope(phase)
      const snap = { t: now / 1000, beatPhase: phase, pulse, bpm: bpmRef.current }
      state.current = snap
      root.style.setProperty('--beat-phase', phase.toFixed(4))
      root.style.setProperty('--pulse', pulse.toFixed(4))
      subs.current.forEach((cb) => cb(snap))
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [reduced])

  const api = useRef({
    subscribe(cb) {
      subs.current.add(cb)
      return () => subs.current.delete(cb)
    },
    get: () => state.current,
  }).current

  return <ClockContext.Provider value={api}>{children}</ClockContext.Provider>
}

export function useClock() {
  return useContext(ClockContext)
}

// Register a per-frame callback (t, beatPhase, pulse, bpm). Used by canvas-based
// consumers that draw every frame; everything else should just read the CSS vars.
export function useClockFrame(callback) {
  const clock = useClock()
  const ref = useRef(callback)
  ref.current = callback
  useEffect(() => {
    if (!clock) return undefined
    return clock.subscribe((s) => ref.current(s))
  }, [clock])
}
