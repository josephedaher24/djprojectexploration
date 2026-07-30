# TRAILSCOPE — CH1 / CH2 / CH3

A **front-end design demo** that reads a PaCMAP song-similarity space as a
**vector-scope climbing instrument**. One rig, three probes: patch the beam to a
channel and the same 20 tracks are re-measured through a different feature
dimension — like switching layers in a Cyberpunk 2077 braindance.

> Pure React + mock data — no backend, no audio. Lives in its own folder and only
> *references* the parent `djprojectexploration` project (the metaphor, the warren
> tracklist). Front-end **visual design** only.

## The idea

A vector-graphics field instrument, not a UI. Everything on screen is a beam that
has been somewhere: the terrain is a sweep, the route is a trace, a number only
exists because a probe measured it. Cold black, hairline strokes, one hot beam and
the ghost it leaves behind. **Nothing is filled, nothing is glossy, nothing has a
bezel** — all information is carried by lines.

## The three channels

Switching channel is the point: the terrain, the node glyphs and the connection
graph are all re-derived, because each channel is a genuinely different dimension.
Of the three neighbour graphs (37 / 24 / 9 edges), only **one edge is shared by all
three** — the constellation really does rewire.

| | CH1 · STYLE | CH2 · HARM | CH3 · GROOVE |
| --- | --- | --- | --- |
| **hue** | `#2BF5FF` cyan | `#FF3DA6` magenta | `#C6FF3D` lime |
| **from** | MAEST timbre distance | Camelot key + energy | 16×3 onset flux + BPM |
| **terrain** | spectrum-analyser **waterfall** of the timbre-density field | **contour relief** traced from the real altitude field | warped 16-step **lattice** that pinches around percussive mass |
| **node** | a **blip** — cross + open diamond sized by energy | a **Camelot rosette** — 12 ticks, the track's own key extended | a **micro-sequencer** — the real 16×3 grid as line ticks |
| **edges** | kNN-3 mesh in style space | only keys **within one step** on the wheel | **tempo rails** — pairs you could phrase-lock without touching pitch |

Patch with the channel rail, the `1` `2` `3` keys, or `[` / `]` to cycle. Switching
plays a **retrace**: a 1px beam sweeps the well while the new frame clips in over
the old — one `clip-path` animation, zero repaints.

## The route is a waveform

The A→B connection isn't a straight line: it's the route's own 24-sample elevation
profile, **detrended against the direct climb** and displaced perpendicular to the
axis. So a steady climb reads straight while a route that crosses a ridge visibly
buckles. Vertical gain is auto-ranged like a real scope and **printed on the trace**
(`±66 m/div`), so the scale is never merely implied.

## The instrument stack

Eight line-art readouts, all fed by the same route engine:

| Module | Reads |
| --- | --- |
| **Alt Scope** | the 24-sample profile over a mini-graticule, camp-altitude reference, min/max cursors, `ΔH` bracket |
| **Bearing** | no dial face — a ring of 36 ticks, a one-line needle with an open V, and the ticks nearest it lit |
| **Cadence** | two beats of the clock's actual envelope, scrolling in tempo with zero per-frame JS |
| **Grade** | a T1–T5 ladder; the active tier is **hatched**, never filled, plus a continuous 0–1 pointer |
| **Bars** | a 32-step ruler with a measurement bracket spanning the mix length |
| **Wind** | tempo Δ as three staggered streaks with open chevrons |
| **Groove** | density drawn as density — cell *n* holds *n*+1 hairlines — plus the target's real kick row |
| **Channel Lock** | all three raw distances side by side; the active channel's row lifts to its hue, 0.7→1.0 is a hatched danger zone |

## One clock

`ClockProvider` runs a single `requestAnimationFrame` loop publishing `--beat`,
`--beat-phase` and `--pulse`. Every recurring animation is a CSS animation whose
duration is a multiple of `var(--beat)`, so the whole rig **retimes itself** when the
camp BPM changes and nothing needs a JS tick. Terrain canvases paint once per
(channel, size) and are cached — switching channels costs one `drawImage`.

## Run it

```bash
cd musicTrail
npm run dev      # → http://localhost:5273
```

Node 18+. Vite + React 18 (plain JSX). Canvas 2D + SVG + CSS only — no WebGL, no
chart libraries. Honors `prefers-reduced-motion`.

## Code map

```
src/
  clock/         ClockProvider (shared beat) + useSpring (needle/pointer glide)
  data/tracks.js the 20 warren tracks + narrative set order
  utils/         metaphor.js (route engine), terrain.js (marching squares), rng.js
  map/           ScopeMap (the well) + layers.js (the three channels: fields,
                 graphs, node values) + terrainRenderers (waterfall / contours /
                 lattice) + NodeGlyph + RouteTrace + ChannelRail + neon + palette
  instruments/   Rack + AltScope, BearingRose, CadenceTrace, GradeLadder,
                 BarsRuler, WindVector, GrooveComb, LockStrip
  styles/        theme + layout + map + instrument CSS
```

*(Earlier art directions — a warm hiking field-kit, a cyberpunk DJ terminal, and a
clean iOS-widget board — were each replaced in turn; the similarity/metaphor engine
survived every re-skin untouched.)*
