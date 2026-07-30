// Light "Apple Maps" terrain palette in JS form, so canvas drawing and CSS stay
// in lock-step. (Mirror of the CSS custom properties in styles/index.css.)
export const C = {
  land: '#eef0e4', // base map paper (light beige-green)
  landHi: '#f5f6ee',
  green: '#cfe6c0', // parks / high ground
  greenDeep: '#b6d6a2',
  water: '#a9d6ec', // lakes
  waterEdge: '#8cc4e0',
  contour: '#b7a98c', // subtle contour ink
  contourBold: '#9c8a68',
  road: '#ffffff',
  ink: '#1c1c1e',
  ink2: '#6b6b70',
  // iOS accents
  blue: '#0a84ff',
  green2: '#34c759',
  orange: '#ff9500',
  red: '#ff3b30',
  teal: '#30b0c7',
  indigo: '#5e5ce6',
}

// Linear blend of two hex colors → 'rgba(r,g,b,a)'.
export function mixRGBA(a, b, t, alpha = 1) {
  const pa = [parseInt(a.slice(1, 3), 16), parseInt(a.slice(3, 5), 16), parseInt(a.slice(5, 7), 16)]
  const pb = [parseInt(b.slice(1, 3), 16), parseInt(b.slice(3, 5), 16), parseInt(b.slice(5, 7), 16)]
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t))
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`
}

// The device-pixel ratio we render at, clamped. Exported so anything that caches
// a bitmap can key on it — a stale-DPR cache entry would blit at the wrong scale.
export function canvasDpr() {
  return Math.min(window.devicePixelRatio || 1, 2)
}

// Set up a crisp Hi-DPI canvas sized to w×h CSS px; returns the 2D context
// pre-scaled so all drawing uses CSS-pixel coordinates.
export function setupCanvas(canvas, w, h) {
  const dpr = canvasDpr()
  canvas.width = Math.max(1, Math.round(w * dpr))
  canvas.height = Math.max(1, Math.round(h * dpr))
  canvas.style.width = `${w}px`
  canvas.style.height = `${h}px`
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  return ctx
}
