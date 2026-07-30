// Line-art neon drawing helpers. The rule everywhere: glow lives in a wide,
// low-alpha pass BEHIND a crisp hairline core — never blur the line itself, and
// never fill a shape where a stroke will do.

// Stroke a path-building callback twice: a soft bloom pass, then a crisp core.
export function neonStroke(ctx, draw, { color, width = 1, glow = 8, bloomAlpha = 0.16, coreAlpha = 1 }) {
  // bloom
  ctx.save()
  ctx.globalAlpha = bloomAlpha
  ctx.strokeStyle = color
  ctx.lineWidth = width * 3.5
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.shadowColor = color
  ctx.shadowBlur = glow
  ctx.beginPath()
  draw(ctx)
  ctx.stroke()
  ctx.restore()

  // crisp core
  ctx.save()
  ctx.globalAlpha = coreAlpha
  ctx.strokeStyle = color
  ctx.lineWidth = width
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.beginPath()
  draw(ctx)
  ctx.stroke()
  ctx.restore()
}

// Read a CSS custom property off :root (so canvas colour follows the stylesheet).
export function cssVar(name, fallback = '#00e5ff') {
  if (typeof window === 'undefined') return fallback
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}

// 'rgba()' string from a #rrggbb hex plus alpha.
export function alpha(hex, a) {
  const h = hex.replace('#', '')
  const n = h.length === 3 ? h.split('').map((c) => c + c).join('') : h
  const r = parseInt(n.slice(0, 2), 16)
  const g = parseInt(n.slice(2, 4), 16)
  const b = parseInt(n.slice(4, 6), 16)
  return `rgba(${r},${g},${b},${a})`
}
