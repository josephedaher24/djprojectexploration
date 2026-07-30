// One shared projection so the paper/terrain canvases and the live SVG layer
// place everything in exactly the same spot. Normalized 0..1 map coords → pixels.
export function makeProject(w, h) {
  const padX = Math.min(64, w * 0.06)
  const padY = Math.min(54, h * 0.07)
  const project = (x, y) => [padX + x * (w - 2 * padX), padY + y * (h - 2 * padY)]
  project.padX = padX
  project.padY = padY
  project.w = w
  project.h = h
  return project
}
