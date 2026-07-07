(function() {
  const records = JSON.parse(document.getElementById('simplex-records-json').textContent);
  const layouts = JSON.parse(document.getElementById('simplex-layouts-json').textContent);
  const simMap = JSON.parse(document.getElementById('simplex-sim-json').textContent);
  const config = JSON.parse(document.getElementById('simplex-config-json').textContent);
  const plot = document.getElementById('__PLOT_ID__');
  const els = {
    ma: document.getElementById('weight-maest'),
    te: document.getElementById('weight-tempo'),
    ch: document.getElementById('weight-chroma'),
    maVal: document.getElementById('weight-maest-val'),
    teVal: document.getElementById('weight-tempo-val'),
    chVal: document.getElementById('weight-chroma-val'),
    simplex: document.getElementById('simplex-control'),
    simplexHandle: document.getElementById('simplex-handle'),
    summary: document.getElementById('weight-summary'),
    meta: document.getElementById('track-meta'),
    audio: document.getElementById('track-audio'),
    panel: document.getElementById('similarity-panel'),
  };
  let currentPoints = [];
  let selectedIdx = null;
  let backgroundTraceIndices = [];
  let selectedTraceIndices = [];

  function esc(v) { return String(v == null ? '' : v).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
  function fmt(v,d) { return Number(v || 0).toFixed(d); }
  function pct(v) { return (Number(v || 0) * 100).toFixed(1) + '%'; }
  function signedPct(v) { const n = Number(v || 0) * 100; return (n >= 0 ? '+' : '') + n.toFixed(1) + '%'; }
  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
  const simplexVertices = {
    maest: {x: 180, y: 34},
    tempo: {x: 44, y: 270},
    chroma: {x: 316, y: 270},
  };
  function weights() {
    const ma = Math.max(0, Number(els.ma.value || 0));
    const te = Math.max(0, Number(els.te.value || 0));
    const ch = Math.max(0, Number(els.ch.value || 0));
    const s = ma + te + ch;
    if (s <= 1e-12) return {maest: 1/3, tempo: 1/3, chroma: 1/3};
    return {maest: ma/s, tempo: te/s, chroma: ch/s};
  }
  function setSliderWeights(w) {
    els.ma.value = String(clamp(w.maest, 0, 1));
    els.te.value = String(clamp(w.tempo, 0, 1));
    els.ch.value = String(clamp(w.chroma, 0, 1));
  }
  function normalizeSliderWeights() { setSliderWeights(weights()); }
  function simplexPoint(w) {
    return {
      x: w.maest * simplexVertices.maest.x + w.tempo * simplexVertices.tempo.x + w.chroma * simplexVertices.chroma.x,
      y: w.maest * simplexVertices.maest.y + w.tempo * simplexVertices.tempo.y + w.chroma * simplexVertices.chroma.y,
    };
  }
  function updateSimplexHandle(w) {
    const p = simplexPoint(w);
    els.simplexHandle.setAttribute('cx', String(p.x));
    els.simplexHandle.setAttribute('cy', String(p.y));
  }
  function simplexWeightsFromPoint(px, py) {
    const a = simplexVertices.maest;
    const b = simplexVertices.tempo;
    const c = simplexVertices.chroma;
    const denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    let ma = ((b.y - c.y) * (px - c.x) + (c.x - b.x) * (py - c.y)) / denom;
    let te = ((c.y - a.y) * (px - c.x) + (a.x - c.x) * (py - c.y)) / denom;
    let ch = 1 - ma - te;
    ma = clamp(ma, 0, 1);
    te = clamp(te, 0, 1);
    ch = clamp(ch, 0, 1);
    const s = Math.max(1e-12, ma + te + ch);
    return {maest: ma / s, tempo: te / s, chroma: ch / s};
  }
  function eventToSvgPoint(ev) {
    const rect = els.simplex.getBoundingClientRect();
    return {
      x: (ev.clientX - rect.left) * 360 / Math.max(1, rect.width),
      y: (ev.clientY - rect.top) * 320 / Math.max(1, rect.height),
    };
  }
  function setWeightsFromSimplexEvent(ev) {
    const p = eventToSvgPoint(ev);
    const w = simplexWeightsFromPoint(p.x, p.y);
    setSliderWeights(w);
    renderAll();
  }
  function gridScale() { return Math.round(1 / Number(config.step || 0.1)); }
  function key(i,j,k) {
    const scale = gridScale();
    return (i/scale).toFixed(1)+','+(j/scale).toFixed(1)+','+(k/scale).toFixed(1);
  }
  function layoutAtGrid(i,j,k) {
    const scale = gridScale();
    if (i < 0 || j < 0 || k < 0 || i + j + k !== scale) return null;
    return layouts[key(i,j,k)];
  }
  function nearestGridCoords(w) {
    const scale = gridScale();
    let i = Math.round(clamp(w.maest, 0, 1) * scale);
    let j = Math.round(clamp(w.tempo, 0, 1) * scale);
    if (i + j > scale) {
      const excess = i + j - scale;
      if (j >= excess) j -= excess;
      else i = Math.max(0, i - (excess - j));
    }
    const k = scale - i - j;
    return {i, j, k, scale};
  }
  function nearestGridWeights(w) {
    const g = nearestGridCoords(w);
    return {maest: g.i / g.scale, tempo: g.j / g.scale, chroma: g.k / g.scale};
  }
  function nearestGridLayout(w) {
    const g = nearestGridCoords(w);
    const i = g.i, j = g.j, k = g.k;
    return layoutAtGrid(i, j, k) || layouts['1.0,0.0,0.0'] || [];
  }
  function simplexLayoutPoints(w) {
    if (config.layout_mode === 'discrete') return nearestGridLayout(w);
    return simplexInterpolatedPoints(w);
  }
  function simplexInterpolatedPoints(w) {
    const scale = gridScale();
    const eps = 1e-7;
    let x = clamp(w.maest, 0, 1) * scale;
    let y = clamp(w.tempo, 0, 1) * scale;
    if (x + y > scale) {
      const factor = scale / Math.max(eps, x + y);
      x *= factor;
      y *= factor;
    }
    x = Math.abs(x - Math.round(x)) < eps ? Math.round(x) : x;
    y = Math.abs(y - Math.round(y)) < eps ? Math.round(y) : y;
    let i = Math.floor(x);
    let j = Math.floor(y);
    if (i + j >= scale) {
      i = Math.min(scale, i);
      j = Math.min(scale - i, j);
      const A = layoutAtGrid(i, j, scale - i - j) || nearestGridLayout(w);
      return A || [];
    }
    const rx = x - i;
    const ry = y - j;
    const kBase = scale - i - j;
    let verts, coeffs;
    if (rx + ry <= 1 + eps || kBase <= 1) {
      verts = [[i,j,scale-i-j], [i+1,j,scale-i-j-1], [i,j+1,scale-i-j-1]];
      coeffs = [Math.max(0, 1-rx-ry), rx, ry];
    } else {
      verts = [[i+1,j+1,scale-i-j-2], [i+1,j,scale-i-j-1], [i,j+1,scale-i-j-1]];
      coeffs = [rx+ry-1, 1-ry, 1-rx];
    }
    const coeffSum = Math.max(eps, coeffs[0] + coeffs[1] + coeffs[2]);
    coeffs = coeffs.map(v => v / coeffSum);
    const L = verts.map(v => layoutAtGrid(v[0], v[1], v[2]));
    if (L.some(v => !v)) return nearestGridLayout(w);
    return L[0].map((_, idx) => [
      coeffs[0]*Number(L[0][idx][0]) + coeffs[1]*Number(L[1][idx][0]) + coeffs[2]*Number(L[2][idx][0]),
      coeffs[0]*Number(L[0][idx][1]) + coeffs[1]*Number(L[1][idx][1]) + coeffs[2]*Number(L[2][idx][1]),
    ]);
  }
  function baseTraceIndices() {
    const names = new Set(records.map(r => String(r.genre)));
    const out = [];
    (plot.data || []).forEach((trace, i) => { if (names.has(String(trace.name))) out.push(i); });
    return out;
  }
  function updatePointCoordinates() {
    const byGenre = new Map();
    for (const r of records) {
      const pt = currentPoints[Number(r.idx)];
      if (!pt) continue;
      const g = String(r.genre);
      if (!byGenre.has(g)) byGenre.set(g, {x: [], y: []});
      byGenre.get(g).x.push(Number(pt[0]));
      byGenre.get(g).y.push(Number(pt[1]));
    }
    for (const traceIdx of baseTraceIndices()) {
      const trace = plot.data[traceIdx] || {};
      const vals = byGenre.get(String(trace.name));
      if (vals) Plotly.restyle(plot, {x: [vals.x], y: [vals.y]}, [traceIdx]);
    }
  }
  function rankedRows(sourceIdx, w) {
    const candidates = ((simMap[String(sourceIdx)] || {}).candidates || []);
    const rows = candidates.map(c => {
      const score = w.maest*Number(c.maest_score_norm||0) + w.tempo*Number(c.tempo_score_norm||0) + w.chroma*Number(c.chroma_score_norm||0);
      return {...c, score};
    }).sort((a,b) => b.score - a.score);
    const temp = Math.max(1e-6, Number(config.temperature || 0.08));
    const maxScore = rows.length ? Number(rows[0].score || 0) : 0;
    let sum = 0;
    rows.forEach(r => { r._exp = Math.exp((Number(r.score || 0) - maxScore) / temp); sum += r._exp; });
    rows.forEach((r, i) => { r.rank = i + 1; r.probability = sum > 0 ? r._exp / sum : 0; });
    return rows;
  }
  function clearTraceSet(indices) {
    if (!indices.length || !window.Plotly) return [];
    try { Plotly.deleteTraces(plot, indices.slice().sort((a,b) => b-a)); } catch (err) {}
    return [];
  }
  function addTraceSet(traces) {
    if (!traces.length || !window.Plotly) return [];
    const start = (plot.data || []).length;
    try { Plotly.addTraces(plot, traces); } catch (err) { return []; }
    return Array.from({length: traces.length}, (_, i) => start + i);
  }
  function colorForDelta(delta, alpha) {
    const scale = Math.max(1e-6, Number(config.bpm_color_scale_pct || 0.10));
    const t = clamp(Number(delta || 0) / scale, -1, 1);
    const f = Math.abs(t);
    const base = [155, 155, 155];
    const hot = [218, 65, 45];
    const cold = [45, 98, 210];
    const target = t >= 0 ? hot : cold;
    const rgb = base.map((v, i) => Math.round(v + (target[i] - v) * f));
    return 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + alpha + ')';
  }
  function linkTraces(sourceIdx, rows, limit, highlighted) {
    const src = currentPoints[Number(sourceIdx)];
    if (!src) return [];
    const traces = [];
    for (const row of rows.slice(0, limit)) {
      const dst = currentPoints[Number(row.idx)];
      if (!dst) continue;
      const prob = Number(row.probability || 0);
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: [Number(src[0]), Number(dst[0])],
        y: [Number(src[1]), Number(dst[1])],
        line: {
          color: colorForDelta(row.bpm_delta_frac, highlighted ? clamp(0.40 + prob * 2.2, 0.40, 0.90) : 0.13),
          width: highlighted ? clamp(1.8 + prob * 10.0, 1.8, 5.0) : 0.85,
          dash: highlighted ? 'solid' : 'dot',
        },
        hoverinfo: 'skip',
        showlegend: false,
      });
    }
    return traces;
  }
  function renderBackgroundLinks(w) {
    backgroundTraceIndices = clearTraceSet(backgroundTraceIndices);
    const limit = Math.max(0, Number(config.background_links_per_song || 0));
    if (limit <= 0) return;
    const traces = [];
    for (const sourceIdx of Object.keys(simMap)) {
      traces.push(...linkTraces(Number(sourceIdx), rankedRows(Number(sourceIdx), w), limit, false));
    }
    backgroundTraceIndices = addTraceSet(traces);
  }
  function renderSelectedLinks(w) {
    selectedTraceIndices = clearTraceSet(selectedTraceIndices);
    if (selectedIdx === null) return;
    const limit = Math.max(0, Number(config.click_links_per_song || 0));
    selectedTraceIndices = addTraceSet(linkTraces(selectedIdx, rankedRows(selectedIdx, w), limit, true));
  }
  function renderPanel(w) {
    if (selectedIdx === null) return;
    const rows = rankedRows(selectedIdx, w).slice(0, Number(config.top_k_rows || 25));
    const body = rows.map(row => '<tr>' +
      '<td class="num">' + row.rank + '</td><td class="num">' + esc(row.track_number) + '</td><td>' + esc(row.mix_slug) + '</td>' +
      '<td>' + esc(row.title) + '</td><td>' + esc(row.artists) + '</td><td>' + esc(row.genre) + '</td><td>' + esc(row.key) + '</td>' +
      '<td class="num">' + signedPct(row.bpm_delta_frac) + '</td><td class="num">' + fmt(row.score,4) + '</td>' +
      '<td class="num">' + fmt(row.maest_score_norm,4) + '</td><td class="num">' + fmt(row.tempo_score_norm,4) + '</td><td class="num">' + fmt(row.chroma_score_norm,4) + '</td>' +
      '<td class="num">' + fmt(row.maest_similarity,4) + '</td><td class="num">' + fmt(row.tempo_similarity,4) + '</td><td class="num">' + fmt(row.chroma_similarity,4) + '</td>' +
      '<td class="num">' + pct(row.probability) + '</td></tr>').join('');
    els.panel.innerHTML = '<b>Top ' + Number(config.top_k_rows || 25) + ' Genre/tempo/key matches</b>' +
      '<div class="muted">Ranking uses normalized component scores derived from the same normalized distances as the layout. Raw similarities are shown for diagnostics.</div>' +
      '<div class="table-wrap"><table><thead><tr><th class="num">#</th><th class="num">Track</th><th>Mix</th><th>Title</th><th>Artists</th><th>Genre</th><th>Key</th>' +
      '<th class="num">dBPM%</th><th class="num">Score</th><th class="num">Genre norm</th><th class="num">Tempo norm</th><th class="num">Key norm</th>' +
      '<th class="num">Genre raw</th><th class="num">Tempo raw</th><th class="num">Key raw</th><th class="num">Prob</th></tr></thead><tbody>' + body + '</tbody></table></div>';
  }
  function renderAll() {
    let w = weights();
    if (config.layout_mode === 'discrete') {
      w = nearestGridWeights(w);
      setSliderWeights(w);
    }
    selectedTraceIndices = clearTraceSet(selectedTraceIndices);
    backgroundTraceIndices = clearTraceSet(backgroundTraceIndices);
    els.maVal.textContent = fmt(w.maest, 3);
    els.teVal.textContent = fmt(w.tempo, 3);
    els.chVal.textContent = fmt(w.chroma, 3);
    updateSimplexHandle(w);
    els.summary.innerHTML = 'Normalized weights: Genre <b>' + pct(w.maest) + '</b>, tempo <b>' + pct(w.tempo) + '</b>, key <b>' + pct(w.chroma) + '</b>.' +
      (config.layout_mode === 'discrete' ? ' Showing nearest precomputed layout.' : '');
    currentPoints = simplexLayoutPoints(w);
    updatePointCoordinates();
    renderBackgroundLinks(w);
    renderSelectedLinks(w);
    renderPanel(w);
  }
  [els.ma, els.te, els.ch].forEach(el => el.addEventListener('input', () => {
    normalizeSliderWeights();
    renderAll();
  }));
  if (els.simplex) {
    let draggingSimplex = false;
    els.simplex.addEventListener('pointerdown', ev => {
      draggingSimplex = true;
      els.simplex.setPointerCapture(ev.pointerId);
      setWeightsFromSimplexEvent(ev);
    });
    els.simplex.addEventListener('pointermove', ev => {
      if (!draggingSimplex) return;
      setWeightsFromSimplexEvent(ev);
    });
    els.simplex.addEventListener('pointerup', ev => {
      draggingSimplex = false;
      try { els.simplex.releasePointerCapture(ev.pointerId); } catch (err) {}
    });
    els.simplex.addEventListener('pointercancel', () => { draggingSimplex = false; });
  }
  if (plot && plot.on) {
    plot.on('plotly_click', ev => {
      if (!ev || !ev.points || !ev.points.length) return;
      const c = ev.points[0].customdata || [];
      const idx = Number(c[13]);
      if (!Number.isFinite(idx)) return;
      selectedIdx = idx;
      els.meta.innerHTML = '<b>' + esc(c[0]) + '</b><br>Artists: ' + esc(c[1]) + '<br>Genre: ' + esc(c[2]) +
        '<br>Mix: ' + esc(c[14]) + '<br>Key: ' + esc(c[3]) + '<br>CSV BPM: ' + esc(c[4]) +
        '<br>Estimated BPM: ' + fmt(c[5], 2) + ' (confidence=' + fmt(c[6], 3) + ')<br>Track #: ' + esc(c[7]) + '<br>File: ' + esc(c[8]);
      if (c[9]) {
        els.audio.src = c[9];
        const p = els.audio.play();
        if (p && p.catch) p.catch(() => {});
      } else {
        els.audio.removeAttribute('src');
        els.audio.load();
      }
      renderSelectedLinks(weights());
      renderPanel(weights());
    });
  }
  renderAll();
})();
