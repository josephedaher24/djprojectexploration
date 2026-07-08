(function() {
  const records = JSON.parse(document.getElementById('simplex-records-json').textContent);
  const layouts = JSON.parse(document.getElementById('simplex-layouts-json').textContent);
  const layoutEntries = JSON.parse(document.getElementById('simplex-layout-entries-json').textContent);
  const simMap = JSON.parse(document.getElementById('simplex-sim-json').textContent);
  const config = JSON.parse(document.getElementById('simplex-config-json').textContent);
  const plot = document.getElementById('__PLOT_ID__');
  const els = {
    style: document.getElementById('weight-style'),
    te: document.getElementById('weight-tempo'),
    gr: document.getElementById('weight-groove'),
    ch: document.getElementById('weight-chroma'),
    styleVal: document.getElementById('weight-style-val'),
    teVal: document.getElementById('weight-tempo-val'),
    grVal: document.getElementById('weight-groove-val'),
    chVal: document.getElementById('weight-chroma-val'),
    simplex: document.getElementById('simplex-control'),
    simplexHandle: document.getElementById('simplex-handle'),
    summary: document.getElementById('weight-summary'),
    highlightLinkLimit: document.getElementById('highlight-link-limit'),
    highlightLinkLimitVal: document.getElementById('highlight-link-limit-val'),
    meta: document.getElementById('track-meta'),
    audio: document.getElementById('track-audio'),
    panel: document.getElementById('similarity-panel'),
  };
  let currentPoints = [];
  let selectedIdx = null;
  let backgroundTraceIndices = [];
  let selectedTraceIndices = [];
  let highlightedLinkLimit = Math.max(0, Number(config.click_links_per_song || 0));

  function esc(v) { return String(v == null ? '' : v).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
  function fmt(v,d) { return Number(v || 0).toFixed(d); }
  function pct(v) { return (Number(v || 0) * 100).toFixed(1) + '%'; }
  function signedPct(v) { const n = Number(v || 0) * 100; return (n >= 0 ? '+' : '') + n.toFixed(1) + '%'; }
  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
  const simplexVertices = {
    tempo: {x: 180, y: 34},
    groove: {x: 44, y: 270},
    chroma: {x: 316, y: 270},
  };
  function mixWeightsRaw() {
    const te = Math.max(0, Number(els.te.value || 0));
    const gr = Math.max(0, Number(els.gr.value || 0));
    const ch = Math.max(0, Number(els.ch.value || 0));
    const s = te + gr + ch;
    if (s <= 1e-12) return {tempo: 1/3, groove: 1/3, chroma: 1/3};
    return {tempo: te/s, groove: gr/s, chroma: ch/s};
  }
  function weights() {
    const style = clamp(Number(els.style.value || 0), 0, 1);
    const mix = mixWeightsRaw();
    const m = 1 - style;
    return {maest: style, tempo: m * mix.tempo, groove: m * mix.groove, chroma: m * mix.chroma, mix};
  }
  function setMixSliders(mix) {
    els.te.value = String(clamp(mix.tempo, 0, 1));
    els.gr.value = String(clamp(mix.groove, 0, 1));
    els.ch.value = String(clamp(mix.chroma, 0, 1));
  }
  function simplexPoint(mix) {
    return {
      x: mix.tempo * simplexVertices.tempo.x + mix.groove * simplexVertices.groove.x + mix.chroma * simplexVertices.chroma.x,
      y: mix.tempo * simplexVertices.tempo.y + mix.groove * simplexVertices.groove.y + mix.chroma * simplexVertices.chroma.y,
    };
  }
  function updateSimplexHandle(mix) {
    const p = simplexPoint(mix);
    els.simplexHandle.setAttribute('cx', String(p.x));
    els.simplexHandle.setAttribute('cy', String(p.y));
  }
  function simplexWeightsFromPoint(px, py) {
    const a = simplexVertices.tempo;
    const b = simplexVertices.groove;
    const c = simplexVertices.chroma;
    const denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    let te = ((b.y - c.y) * (px - c.x) + (c.x - b.x) * (py - c.y)) / denom;
    let gr = ((c.y - a.y) * (px - c.x) + (a.x - c.x) * (py - c.y)) / denom;
    let ch = 1 - te - gr;
    te = clamp(te, 0, 1); gr = clamp(gr, 0, 1); ch = clamp(ch, 0, 1);
    const s = Math.max(1e-12, te + gr + ch);
    return {tempo: te / s, groove: gr / s, chroma: ch / s};
  }
  function eventToSvgPoint(ev) {
    const rect = els.simplex.getBoundingClientRect();
    return {x: (ev.clientX - rect.left) * 360 / Math.max(1, rect.width), y: (ev.clientY - rect.top) * 320 / Math.max(1, rect.height)};
  }
  function setWeightsFromSimplexEvent(ev) {
    const p = eventToSvgPoint(ev);
    setMixSliders(simplexWeightsFromPoint(p.x, p.y));
    renderAll();
  }
  function layoutInterpolatedPoints(w) {
    const target = [w.maest, w.tempo, w.groove, w.chroma];
    const ranked = layoutEntries.map(entry => {
      const d2 = entry.weights.reduce((acc, val, idx) => acc + Math.pow(Number(val) - target[idx], 2), 0);
      return {entry, d2};
    }).sort((a,b) => a.d2 - b.d2).slice(0, 12);
    if (!ranked.length) return layouts['1.0,0.0,0.0,0.0'] || [];
    if ((config.layout_selection_mode || 'interpolated') === 'discrete' || ranked[0].d2 <= 1e-12) return ranked[0].entry.points;
    const weightsLocal = ranked.map(r => 1 / Math.max(1e-9, r.d2));
    const sum = weightsLocal.reduce((a,b) => a + b, 0);
    return ranked[0].entry.points.map((_, idx) => {
      let x = 0, y = 0;
      ranked.forEach((r, ridx) => {
        const c = weightsLocal[ridx] / sum;
        x += c * Number(r.entry.points[idx][0]);
        y += c * Number(r.entry.points[idx][1]);
      });
      return [x, y];
    });
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
      const score = w.maest*Number(c.maest_score_norm||0) + w.tempo*Number(c.tempo_score_norm||0) + w.groove*Number(c.groove_score_norm||0) + w.chroma*Number(c.chroma_score_norm||0);
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
    const base = [155, 155, 155], hot = [218, 65, 45], cold = [45, 98, 210];
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
        type: 'scatter', mode: 'lines',
        x: [Number(src[0]), Number(dst[0])], y: [Number(src[1]), Number(dst[1])],
        line: {color: colorForDelta(row.bpm_delta_frac, highlighted ? clamp(0.40 + prob * 2.2, 0.40, 0.90) : 0.13), width: highlighted ? clamp(1.8 + prob * 10.0, 1.8, 5.0) : 0.85, dash: highlighted ? 'solid' : 'dot'},
        hoverinfo: 'skip', showlegend: false,
      });
      if (highlighted) {
        traces.push({
          type: 'scatter',
          mode: 'markers+text',
          x: [Number(dst[0])],
          y: [Number(dst[1])],
          marker: {size: 18, color: 'rgba(255,255,255,0.94)', line: {color: colorForDelta(row.bpm_delta_frac, 0.85), width: 2}},
          text: [String(row.rank || traces.length + 1)],
          textposition: 'middle center',
          textfont: {size: 10, color: '#17202a'},
          hovertemplate: '#' + esc(row.rank || '') + ' ' + esc(row.title || '') + '<extra></extra>',
          showlegend: false,
        });
      }
    }
    return traces;
  }
  function renderBackgroundLinks(w) {
    backgroundTraceIndices = clearTraceSet(backgroundTraceIndices);
    const limit = Math.max(0, Number(config.background_links_per_song || 0));
    if (limit <= 0) return;
    const traces = [];
    for (const sourceIdx of Object.keys(simMap)) traces.push(...linkTraces(Number(sourceIdx), rankedRows(Number(sourceIdx), w), limit, false));
    backgroundTraceIndices = addTraceSet(traces);
  }
  function renderSelectedLinks(w) {
    selectedTraceIndices = clearTraceSet(selectedTraceIndices);
    if (selectedIdx === null) return;
    const limit = Math.max(0, Number(highlightedLinkLimit || 0));
    selectedTraceIndices = addTraceSet(linkTraces(selectedIdx, rankedRows(selectedIdx, w), limit, true));
  }
  function renderPanel(w) {
    if (selectedIdx === null) return;
    const rows = rankedRows(selectedIdx, w).slice(0, Number(config.top_k_rows || 25));
    const body = rows.map(row => '<tr>' +
      '<td class="num">' + row.rank + '</td><td class="num">' + esc(row.track_number) + '</td><td>' + esc(row.mix_slug) + '</td>' +
      '<td>' + esc(row.title) + '</td><td>' + esc(row.artists) + '</td><td>' + esc(row.genre) + '</td><td>' + esc(row.key) + '</td>' +
      '<td class="num">' + signedPct(row.bpm_delta_frac) + '</td><td class="num">' + fmt(row.score,4) + '</td>' +
      '<td class="num">' + fmt(row.maest_score_norm,4) + '</td><td class="num">' + fmt(row.tempo_score_norm,4) + '</td><td class="num">' + fmt(row.groove_score_norm,4) + '</td><td class="num">' + fmt(row.chroma_score_norm,4) + '</td>' +
      '<td class="num">' + fmt(row.maest_similarity,4) + '</td><td class="num">' + fmt(row.tempo_similarity,4) + '</td><td class="num">' + fmt(row.groove_similarity,4) + '</td><td class="num">' + fmt(row.chroma_similarity,4) + '</td>' +
      '<td class="num">' + pct(row.probability) + '</td></tr>').join('');
    els.panel.innerHTML = '<b>Top ' + Number(config.top_k_rows || 25) + ' genre/mixability matches</b>' +
      '<div class="muted">Score = style slider + mixability triangle split across tempo, groove, and key.</div>' +
      '<div class="table-wrap"><table><thead><tr><th class="num">#</th><th class="num">Track</th><th>Mix</th><th>Title</th><th>Artists</th><th>Genre</th><th>Key</th>' +
      '<th class="num">dBPM%</th><th class="num">Score</th><th class="num">Style</th><th class="num">Tempo</th><th class="num">Groove</th><th class="num">Key</th>' +
      '<th class="num">Style raw</th><th class="num">Tempo raw</th><th class="num">Groove raw</th><th class="num">Key raw</th><th class="num">Prob</th></tr></thead><tbody>' + body + '</tbody></table></div>';
  }
  function renderAll() {
    const w = weights();
    selectedTraceIndices = clearTraceSet(selectedTraceIndices);
    backgroundTraceIndices = clearTraceSet(backgroundTraceIndices);
    els.styleVal.textContent = fmt(w.maest, 3);
    els.teVal.textContent = fmt(w.mix.tempo, 3);
    els.grVal.textContent = fmt(w.mix.groove, 3);
    els.chVal.textContent = fmt(w.mix.chroma, 3);
    if (els.highlightLinkLimitVal) els.highlightLinkLimitVal.textContent = String(Math.max(0, Number(highlightedLinkLimit || 0)));
    updateSimplexHandle(w.mix);
    els.summary.innerHTML = 'Global weights: Style <b>' + pct(w.maest) + '</b>, tempo <b>' + pct(w.tempo) + '</b>, groove <b>' + pct(w.groove) + '</b>, key <b>' + pct(w.chroma) + '</b>.';
    currentPoints = layoutInterpolatedPoints(w);
    updatePointCoordinates();
    renderBackgroundLinks(w);
    renderSelectedLinks(w);
    renderPanel(w);
  }
  [els.style, els.te, els.gr, els.ch].forEach(el => el.addEventListener('input', () => {
    if (el !== els.style) setMixSliders(mixWeightsRaw());
    renderAll();
  }));
  if (els.highlightLinkLimit) {
    els.highlightLinkLimit.value = String(highlightedLinkLimit);
    if (els.highlightLinkLimitVal) els.highlightLinkLimitVal.textContent = String(highlightedLinkLimit);
    els.highlightLinkLimit.addEventListener('input', () => {
      highlightedLinkLimit = Math.max(0, Number(els.highlightLinkLimit.value || 0));
      if (els.highlightLinkLimitVal) els.highlightLinkLimitVal.textContent = String(highlightedLinkLimit);
      renderSelectedLinks(weights());
    });
  }
  if (els.simplex) {
    let draggingSimplex = false;
    els.simplex.addEventListener('pointerdown', ev => { draggingSimplex = true; els.simplex.setPointerCapture(ev.pointerId); setWeightsFromSimplexEvent(ev); });
    els.simplex.addEventListener('pointermove', ev => { if (draggingSimplex) setWeightsFromSimplexEvent(ev); });
    els.simplex.addEventListener('pointerup', ev => { draggingSimplex = false; try { els.simplex.releasePointerCapture(ev.pointerId); } catch (err) {} });
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
  setMixSliders(mixWeightsRaw());
  renderAll();
})();
