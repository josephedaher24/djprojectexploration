(function() {
  const records = JSON.parse(document.getElementById('simplex-records-json').textContent);
  const layouts = JSON.parse(document.getElementById('simplex-layouts-json').textContent);
  const layoutEntries = JSON.parse(document.getElementById('simplex-layout-entries-json').textContent);
  const simMap = JSON.parse(document.getElementById('simplex-sim-json').textContent);
  const config = JSON.parse(document.getElementById('simplex-config-json').textContent);
  const plot = document.getElementById('__PLOT_ID__');
  const els = {
    style: document.getElementById('weight-style'),
    rhythm: document.getElementById('weight-rhythm'),
    harmony: document.getElementById('weight-harmony'),
    styleVal: document.getElementById('weight-style-val'),
    rhythmVal: document.getElementById('weight-rhythm-val'),
    harmonyVal: document.getElementById('weight-harmony-val'),
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
  const configuredRhythmTempoWeight = Number(config.rhythm_tempo_weight);
  const rhythmTempoWeight = Number.isFinite(configuredRhythmTempoWeight)
    ? clamp(configuredRhythmTempoWeight, 0, 1)
    : 0.70;
  const simplexVertices = {
    style: {x: 180, y: 34},
    rhythm: {x: 44, y: 270},
    harmony: {x: 316, y: 270},
  };
  function weights() {
    const style = Math.max(0, Number(els.style.value || 0));
    const rhythm = Math.max(0, Number(els.rhythm.value || 0));
    const harmony = Math.max(0, Number(els.harmony.value || 0));
    const total = style + rhythm + harmony;
    if (total <= 1e-12) return {style: 0.5, rhythm: 0.3, harmony: 0.2};
    return {style: style / total, rhythm: rhythm / total, harmony: harmony / total};
  }
  function setWeightSliders(w) {
    els.style.value = String(clamp(w.style, 0, 1));
    els.rhythm.value = String(clamp(w.rhythm, 0, 1));
    els.harmony.value = String(clamp(w.harmony, 0, 1));
  }
  function simplexPoint(w) {
    return {
      x: w.style * simplexVertices.style.x + w.rhythm * simplexVertices.rhythm.x + w.harmony * simplexVertices.harmony.x,
      y: w.style * simplexVertices.style.y + w.rhythm * simplexVertices.rhythm.y + w.harmony * simplexVertices.harmony.y,
    };
  }
  function updateSimplexHandle(w) {
    const p = simplexPoint(w);
    els.simplexHandle.setAttribute('cx', String(p.x));
    els.simplexHandle.setAttribute('cy', String(p.y));
  }
  function simplexWeightsFromPoint(px, py) {
    const a = simplexVertices.style;
    const b = simplexVertices.rhythm;
    const c = simplexVertices.harmony;
    const denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    let style = ((b.y - c.y) * (px - c.x) + (c.x - b.x) * (py - c.y)) / denom;
    let rhythm = ((c.y - a.y) * (px - c.x) + (a.x - c.x) * (py - c.y)) / denom;
    let harmony = 1 - style - rhythm;
    style = clamp(style, 0, 1); rhythm = clamp(rhythm, 0, 1); harmony = clamp(harmony, 0, 1);
    const total = Math.max(1e-12, style + rhythm + harmony);
    return {style: style / total, rhythm: rhythm / total, harmony: harmony / total};
  }
  function eventToSvgPoint(ev) {
    const rect = els.simplex.getBoundingClientRect();
    return {x: (ev.clientX - rect.left) * 360 / Math.max(1, rect.width), y: (ev.clientY - rect.top) * 320 / Math.max(1, rect.height)};
  }
  function setWeightsFromSimplexEvent(ev) {
    const p = eventToSvgPoint(ev);
    setWeightSliders(simplexWeightsFromPoint(p.x, p.y));
    renderAll();
  }
  function layoutInterpolatedPoints(w) {
    const target = [w.style, w.rhythm, w.harmony];
    const ranked = layoutEntries.map(entry => {
      const d2 = entry.weights.reduce((acc, val, idx) => acc + Math.pow(Number(val) - target[idx], 2), 0);
      return {entry, d2};
    }).sort((a,b) => a.d2 - b.d2).slice(0, 12);
    if (!ranked.length) return layouts['1.0,0.0,0.0'] || [];
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
      const tempoScore = Number(c.tempo_score_norm || 0);
      const grooveScore = Number(c.groove_score_norm || 0);
      const rhythmScore = Number(c.rhythm_score_norm ?? (rhythmTempoWeight * tempoScore + (1 - rhythmTempoWeight) * grooveScore));
      const styleScore = Number(c.style_score_norm ?? c.maest_score_norm ?? 0);
      const harmonyScore = Number(c.harmony_score_norm ?? c.chroma_score_norm ?? 0);
      const score = w.style * styleScore + w.rhythm * rhythmScore + w.harmony * harmonyScore;
      return {...c, styleScore, rhythmScore, harmonyScore, score};
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
      '<td class="num">' + fmt(row.styleScore,4) + '</td><td class="num">' + fmt(row.rhythmScore,4) + '</td><td class="num">' + fmt(row.harmonyScore,4) + '</td>' +
      '<td class="num">' + fmt(row.maest_similarity,4) + '</td><td class="num">' + fmt(row.tempo_similarity,4) + '</td><td class="num">' + fmt(row.groove_similarity,4) + '</td><td class="num">' + fmt(row.chroma_similarity,4) + '</td>' +
      '<td class="num">' + pct(row.probability) + '</td></tr>').join('');
    els.panel.innerHTML = '<b>Top ' + Number(config.top_k_rows || 25) + ' transition matches</b>' +
      '<div class="muted">Score = Style + Rhythm + Harmony; Rhythm = ' + fmt(rhythmTempoWeight, 2) + ' tempo + ' + fmt(1 - rhythmTempoWeight, 2) + ' groove.</div>' +
      '<div class="table-wrap"><table><thead><tr><th class="num">#</th><th class="num">Track</th><th>Mix</th><th>Title</th><th>Artists</th><th>Genre</th><th>Key</th>' +
      '<th class="num">dBPM%</th><th class="num">Score</th><th class="num">Style</th><th class="num">Rhythm</th><th class="num">Harmony</th>' +
      '<th class="num">Style raw</th><th class="num">Tempo detail</th><th class="num">Groove detail</th><th class="num">Harmony raw</th><th class="num">Prob</th></tr></thead><tbody>' + body + '</tbody></table></div>';
  }
  function renderAll() {
    const w = weights();
    selectedTraceIndices = clearTraceSet(selectedTraceIndices);
    backgroundTraceIndices = clearTraceSet(backgroundTraceIndices);
    els.styleVal.textContent = fmt(w.style, 3);
    els.rhythmVal.textContent = fmt(w.rhythm, 3);
    els.harmonyVal.textContent = fmt(w.harmony, 3);
    if (els.highlightLinkLimitVal) els.highlightLinkLimitVal.textContent = String(Math.max(0, Number(highlightedLinkLimit || 0)));
    updateSimplexHandle(w);
    els.summary.innerHTML = 'Weights: <b>' + fmt(w.style, 2) + 'S/' + fmt(w.rhythm, 2) + 'R/' + fmt(w.harmony, 2) + 'H</b>; rhythm=' + fmt(rhythmTempoWeight, 2) + 'T+' + fmt(1 - rhythmTempoWeight, 2) + 'G.';
    currentPoints = layoutInterpolatedPoints(w);
    updatePointCoordinates();
    renderBackgroundLinks(w);
    renderSelectedLinks(w);
    renderPanel(w);
  }
  [els.style, els.rhythm, els.harmony].forEach(el => el.addEventListener('input', () => {
    setWeightSliders(weights());
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
  setWeightSliders({
    style: Number(config.default_weights && config.default_weights.style_weight) || 0.50,
    rhythm: Number(config.default_weights && config.default_weights.rhythm_weight) || 0.30,
    harmony: Number(config.default_weights && config.default_weights.harmony_weight) || 0.20,
  });
  renderAll();
})();
