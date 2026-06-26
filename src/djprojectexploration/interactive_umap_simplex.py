"""Export an interactive 3-way UMAP visualization with simplex interpolation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from djprojectexploration.interactive_umap_precomputed import (
    PROJECT_ROOT,
    _align_to_reference,
    _build_plot,
    _json_script_payload,
    _load_combined_records_and_features,
    _normalize_distance_matrix,
)
from djprojectexploration.multimodal_compatibility import (
    _build_harmonic_kernel,
    _pairwise_cosine_similarity_matrix,
    _pairwise_fifth_aware_similarity_matrix,
    _pairwise_tempo_similarity_matrix,
)


def _simplex_grid(step: float) -> list[tuple[float, float, float]]:
    scale = int(round(1.0 / float(step)))
    if scale <= 0:
        raise ValueError("step must be > 0.")
    return [
        (i / scale, j / scale, (scale - i - j) / scale)
        for i in range(scale + 1)
        for j in range(scale + 1 - i)
    ]


def _simplex_key(weights: tuple[float, float, float]) -> str:
    return ",".join(f"{w:.1f}" for w in weights)


def _component_matrices_3way(
    features,
    *,
    tempo_bandwidth: float,
    tempo_decay: float,
    tempo_allow_octave: bool,
    tempo_octave_penalty: float,
    tempo_similarity_shape: str,
    tempo_softflat_sharpness: float,
    tempo_use_confidence: bool,
    harmonic_exact_weight: float,
    harmonic_first_fifth_weight: float,
    harmonic_second_fifth_weight: float,
    harmonic_other_weight: float,
    harmonic_self_normalize: bool,
) -> dict[str, np.ndarray]:
    maest_cos = _pairwise_cosine_similarity_matrix(features.maest)
    maest_similarity = np.clip(0.5 * (maest_cos + 1.0), 0.0, 1.0).astype(np.float32)

    tempo_similarity = _pairwise_tempo_similarity_matrix(
        features.tempo_bpm,
        features.tempo_confidence,
        bandwidth=tempo_bandwidth,
        decay=tempo_decay,
        allow_octave=tempo_allow_octave,
        octave_penalty=tempo_octave_penalty,
        tempo_similarity_shape=tempo_similarity_shape,
        softflat_sharpness=tempo_softflat_sharpness,
        use_confidence=tempo_use_confidence,
    )
    tempo_similarity = np.clip(tempo_similarity, 0.0, 1.0).astype(np.float32)

    harmonic_kernel = _build_harmonic_kernel(
        exact_weight=harmonic_exact_weight,
        first_fifth_weight=harmonic_first_fifth_weight,
        second_fifth_weight=harmonic_second_fifth_weight,
        other_weight=harmonic_other_weight,
    )
    chroma_similarity = _pairwise_fifth_aware_similarity_matrix(
        features.chroma_pitch,
        kernel=harmonic_kernel,
        normalize_by_self=harmonic_self_normalize,
    )
    chroma_similarity = np.clip(chroma_similarity, 0.0, 1.0).astype(np.float32)

    return {
        "maest_similarity": maest_similarity,
        "tempo_similarity": tempo_similarity,
        "chroma_similarity": chroma_similarity,
        "maest_distance": _normalize_distance_matrix(1.0 - maest_similarity),
        "tempo_distance": _normalize_distance_matrix(1.0 - tempo_similarity),
        "chroma_distance": _normalize_distance_matrix(1.0 - chroma_similarity),
    }


def _combined_distance_3way(
    D_maest: np.ndarray,
    D_tempo: np.ndarray,
    D_chroma: np.ndarray,
    weights: tuple[float, float, float],
) -> np.ndarray:
    wm, wt, wc = [float(np.clip(v, 0.0, 1.0)) for v in weights]
    total = max(wm + wt + wc, 1e-12)
    wm, wt, wc = wm / total, wt / total, wc / total
    D = np.sqrt((wm * D_maest**2) + (wt * D_tempo**2) + (wc * D_chroma**2)).astype(np.float32)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def _compute_umap_simplex_layouts(
    *,
    D_maest: np.ndarray,
    D_tempo: np.ndarray,
    D_chroma: np.ndarray,
    grid: list[tuple[float, float, float]],
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    align: bool,
) -> dict[str, list[list[float]]]:
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError("UMAP is not installed. Install with: uv add umap-learn") from exc

    layouts: dict[str, list[list[float]]] = {}
    reference: np.ndarray | None = None
    n = D_maest.shape[0]
    effective_neighbors = min(max(2, int(n_neighbors)), max(2, n - 1))

    # Compute high-MAEST layouts first so every other layout can align to a stable reference.
    ordered_grid = sorted(grid, key=lambda w: (-w[0], -w[1], -w[2]))
    for weights in ordered_grid:
        D = _combined_distance_3way(D_maest, D_tempo, D_chroma, weights)
        reducer = UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=float(min_dist),
            metric="precomputed",
            random_state=int(random_state),
        )
        coords = np.asarray(reducer.fit_transform(D), dtype=np.float32)
        if align and reference is not None:
            coords = _align_to_reference(reference, coords)
        if reference is None:
            reference = coords
        layouts[_simplex_key(weights)] = [[float(x), float(y)] for x, y in coords]
    return layouts


def _build_similarity_payload_3way(
    *,
    records: list[dict[str, Any]],
    maest_similarity: np.ndarray,
    tempo_similarity: np.ndarray,
    chroma_similarity: np.ndarray,
    maest_distance: np.ndarray,
    tempo_distance: np.ndarray,
    chroma_distance: np.ndarray,
    temperature: float,
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    n = len(records)
    for src_idx in range(n):
        rows: list[dict[str, Any]] = []
        for cand_idx in range(n):
            if cand_idx == src_idx:
                continue
            cand = records[cand_idx]
            src_bpm = float(records[src_idx]["est_bpm"])
            cand_bpm = float(cand["est_bpm"])
            bpm_delta_frac = 0.0
            if src_bpm > 0.0 and cand_bpm > 0.0:
                bpm_delta_frac = float((cand_bpm - src_bpm) / src_bpm)
            rows.append(
                {
                    "idx": int(cand_idx),
                    "track_number": str(cand["track_number"]),
                    "mix_slug": str(cand["mix_slug"]),
                    "title": str(cand["title"]),
                    "artists": str(cand["artists"]),
                    "genre": str(cand["genre"]),
                    "key": str(cand["key"]),
                    "csv_bpm": str(cand["csv_bpm"]),
                    "est_bpm": cand_bpm,
                    "est_conf": float(cand["est_conf"]),
                    "maest_similarity": float(maest_similarity[src_idx, cand_idx]),
                    "tempo_similarity": float(tempo_similarity[src_idx, cand_idx]),
                    "chroma_similarity": float(chroma_similarity[src_idx, cand_idx]),
                    "maest_score_norm": float(1.0 - maest_distance[src_idx, cand_idx]),
                    "tempo_score_norm": float(1.0 - tempo_distance[src_idx, cand_idx]),
                    "chroma_score_norm": float(1.0 - chroma_distance[src_idx, cand_idx]),
                    "bpm_delta_frac": bpm_delta_frac,
                }
            )
        payload[str(src_idx)] = {"candidates": rows, "temperature": float(temperature)}
    return payload


def _build_simplex_html(
    *,
    plot_html: str,
    records: list[dict[str, Any]],
    layouts: dict[str, list[list[float]]],
    similarity_payload: dict[str, dict[str, Any]],
    plot_div_id: str,
    title: str,
    step: float,
    top_k_rows: int,
    temperature: float,
    background_links_per_song: int,
    click_links_per_song: int,
    bpm_color_scale_pct: float,
) -> str:
    record_payload = [
        {
            "idx": int(r["idx"]),
            "title": str(r["title"]),
            "artists": str(r["artists"]),
            "genre": str(r["genre"]),
            "key": str(r["key"]),
            "csv_bpm": str(r["csv_bpm"]),
            "est_bpm": float(r["est_bpm"]),
            "est_conf": float(r["est_conf"]),
            "track_number": str(r["track_number"]),
            "filename": str(r["filename"]),
            "snippet_uri": str(r["snippet_uri"]),
            "snippet_start": float(r["snippet_start"]),
            "snippet_end": float(r["snippet_end"]),
            "snippet_rms": float(r["snippet_rms"]),
            "mix_slug": str(r["mix_slug"]),
        }
        for r in records
    ]
    data_scripts = "\n".join(
        [
            f'<script id="simplex-records-json" type="application/json">{_json_script_payload(record_payload)}</script>',
            f'<script id="simplex-layouts-json" type="application/json">{_json_script_payload(layouts)}</script>',
            f'<script id="simplex-sim-json" type="application/json">{_json_script_payload(similarity_payload)}</script>',
            f'<script id="simplex-config-json" type="application/json">{_json_script_payload({"step": step, "top_k_rows": top_k_rows, "temperature": temperature, "background_links_per_song": background_links_per_song, "click_links_per_song": click_links_per_song, "bpm_color_scale_pct": bpm_color_scale_pct})}</script>',
        ]
    )
    style = """
<style>
  :root { color-scheme: light; --line:#d9dee8; --ink:#17202a; --muted:#5b6678; }
  body { font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 18px; color: var(--ink); background: #f6f8fb; }
  h1 { font-size: 22px; margin: 0 0 14px; }
  .panel { display:grid; gap:10px; margin-top:12px; }
  .box, .weights { border:1px solid var(--line); background:#fff; padding:10px; }
  .control-grid { display:grid; grid-template-columns:minmax(260px, 380px) 1fr; gap:12px; align-items:center; }
  .simplex-control { width:100%; max-width:380px; display:block; touch-action:none; user-select:none; }
  .simplex-area { fill:#fbfcff; stroke:#9aa8bc; stroke-width:1.5; }
  .simplex-grid-line { stroke:#d8deea; stroke-width:0.8; }
  .simplex-label { fill:#2f3a4c; font-size:13px; font-weight:600; text-anchor:middle; }
  .simplex-handle { fill:#111827; stroke:#fff; stroke-width:4; cursor:grab; }
  .simplex-handle:active { cursor:grabbing; }
  .weights { display:grid; grid-template-columns: 72px 1fr 56px; gap:8px; align-items:center; }
  .muted { color:var(--muted); font-size:12px; }
  audio { width:100%; display:block; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th, td { border-bottom:1px solid #edf0f5; padding:5px 6px; vertical-align:top; }
  th { position:sticky; top:0; background:#f9fafc; z-index:1; text-align:left; color:#3c4758; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .table-wrap { max-height:380px; overflow:auto; border:1px solid #edf0f5; }
</style>
"""
    controls = """
<section class="panel">
  <div class="box control-grid">
    <svg id="simplex-control" class="simplex-control" viewBox="0 0 360 320" role="img" aria-label="MAEST tempo chroma blend triangle">
      <polygon id="simplex-area" class="simplex-area" points="180,34 44,270 316,270"></polygon>
      <line class="simplex-grid-line" x1="166.4" y1="57.6" x2="71.2" y2="270"></line>
      <line class="simplex-grid-line" x1="193.6" y1="57.6" x2="288.8" y2="270"></line>
      <line class="simplex-grid-line" x1="153.0" y1="80.8" x2="98.4" y2="270"></line>
      <line class="simplex-grid-line" x1="207.0" y1="80.8" x2="261.6" y2="270"></line>
      <line class="simplex-grid-line" x1="139.2" y1="104.4" x2="125.6" y2="270"></line>
      <line class="simplex-grid-line" x1="220.8" y1="104.4" x2="234.4" y2="270"></line>
      <line class="simplex-grid-line" x1="125.6" y1="128.0" x2="152.8" y2="270"></line>
      <line class="simplex-grid-line" x1="234.4" y1="128.0" x2="207.2" y2="270"></line>
      <line class="simplex-grid-line" x1="112.0" y1="151.6" x2="180.0" y2="270"></line>
      <line class="simplex-grid-line" x1="248.0" y1="151.6" x2="180.0" y2="270"></line>
      <line class="simplex-grid-line" x1="112.0" y1="151.6" x2="248.0" y2="151.6"></line>
      <line class="simplex-grid-line" x1="98.4" y1="175.2" x2="261.6" y2="175.2"></line>
      <line class="simplex-grid-line" x1="84.8" y1="198.8" x2="275.2" y2="198.8"></line>
      <line class="simplex-grid-line" x1="71.2" y1="222.4" x2="288.8" y2="222.4"></line>
      <line class="simplex-grid-line" x1="57.6" y1="246.0" x2="302.4" y2="246.0"></line>
      <text class="simplex-label" x="180" y="22">MAEST</text>
      <text class="simplex-label" x="44" y="294">Tempo</text>
      <text class="simplex-label" x="316" y="294">Chroma</text>
      <circle id="simplex-handle" class="simplex-handle" cx="180" cy="128.4" r="10"></circle>
    </svg>
    <div class="weights">
      <div>MAEST</div><input id="weight-maest" type="range" min="0" max="1" step="0.01" value="0.6"><output id="weight-maest-val">0.600</output>
      <div>Tempo</div><input id="weight-tempo" type="range" min="0" max="1" step="0.01" value="0.2"><output id="weight-tempo-val">0.200</output>
      <div>Chroma</div><input id="weight-chroma" type="range" min="0" max="1" step="0.01" value="0.2"><output id="weight-chroma-val">0.200</output>
      <div></div><div id="weight-summary" class="muted"></div><div></div>
    </div>
  </div>
  <div id="track-meta" class="box">Click a point to play its snippet and show MAEST/tempo/chroma recommendations.</div>
  <audio id="track-audio" controls></audio>
  <div id="similarity-panel" class="box">Recommendations will appear here after you click a point.</div>
</section>
"""
    script = f"""
<script>
(function() {{
  const records = JSON.parse(document.getElementById('simplex-records-json').textContent);
  const layouts = JSON.parse(document.getElementById('simplex-layouts-json').textContent);
  const simMap = JSON.parse(document.getElementById('simplex-sim-json').textContent);
  const config = JSON.parse(document.getElementById('simplex-config-json').textContent);
  const plot = document.getElementById('{plot_div_id}');
  const els = {{
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
  }};
  let currentPoints = [];
  let selectedIdx = null;
  let backgroundTraceIndices = [];
  let selectedTraceIndices = [];

  function esc(v) {{ return String(v == null ? '' : v).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch])); }}
  function fmt(v,d) {{ return Number(v || 0).toFixed(d); }}
  function pct(v) {{ return (Number(v || 0) * 100).toFixed(1) + '%'; }}
  function signedPct(v) {{ const n = Number(v || 0) * 100; return (n >= 0 ? '+' : '') + n.toFixed(1) + '%'; }}
  function clamp(v, lo, hi) {{ return Math.min(hi, Math.max(lo, v)); }}
  const simplexVertices = {{
    maest: {{x: 180, y: 34}},
    tempo: {{x: 44, y: 270}},
    chroma: {{x: 316, y: 270}},
  }};
  function weights() {{
    const ma = Math.max(0, Number(els.ma.value || 0));
    const te = Math.max(0, Number(els.te.value || 0));
    const ch = Math.max(0, Number(els.ch.value || 0));
    const s = ma + te + ch;
    if (s <= 1e-12) return {{maest: 1/3, tempo: 1/3, chroma: 1/3}};
    return {{maest: ma/s, tempo: te/s, chroma: ch/s}};
  }}
  function setSliderWeights(w) {{
    els.ma.value = String(clamp(w.maest, 0, 1));
    els.te.value = String(clamp(w.tempo, 0, 1));
    els.ch.value = String(clamp(w.chroma, 0, 1));
  }}
  function normalizeSliderWeights() {{ setSliderWeights(weights()); }}
  function simplexPoint(w) {{
    return {{
      x: w.maest * simplexVertices.maest.x + w.tempo * simplexVertices.tempo.x + w.chroma * simplexVertices.chroma.x,
      y: w.maest * simplexVertices.maest.y + w.tempo * simplexVertices.tempo.y + w.chroma * simplexVertices.chroma.y,
    }};
  }}
  function updateSimplexHandle(w) {{
    const p = simplexPoint(w);
    els.simplexHandle.setAttribute('cx', String(p.x));
    els.simplexHandle.setAttribute('cy', String(p.y));
  }}
  function simplexWeightsFromPoint(px, py) {{
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
    return {{maest: ma / s, tempo: te / s, chroma: ch / s}};
  }}
  function eventToSvgPoint(ev) {{
    const rect = els.simplex.getBoundingClientRect();
    return {{
      x: (ev.clientX - rect.left) * 360 / Math.max(1, rect.width),
      y: (ev.clientY - rect.top) * 320 / Math.max(1, rect.height),
    }};
  }}
  function setWeightsFromSimplexEvent(ev) {{
    const p = eventToSvgPoint(ev);
    const w = simplexWeightsFromPoint(p.x, p.y);
    setSliderWeights(w);
    renderAll();
  }}
  function gridScale() {{ return Math.round(1 / Number(config.step || 0.1)); }}
  function key(i,j,k) {{
    const scale = gridScale();
    return (i/scale).toFixed(1)+','+(j/scale).toFixed(1)+','+(k/scale).toFixed(1);
  }}
  function layoutAtGrid(i,j,k) {{
    const scale = gridScale();
    if (i < 0 || j < 0 || k < 0 || i + j + k !== scale) return null;
    return layouts[key(i,j,k)];
  }}
  function nearestGridLayout(w) {{
    const scale = gridScale();
    let i = Math.round(clamp(w.maest, 0, 1) * scale);
    let j = Math.round(clamp(w.tempo, 0, 1) * scale);
    if (i + j > scale) {{
      const excess = i + j - scale;
      if (j >= excess) j -= excess;
      else i = Math.max(0, i - (excess - j));
    }}
    const k = scale - i - j;
    return layoutAtGrid(i, j, k) || layouts['1.0,0.0,0.0'] || [];
  }}
  function simplexInterpolatedPoints(w) {{
    const scale = gridScale();
    const eps = 1e-7;
    let x = clamp(w.maest, 0, 1) * scale;
    let y = clamp(w.tempo, 0, 1) * scale;
    if (x + y > scale) {{
      const factor = scale / Math.max(eps, x + y);
      x *= factor;
      y *= factor;
    }}
    x = Math.abs(x - Math.round(x)) < eps ? Math.round(x) : x;
    y = Math.abs(y - Math.round(y)) < eps ? Math.round(y) : y;
    let i = Math.floor(x);
    let j = Math.floor(y);
    if (i + j >= scale) {{
      i = Math.min(scale, i);
      j = Math.min(scale - i, j);
      const A = layoutAtGrid(i, j, scale - i - j) || nearestGridLayout(w);
      return A || [];
    }}
    const rx = x - i;
    const ry = y - j;
    const kBase = scale - i - j;
    let verts, coeffs;
    if (rx + ry <= 1 + eps || kBase <= 1) {{
      verts = [[i,j,scale-i-j], [i+1,j,scale-i-j-1], [i,j+1,scale-i-j-1]];
      coeffs = [Math.max(0, 1-rx-ry), rx, ry];
    }} else {{
      verts = [[i+1,j+1,scale-i-j-2], [i+1,j,scale-i-j-1], [i,j+1,scale-i-j-1]];
      coeffs = [rx+ry-1, 1-ry, 1-rx];
    }}
    const coeffSum = Math.max(eps, coeffs[0] + coeffs[1] + coeffs[2]);
    coeffs = coeffs.map(v => v / coeffSum);
    const L = verts.map(v => layoutAtGrid(v[0], v[1], v[2]));
    if (L.some(v => !v)) return nearestGridLayout(w);
    return L[0].map((_, idx) => [
      coeffs[0]*Number(L[0][idx][0]) + coeffs[1]*Number(L[1][idx][0]) + coeffs[2]*Number(L[2][idx][0]),
      coeffs[0]*Number(L[0][idx][1]) + coeffs[1]*Number(L[1][idx][1]) + coeffs[2]*Number(L[2][idx][1]),
    ]);
  }}
  function baseTraceIndices() {{
    const names = new Set(records.map(r => String(r.genre)));
    const out = [];
    (plot.data || []).forEach((trace, i) => {{ if (names.has(String(trace.name))) out.push(i); }});
    return out;
  }}
  function updatePointCoordinates() {{
    const byGenre = new Map();
    for (const r of records) {{
      const pt = currentPoints[Number(r.idx)];
      if (!pt) continue;
      const g = String(r.genre);
      if (!byGenre.has(g)) byGenre.set(g, {{x: [], y: []}});
      byGenre.get(g).x.push(Number(pt[0]));
      byGenre.get(g).y.push(Number(pt[1]));
    }}
    for (const traceIdx of baseTraceIndices()) {{
      const trace = plot.data[traceIdx] || {{}};
      const vals = byGenre.get(String(trace.name));
      if (vals) Plotly.restyle(plot, {{x: [vals.x], y: [vals.y]}}, [traceIdx]);
    }}
  }}
  function rankedRows(sourceIdx, w) {{
    const candidates = ((simMap[String(sourceIdx)] || {{}}).candidates || []);
    const rows = candidates.map(c => {{
      const score = w.maest*Number(c.maest_score_norm||0) + w.tempo*Number(c.tempo_score_norm||0) + w.chroma*Number(c.chroma_score_norm||0);
      return {{...c, score}};
    }}).sort((a,b) => b.score - a.score);
    const temp = Math.max(1e-6, Number(config.temperature || 0.08));
    const maxScore = rows.length ? Number(rows[0].score || 0) : 0;
    let sum = 0;
    rows.forEach(r => {{ r._exp = Math.exp((Number(r.score || 0) - maxScore) / temp); sum += r._exp; }});
    rows.forEach((r, i) => {{ r.rank = i + 1; r.probability = sum > 0 ? r._exp / sum : 0; }});
    return rows;
  }}
  function clearTraceSet(indices) {{
    if (!indices.length || !window.Plotly) return [];
    try {{ Plotly.deleteTraces(plot, indices.slice().sort((a,b) => b-a)); }} catch (err) {{}}
    return [];
  }}
  function addTraceSet(traces) {{
    if (!traces.length || !window.Plotly) return [];
    const start = (plot.data || []).length;
    try {{ Plotly.addTraces(plot, traces); }} catch (err) {{ return []; }}
    return Array.from({{length: traces.length}}, (_, i) => start + i);
  }}
  function colorForDelta(delta, alpha) {{
    const scale = Math.max(1e-6, Number(config.bpm_color_scale_pct || 0.10));
    const t = clamp(Number(delta || 0) / scale, -1, 1);
    const f = Math.abs(t);
    const base = [155, 155, 155];
    const hot = [218, 65, 45];
    const cold = [45, 98, 210];
    const target = t >= 0 ? hot : cold;
    const rgb = base.map((v, i) => Math.round(v + (target[i] - v) * f));
    return 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + alpha + ')';
  }}
  function linkTraces(sourceIdx, rows, limit, highlighted) {{
    const src = currentPoints[Number(sourceIdx)];
    if (!src) return [];
    const traces = [];
    for (const row of rows.slice(0, limit)) {{
      const dst = currentPoints[Number(row.idx)];
      if (!dst) continue;
      const prob = Number(row.probability || 0);
      traces.push({{
        type: 'scatter',
        mode: 'lines',
        x: [Number(src[0]), Number(dst[0])],
        y: [Number(src[1]), Number(dst[1])],
        line: {{
          color: colorForDelta(row.bpm_delta_frac, highlighted ? clamp(0.40 + prob * 2.2, 0.40, 0.90) : 0.13),
          width: highlighted ? clamp(1.8 + prob * 10.0, 1.8, 5.0) : 0.85,
          dash: highlighted ? 'solid' : 'dot',
        }},
        hoverinfo: 'skip',
        showlegend: false,
      }});
    }}
    return traces;
  }}
  function renderBackgroundLinks(w) {{
    backgroundTraceIndices = clearTraceSet(backgroundTraceIndices);
    const limit = Math.max(0, Number(config.background_links_per_song || 0));
    if (limit <= 0) return;
    const traces = [];
    for (const sourceIdx of Object.keys(simMap)) {{
      traces.push(...linkTraces(Number(sourceIdx), rankedRows(Number(sourceIdx), w), limit, false));
    }}
    backgroundTraceIndices = addTraceSet(traces);
  }}
  function renderSelectedLinks(w) {{
    selectedTraceIndices = clearTraceSet(selectedTraceIndices);
    if (selectedIdx === null) return;
    const limit = Math.max(0, Number(config.click_links_per_song || 0));
    selectedTraceIndices = addTraceSet(linkTraces(selectedIdx, rankedRows(selectedIdx, w), limit, true));
  }}
  function renderPanel(w) {{
    if (selectedIdx === null) return;
    const rows = rankedRows(selectedIdx, w).slice(0, Number(config.top_k_rows || 25));
    const body = rows.map(row => '<tr>' +
      '<td class="num">' + row.rank + '</td><td class="num">' + esc(row.track_number) + '</td><td>' + esc(row.mix_slug) + '</td>' +
      '<td>' + esc(row.title) + '</td><td>' + esc(row.artists) + '</td><td>' + esc(row.genre) + '</td><td>' + esc(row.key) + '</td>' +
      '<td class="num">' + signedPct(row.bpm_delta_frac) + '</td><td class="num">' + fmt(row.score,4) + '</td>' +
      '<td class="num">' + fmt(row.maest_score_norm,4) + '</td><td class="num">' + fmt(row.tempo_score_norm,4) + '</td><td class="num">' + fmt(row.chroma_score_norm,4) + '</td>' +
      '<td class="num">' + fmt(row.maest_similarity,4) + '</td><td class="num">' + fmt(row.tempo_similarity,4) + '</td><td class="num">' + fmt(row.chroma_similarity,4) + '</td>' +
      '<td class="num">' + pct(row.probability) + '</td></tr>').join('');
    els.panel.innerHTML = '<b>Top ' + Number(config.top_k_rows || 25) + ' MAEST/tempo/chroma matches</b>' +
      '<div class="muted">Ranking uses normalized component scores derived from the same normalized distances as the layout. Raw similarities are shown for diagnostics.</div>' +
      '<div class="table-wrap"><table><thead><tr><th class="num">#</th><th class="num">Track</th><th>Mix</th><th>Title</th><th>Artists</th><th>Genre</th><th>Key</th>' +
      '<th class="num">dBPM%</th><th class="num">Score</th><th class="num">MAEST norm</th><th class="num">Tempo norm</th><th class="num">Chroma norm</th>' +
      '<th class="num">MAEST raw</th><th class="num">Tempo raw</th><th class="num">Chroma raw</th><th class="num">Prob</th></tr></thead><tbody>' + body + '</tbody></table></div>';
  }}
  function renderAll() {{
    const w = weights();
    selectedTraceIndices = clearTraceSet(selectedTraceIndices);
    backgroundTraceIndices = clearTraceSet(backgroundTraceIndices);
    els.maVal.textContent = fmt(w.maest, 3);
    els.teVal.textContent = fmt(w.tempo, 3);
    els.chVal.textContent = fmt(w.chroma, 3);
    updateSimplexHandle(w);
    els.summary.innerHTML = 'Normalized weights: MAEST <b>' + pct(w.maest) + '</b>, tempo <b>' + pct(w.tempo) + '</b>, chroma <b>' + pct(w.chroma) + '</b>.';
    currentPoints = simplexInterpolatedPoints(w);
    updatePointCoordinates();
    renderBackgroundLinks(w);
    renderSelectedLinks(w);
    renderPanel(w);
  }}
  [els.ma, els.te, els.ch].forEach(el => el.addEventListener('input', () => {{
    normalizeSliderWeights();
    renderAll();
  }}));
  if (els.simplex) {{
    let draggingSimplex = false;
    els.simplex.addEventListener('pointerdown', ev => {{
      draggingSimplex = true;
      els.simplex.setPointerCapture(ev.pointerId);
      setWeightsFromSimplexEvent(ev);
    }});
    els.simplex.addEventListener('pointermove', ev => {{
      if (!draggingSimplex) return;
      setWeightsFromSimplexEvent(ev);
    }});
    els.simplex.addEventListener('pointerup', ev => {{
      draggingSimplex = false;
      try {{ els.simplex.releasePointerCapture(ev.pointerId); }} catch (err) {{}}
    }});
    els.simplex.addEventListener('pointercancel', () => {{ draggingSimplex = false; }});
  }}
  if (plot && plot.on) {{
    plot.on('plotly_click', ev => {{
      if (!ev || !ev.points || !ev.points.length) return;
      const c = ev.points[0].customdata || [];
      const idx = Number(c[13]);
      if (!Number.isFinite(idx)) return;
      selectedIdx = idx;
      els.meta.innerHTML = '<b>' + esc(c[0]) + '</b><br>Artists: ' + esc(c[1]) + '<br>Genre: ' + esc(c[2]) +
        '<br>Mix: ' + esc(c[14]) + '<br>Key: ' + esc(c[3]) + '<br>CSV BPM: ' + esc(c[4]) +
        '<br>Estimated BPM: ' + fmt(c[5], 2) + ' (confidence=' + fmt(c[6], 3) + ')<br>Track #: ' + esc(c[7]) + '<br>File: ' + esc(c[8]);
      if (c[9]) {{
        els.audio.src = c[9];
        const p = els.audio.play();
        if (p && p.catch) p.catch(() => {{}});
      }} else {{
        els.audio.removeAttribute('src');
        els.audio.load();
      }}
      renderSelectedLinks(weights());
      renderPanel(weights());
    }});
  }}
  renderAll();
}})();
</script>
"""
    return "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{title}</title>",
            style,
            "</head>",
            "<body>",
            f"<h1>{title}</h1>",
            data_scripts,
            plot_html,
            controls,
            script,
            "</body>",
            "</html>",
        ]
    )


def export_interactive_umap_simplex(
    *,
    project_root: Path = PROJECT_ROOT,
    mix_slugs: list[str] | None = None,
    output_file: Path | None = None,
    random_state: int = 7777,
    n_neighbors: int = 10,
    min_dist: float = 0.2,
    step: float = 0.1,
    align_layouts: bool = True,
    temperature: float = 0.08,
    background_links_per_song: int = 3,
    click_links_per_song: int = 5,
    bpm_color_scale_pct: float = 0.10,
) -> Path:
    project_root = project_root.expanduser().resolve()
    mix_slugs = mix_slugs or ["aries-mix", "ara-mix"]
    dataset_tag = "__".join(
        str((project_root / "music" / slug / f"{slug.replace('-', '_')}_tracks.csv").stem)
        for slug in mix_slugs
    )
    output_file = output_file or (
        project_root / "data" / "exports" / f"{dataset_tag}_interactive_umap_simplex_maest_tempo_chroma.html"
    )
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records, features = _load_combined_records_and_features(
        project_root=project_root,
        mix_slugs=mix_slugs,
        maest_dir=(project_root / "data" / "maest_embeddings"),
        chroma_dir=(project_root / "data" / "chroma_embeddings"),
        tempo_dir=(project_root / "data" / "tempo_embeddings"),
        html_output_dir=output_file.parent,
        snippet_seconds=8.0,
        snippet_middle_fraction=0.66,
        snippet_hop_seconds=0.25,
        snippet_cache_overwrite=False,
    )
    matrices = _component_matrices_3way(
        features,
        tempo_bandwidth=0.06,
        tempo_decay=0.5,
        tempo_allow_octave=True,
        tempo_octave_penalty=0.5,
        tempo_similarity_shape="gaussian",
        tempo_softflat_sharpness=8.0,
        tempo_use_confidence=False,
        harmonic_exact_weight=1.0,
        harmonic_first_fifth_weight=0.0,
        harmonic_second_fifth_weight=0.0,
        harmonic_other_weight=0.0,
        harmonic_self_normalize=True,
    )
    grid = _simplex_grid(step)
    layouts = _compute_umap_simplex_layouts(
        D_maest=matrices["maest_distance"],
        D_tempo=matrices["tempo_distance"],
        D_chroma=matrices["chroma_distance"],
        grid=grid,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        align=align_layouts,
    )
    initial_key = _simplex_key((0.6, 0.2, 0.2))
    initial_coords = np.asarray(layouts[initial_key], dtype=np.float32)
    similarity_payload = _build_similarity_payload_3way(
        records=records,
        maest_similarity=matrices["maest_similarity"],
        tempo_similarity=matrices["tempo_similarity"],
        chroma_similarity=matrices["chroma_similarity"],
        maest_distance=matrices["maest_distance"],
        tempo_distance=matrices["tempo_distance"],
        chroma_distance=matrices["chroma_distance"],
        temperature=temperature,
    )
    mix_title = " + ".join(mix_slugs)
    title = f"{mix_title}: Interactive UMAP Simplex MAEST/Tempo/Chroma"
    plot_div_id = f"{dataset_tag}_umap_simplex_maest_tempo_chroma".replace("-", "_")
    plot_html = _build_plot(
        records,
        initial_coords,
        plot_div_id=plot_div_id,
        title=title,
        xaxis_title="UMAP-1 (barycentric MAEST/tempo/chroma)",
        yaxis_title="UMAP-2 (barycentric MAEST/tempo/chroma)",
    )
    html = _build_simplex_html(
        plot_html=plot_html,
        records=records,
        layouts=layouts,
        similarity_payload=similarity_payload,
        plot_div_id=plot_div_id,
        title=title,
        step=step,
        top_k_rows=25,
        temperature=temperature,
        background_links_per_song=background_links_per_song,
        click_links_per_song=click_links_per_song,
        bpm_color_scale_pct=bpm_color_scale_pct,
    )
    output_file.write_text(html, encoding="utf-8")
    print(f"Loaded aligned tracks: {len(records)}")
    print(f"Computed UMAP simplex layouts: {len(layouts)}")
    print(f"Standalone HTML saved to: {output_file}")
    return output_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an interactive 3-way UMAP simplex HTML.")
    parser.add_argument("--mix-slug", action="append", dest="mix_slugs")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=7777)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--no-align-layouts", action="store_true")
    parser.add_argument("--background-links-per-song", type=int, default=3)
    parser.add_argument("--click-links-per-song", type=int, default=5)
    parser.add_argument("--bpm-color-scale-pct", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_interactive_umap_simplex(
        mix_slugs=args.mix_slugs,
        output_file=args.output_file,
        random_state=args.random_state,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        step=args.step,
        align_layouts=not args.no_align_layouts,
        background_links_per_song=args.background_links_per_song,
        click_links_per_song=args.click_links_per_song,
        bpm_color_scale_pct=args.bpm_color_scale_pct,
    )


if __name__ == "__main__":
    main()
