"""Export an interactive UMAP visualization from precomputed MAEST/tempo distances."""

from __future__ import annotations

import argparse
import csv
import json
import os
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from scipy.linalg import orthogonal_procrustes

from djprojectexploration.audio_snippets import ensure_cached_snippet
from djprojectexploration.multimodal_compatibility import (
    SongFeatureSet,
    SongMetadata,
    _pairwise_cosine_similarity_matrix,
    _pairwise_tempo_similarity_matrix,
    load_aries_mix_feature_set,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _norm_token(value: str) -> str:
    return Path(str(value).strip()).name.lower()


def _json_script_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _row_token(row: dict[str, str]) -> str | None:
    for key in ("mp3_name", "filename", "filepath", "location"):
        value = (row.get(key) or "").strip()
        if value:
            return _norm_token(value)
    return None


GENRE_SIMPLIFICATION_MAP = {
    "alternative": "Alternative / Indie",
    "electronica": "Alternative / Indie",
    "experimental electronic": "Alternative / Indie",
    "french pop": "Pop / Dance Pop",
    "pop": "Pop / Dance Pop",
    "dance pop": "Pop / Dance Pop",
    "r&b": "Hip-Hop / R&B",
    "rap": "Hip-Hop / R&B",
    "hip hop": "Hip-Hop / R&B",
    "dancehall": "Global / Dancehall",
    "reggaeton": "Global / Dancehall",
    "house": "House",
    "deep house": "House",
    "disco": "House",
    "nu disco": "House",
    "progressive house": "House",
    "tech house": "House",
    "bass house": "Bass / Dubstep",
    "bass house]": "Bass / Dubstep",
    "dubstep": "Bass / Dubstep",
    "drum & bass": "Bass / Dubstep",
    "future bass": "Bass / Dubstep",
    "trap": "Bass / Dubstep",
    "uk garage": "Bass / Dubstep",
    "big room": "Electro / Big Room",
    "electro house": "Electro / Big Room",
    "future house": "Electro / Big Room",
    "future rave": "Electro / Big Room",
    "melodic techno": "Techno",
    "techno": "Techno",
    "synthwave": "Techno",
    "progressive trance": "Trance",
    "psytrance": "Trance",
    "tech trance": "Trance",
    "trance": "Trance",
    "uplifting trance": "Trance",
    "hardstyle": "Hard Dance",
    "ambient": "Electronic / Other",
    "chillout": "Electronic / Other",
    "electronic": "Electronic / Other",
    "other": "Electronic / Other",
}


def simplify_genre(raw_genre: str) -> str:
    genre = str(raw_genre or "").strip()
    if not genre:
        return "Unknown"
    key = " ".join(genre.lower().replace("_", " ").split())
    return GENRE_SIMPLIFICATION_MAP.get(key, genre)


def _resolve_audio_path(
    row: dict[str, str],
    fallback_filename: str,
    *,
    tracklist_csv: Path,
    music_dir: Path,
) -> Path | None:
    for key in ("filepath", "location"):
        value = (row.get(key) or "").strip()
        if value:
            path = Path(value).expanduser()
            if path.exists():
                return path.resolve()

    mp3_name = (row.get("mp3_name") or row.get("filename") or fallback_filename or "").strip()
    if not mp3_name:
        return None

    rel = Path(mp3_name)
    candidates = [
        tracklist_csv.parent / rel,
        PROJECT_ROOT / rel,
        music_dir / rel.name,
        PROJECT_ROOT / "music" / rel.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_combined_records_and_features(
    *,
    project_root: Path,
    mix_slugs: list[str],
    maest_dir: Path,
    chroma_dir: Path,
    tempo_dir: Path,
    html_output_dir: Path,
    snippet_seconds: float,
    snippet_middle_fraction: float,
    snippet_hop_seconds: float,
    snippet_cache_overwrite: bool,
) -> tuple[list[dict[str, Any]], SongFeatureSet]:
    combined_metadata: list[SongMetadata] = []
    combined_maest: list[np.ndarray] = []
    combined_chroma: list[np.ndarray] = []
    combined_chroma_pitch: list[np.ndarray] = []
    combined_tempo_bpm: list[float] = []
    combined_tempo_conf: list[float] = []
    records: list[dict[str, Any]] = []

    for mix_slug in mix_slugs:
        tracklist_csv = project_root / "music" / mix_slug / f"{mix_slug.replace('-', '_')}_tracks.csv"
        if not tracklist_csv.exists():
            raise FileNotFoundError(f"Tracklist CSV not found: {tracklist_csv}")

        music_dir = tracklist_csv.parent
        snippet_cache_dir = project_root / "data" / "snippets" / tracklist_csv.stem
        mix_features = load_aries_mix_feature_set(
            mix_csv_path=tracklist_csv,
            maest_dir=maest_dir,
            chroma_dir=chroma_dir,
            tempo_dir=tempo_dir,
            chroma_use_base_only=True,
        )

        csv_rows = _read_csv_rows(tracklist_csv)
        csv_by_token: dict[str, dict[str, str]] = {}
        for row in csv_rows:
            token = _row_token(row)
            if token and token not in csv_by_token:
                csv_by_token[token] = row

        for local_i, meta in enumerate(mix_features.metadata):
            global_idx = len(records)
            global_track_number = global_idx + 1
            token = _norm_token(meta.filename)
            row = csv_by_token.get(token, {})

            title = (row.get("title") or meta.title or Path(meta.filename).stem).strip()
            artists = (row.get("artists") or meta.artist or "").strip()
            key_tag = (row.get("key") or "").strip()
            bpm_tag = (row.get("bpm") or "").strip()
            raw_genre = (row.get("genre") or meta.genre or "Unknown").strip() or "Unknown"
            genre = simplify_genre(raw_genre)
            track_num_tag = (row.get("track_number") or row.get("#") or str(meta.track_number)).strip()
            est_bpm = float(mix_features.tempo_bpm[local_i])
            est_conf = float(mix_features.tempo_confidence[local_i])
            audio_path = _resolve_audio_path(
                row,
                fallback_filename=meta.filename,
                tracklist_csv=tracklist_csv,
                music_dir=music_dir,
            )

            try:
                snippet = ensure_cached_snippet(
                    audio_path=audio_path,
                    output_dir=snippet_cache_dir,
                    key=meta.filename,
                    snippet_seconds=snippet_seconds,
                    middle_fraction=snippet_middle_fraction,
                    hop_seconds=snippet_hop_seconds,
                    overwrite=snippet_cache_overwrite,
                    project_root=project_root,
                )
            except Exception:
                snippet = None

            if snippet is None:
                snippet_uri = ""
                snippet_path = ""
                snippet_start = 0.0
                snippet_end = 0.0
                snippet_rms = 0.0
            else:
                snippet_file = snippet.snippet_path.resolve()
                snippet_path = str(snippet_file)
                snippet_start = float(snippet.start_seconds)
                snippet_end = float(snippet.end_seconds)
                snippet_rms = float(snippet.rms)
                snippet_uri = Path(os.path.relpath(snippet_file, html_output_dir.resolve())).as_posix()

            combined_metadata.append(
                SongMetadata(
                    track_number=int(global_track_number),
                    title=str(title),
                    artist=str(artists),
                    filename=str(meta.filename),
                    genre=str(genre),
                )
            )
            combined_maest.append(np.asarray(mix_features.maest[local_i], dtype=np.float32))
            combined_chroma.append(np.asarray(mix_features.chroma[local_i], dtype=np.float32))
            combined_chroma_pitch.append(np.asarray(mix_features.chroma_pitch[local_i], dtype=np.float32))
            combined_tempo_bpm.append(est_bpm)
            combined_tempo_conf.append(est_conf)

            records.append(
                {
                    "idx": global_idx,
                    "global_track_number": int(global_track_number),
                    "track_number": str(track_num_tag),
                    "filename": meta.filename,
                    "title": title,
                    "artists": artists,
                    "genre": genre,
                    "raw_genre": raw_genre,
                    "key": key_tag,
                    "csv_bpm": bpm_tag,
                    "est_bpm": est_bpm,
                    "est_conf": est_conf,
                    "audio_path": "" if audio_path is None else str(audio_path),
                    "snippet_uri": snippet_uri,
                    "snippet_path": snippet_path,
                    "snippet_start": snippet_start,
                    "snippet_end": snippet_end,
                    "snippet_rms": snippet_rms,
                    "mix_slug": mix_slug,
                }
            )

    if len(records) < 2:
        raise ValueError("Need at least two aligned tracks.")

    features = SongFeatureSet(
        metadata=combined_metadata,
        maest=np.vstack(combined_maest).astype(np.float32),
        chroma=np.vstack(combined_chroma).astype(np.float32),
        chroma_pitch=np.vstack(combined_chroma_pitch).astype(np.float32),
        tempo_bpm=np.asarray(combined_tempo_bpm, dtype=np.float32),
        tempo_confidence=np.asarray(combined_tempo_conf, dtype=np.float32),
    )
    return records, features


def _normalize_distance_matrix(D: np.ndarray, *, percentile: float = 95.0) -> np.ndarray:
    out = np.asarray(D, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError(f"Expected square distance matrix, got shape {out.shape}.")
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 0.0)

    mask = ~np.eye(out.shape[0], dtype=bool)
    vals = out[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("Distance matrix has no finite off-diagonal values.")
    scale = float(np.percentile(vals, percentile))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(vals))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Distance matrix scale is zero or invalid.")

    out = np.clip(out / scale, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(out, 0.0)
    return out


def _component_matrices(
    features: SongFeatureSet,
    *,
    tempo_bandwidth: float,
    tempo_decay: float,
    tempo_allow_octave: bool,
    tempo_octave_penalty: float,
    tempo_similarity_shape: str,
    tempo_softflat_sharpness: float,
    tempo_use_confidence: bool,
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

    maest_distance = _normalize_distance_matrix(1.0 - maest_similarity)
    tempo_distance = _normalize_distance_matrix(1.0 - tempo_similarity)

    return {
        "maest_similarity": maest_similarity,
        "tempo_similarity": tempo_similarity,
        "maest_distance": maest_distance,
        "tempo_distance": tempo_distance,
    }


def _combined_distance(D_maest: np.ndarray, D_tempo: np.ndarray, *, maest_weight: float) -> np.ndarray:
    wm = float(np.clip(maest_weight, 0.0, 1.0))
    wt = 1.0 - wm
    D = np.sqrt((wm * np.square(D_maest)) + (wt * np.square(D_tempo))).astype(np.float32)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def _align_to_reference(reference: np.ndarray, coords: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(coords, dtype=np.float64)
    ref_center = np.mean(ref, axis=0, keepdims=True)
    cur_center = np.mean(cur, axis=0, keepdims=True)
    ref0 = ref - ref_center
    cur0 = cur - cur_center

    ref_norm = float(np.linalg.norm(ref0))
    cur_norm = float(np.linalg.norm(cur0))
    if ref_norm <= 0.0 or cur_norm <= 0.0:
        return cur.astype(np.float32)

    R, _ = orthogonal_procrustes(cur0 / cur_norm, ref0 / ref_norm)
    aligned = ((cur0 / cur_norm) @ R) * ref_norm + ref_center
    return aligned.astype(np.float32)


def _compute_umap_layouts(
    *,
    D_maest: np.ndarray,
    D_tempo: np.ndarray,
    weight_values: list[float],
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
    previous: np.ndarray | None = None
    n = D_maest.shape[0]
    effective_neighbors = min(max(2, int(n_neighbors)), max(2, n - 1))

    for maest_weight in weight_values:
        D = _combined_distance(D_maest, D_tempo, maest_weight=maest_weight)
        reducer = UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=float(min_dist),
            metric="precomputed",
            random_state=int(random_state),
        )
        coords = np.asarray(reducer.fit_transform(D), dtype=np.float32)
        if align and previous is not None:
            coords = _align_to_reference(previous, coords)
        previous = coords
        key = f"{maest_weight:.1f}"
        layouts[key] = [[float(x), float(y)] for x, y in coords]

    return layouts


def _build_similarity_payload(
    *,
    records: list[dict[str, Any]],
    maest_similarity: np.ndarray,
    tempo_similarity: np.ndarray,
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
                    "global_track_number": int(cand["global_track_number"]),
                    "track_number": str(cand["track_number"]),
                    "mix_slug": str(cand["mix_slug"]),
                    "title": str(cand["title"]),
                    "artists": str(cand["artists"]),
                    "genre": str(cand["genre"]),
                    "key": str(cand["key"]),
                    "csv_bpm": str(cand["csv_bpm"]),
                    "est_bpm": cand_bpm,
                    "est_conf": float(cand["est_conf"]),
                    "filename": str(cand["filename"]),
                    "maest_similarity": float(maest_similarity[src_idx, cand_idx]),
                    "tempo_similarity": float(tempo_similarity[src_idx, cand_idx]),
                    "bpm_delta_frac": bpm_delta_frac,
                }
            )
        payload[str(src_idx)] = {"candidates": rows, "temperature": float(temperature)}
    return payload


def _build_plot(
    records: list[dict[str, Any]],
    initial_coords: np.ndarray,
    *,
    plot_div_id: str,
    title: str,
    xaxis_title: str = "UMAP-1 (precomputed weighted MAEST/tempo distance)",
    yaxis_title: str = "UMAP-2 (precomputed weighted MAEST/tempo distance)",
    legend_title: str = "Genre",
    show_axis_ticks: bool = True,
    show_grid: bool = True,
) -> str:
    genres = sorted({str(r["genre"]) for r in records}, key=lambda g: g.lower())
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939",
    ]
    genre_color = {g: palette[i % len(palette)] for i, g in enumerate(genres)}

    fig = go.Figure()
    for genre in genres:
        pts = [r for r in records if str(r["genre"]) == genre]
        idxs = [int(p["idx"]) for p in pts]
        custom = [
            [
                p["title"], p["artists"], p["genre"], p["key"], p["csv_bpm"],
                p["est_bpm"], p["est_conf"], p["track_number"], p["filename"],
                p["snippet_uri"], p["snippet_start"], p["snippet_end"], p["snippet_rms"],
                p["idx"], p["mix_slug"], p.get("raw_genre", p["genre"]),
            ]
            for p in pts
        ]
        fig.add_scatter(
            x=initial_coords[idxs, 0],
            y=initial_coords[idxs, 1],
            mode="markers",
            name=genre,
            marker={"size": 9, "color": genre_color[genre], "line": {"color": "black", "width": 0.5}},
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Artists: %{customdata[1]}<br>"
                "Genre: %{customdata[2]}<br>"
                "Tagged Genre: %{customdata[15]}<br>"
                "Mix: %{customdata[14]}<br>"
                "Key: %{customdata[3]}<br>"
                "CSV BPM: %{customdata[4]}<br>"
                "Estimated BPM: %{customdata[5]:.2f} (conf=%{customdata[6]:.3f})<br>"
                "Track #: %{customdata[7]}<br>"
                "File: %{customdata[8]}<extra></extra>"
            ),
        )

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=1180,
        height=760,
        margin={"t": 80, "r": 30, "b": 50, "l": 55},
        legend_title_text=legend_title,
        template="plotly_white",
    )
    fig.update_xaxes(showticklabels=show_axis_ticks, showgrid=show_grid, zeroline=show_grid)
    fig.update_yaxes(showticklabels=show_axis_ticks, showgrid=show_grid, zeroline=show_grid)
    return fig.to_html(include_plotlyjs=True, full_html=False, div_id=plot_div_id)


def _build_html(
    *,
    plot_html: str,
    records: list[dict[str, Any]],
    layouts: dict[str, list[list[float]]],
    similarity_payload: dict[str, dict[str, Any]],
    plot_div_id: str,
    title: str,
    default_maest_weight: float,
    top_k_rows: int,
    background_links_per_song: int,
    click_links_per_song: int,
    bpm_color_scale_pct: float,
    temperature: float,
    method_name: str = "UMAP",
    layout_description: str = "Layout interpolates between precomputed 10% UMAP splits.",
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
            f'<script id="umap-records-json" type="application/json">{_json_script_payload(record_payload)}</script>',
            f'<script id="umap-layouts-json" type="application/json">{_json_script_payload(layouts)}</script>',
            f'<script id="umap-sim-json" type="application/json">{_json_script_payload(similarity_payload)}</script>',
            f'<script id="umap-config-json" type="application/json">{_json_script_payload({"default_maest_weight": default_maest_weight, "top_k_rows": top_k_rows, "background_links_per_song": background_links_per_song, "click_links_per_song": click_links_per_song, "bpm_color_scale_pct": bpm_color_scale_pct, "temperature": temperature})}</script>',
        ]
    )

    controls = """
<section class="panel">
  <div class="weight-row">
    <label>MAEST layout/score weight <input id="maest-weight" type="range" min="0" max="1" step="0.01"></label>
    <output id="maest-weight-val">1.00</output>
    <div id="weight-summary" class="muted"></div>
  </div>
  <div id="track-meta" class="box">Click a point to play its snippet and show MAEST/tempo recommendations.</div>
  <audio id="track-audio" controls></audio>
  <div id="similarity-panel" class="box">Recommendations will appear here after you click a point.</div>
</section>
"""
    style = """
<style>
  :root { color-scheme: light; --line:#d9dee8; --ink:#17202a; --muted:#5b6678; --accent:#0c6b58; }
  body { font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 18px; color: var(--ink); background: #f6f8fb; }
  h1 { font-size: 22px; margin: 0 0 14px; }
  .panel { display: grid; gap: 10px; margin-top: 12px; }
  .weight-row, .box { border: 1px solid var(--line); background: #fff; padding: 10px; }
  .weight-row { display: grid; grid-template-columns: minmax(260px, 1fr) 64px minmax(280px, 1fr); gap: 10px; align-items: center; }
  label { display: grid; gap: 5px; font-size: 12px; color: var(--muted); }
  output { text-align: right; font-variant-numeric: tabular-nums; }
  audio { width: 100%; display: block; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { border-bottom: 1px solid #edf0f5; padding: 5px 6px; vertical-align: top; }
  th { position: sticky; top: 0; background: #f9fafc; z-index: 1; text-align: left; color: #3c4758; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .muted { color: var(--muted); font-size: 12px; }
  .table-wrap { max-height: 380px; overflow: auto; border: 1px solid #edf0f5; }
  @media (max-width: 900px) { .weight-row { grid-template-columns: 1fr; } output { text-align:left; } }
</style>
"""
    script = f"""
<script>
(function() {{
  const records = JSON.parse(document.getElementById('umap-records-json').textContent);
  const layouts = JSON.parse(document.getElementById('umap-layouts-json').textContent);
  const simMap = JSON.parse(document.getElementById('umap-sim-json').textContent);
  const config = JSON.parse(document.getElementById('umap-config-json').textContent);
  const plot = document.getElementById('{plot_div_id}');
  const slider = document.getElementById('maest-weight');
  const sliderVal = document.getElementById('maest-weight-val');
  const summary = document.getElementById('weight-summary');
  const meta = document.getElementById('track-meta');
  const audio = document.getElementById('track-audio');
  const panel = document.getElementById('similarity-panel');

  let currentPoints = [];
  let selectedIdx = null;
  let linkTraceIndices = [];
  let backgroundTraceIndices = [];

  function fmt(v, d) {{ return Number(v || 0).toFixed(d); }}
  function pct(v) {{ return (Number(v || 0) * 100).toFixed(1) + '%'; }}
  function signedPct(v) {{ const n = Number(v || 0) * 100; return (n >= 0 ? '+' : '') + n.toFixed(1) + '%'; }}
  function clamp(v, lo, hi) {{ return Math.min(hi, Math.max(lo, v)); }}
  function esc(v) {{
    return String(v == null ? '' : v).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
  }}
  function parsedWeight() {{ return clamp(Number(slider.value || 0), 0, 1); }}
  function weightKey(w) {{ return (Math.round(w * 10) / 10).toFixed(1); }}

  function interpolatedPoints(w) {{
    const lo = Math.floor(w * 10) / 10;
    const hi = Math.ceil(w * 10) / 10;
    const loKey = lo.toFixed(1);
    const hiKey = hi.toFixed(1);
    const A = layouts[loKey] || layouts[weightKey(w)];
    const B = layouts[hiKey] || A;
    if (!A || !B || loKey === hiKey) return A || [];
    const t = (w - lo) / Math.max(0.1, hi - lo);
    return A.map((p, i) => [
      Number(p[0]) + (Number(B[i][0]) - Number(p[0])) * t,
      Number(p[1]) + (Number(B[i][1]) - Number(p[1])) * t,
    ]);
  }}

  function baseTraceIndices() {{
    const names = new Set(records.map(r => String(r.genre)));
    const out = [];
    (plot.data || []).forEach((trace, i) => {{
      if (names.has(String(trace.name))) out.push(i);
    }});
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
      const values = byGenre.get(String(trace.name));
      if (!values) continue;
      Plotly.restyle(plot, {{x: [values.x], y: [values.y]}}, [traceIdx]);
    }}
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

  function rankedRows(sourceIdx, maestWeight) {{
    const entry = simMap[String(sourceIdx)] || {{}};
    const candidates = Array.isArray(entry.candidates) ? entry.candidates : [];
    const tempoWeight = 1 - maestWeight;
    const rows = candidates.map(c => {{
      const score = maestWeight * Number(c.maest_similarity || 0) + tempoWeight * Number(c.tempo_similarity || 0);
      return {{...c, score}};
    }}).sort((a, b) => b.score - a.score);
    if (!rows.length) return rows;
    const temp = Math.max(1e-6, Number(config.temperature || 0.08));
    const maxScore = Number(rows[0].score || 0);
    let sum = 0;
    for (const row of rows) {{
      row._exp = Math.exp((Number(row.score || 0) - maxScore) / temp);
      sum += row._exp;
    }}
    rows.forEach((row, i) => {{
      row.rank = i + 1;
      row.probability = sum > 0 ? row._exp / sum : 0;
    }});
    return rows;
  }}

  function linkTraces(sourceIdx, rows, limit, highlighted) {{
    const src = currentPoints[Number(sourceIdx)];
    if (!src) return [];
    const traces = [];
    for (const row of rows.slice(0, limit)) {{
      const dst = currentPoints[Number(row.idx)];
      if (!dst) continue;
      traces.push({{
        type: 'scatter',
        mode: 'lines',
        x: [Number(src[0]), Number(dst[0])],
        y: [Number(src[1]), Number(dst[1])],
        line: {{
          color: colorForDelta(row.bpm_delta_frac, highlighted ? 0.65 : 0.16),
          width: highlighted ? 2.6 : 1.0,
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
    for (const key of Object.keys(simMap)) {{
      traces.push(...linkTraces(Number(key), rankedRows(Number(key), w), limit, false));
    }}
    backgroundTraceIndices = addTraceSet(traces);
  }}

  function renderSelectedLinks(w) {{
    linkTraceIndices = clearTraceSet(linkTraceIndices);
    if (selectedIdx === null) return;
    const limit = Math.max(0, Number(config.click_links_per_song || 0));
    linkTraceIndices = addTraceSet(linkTraces(selectedIdx, rankedRows(selectedIdx, w), limit, true));
  }}

  function renderPanel(w) {{
    if (selectedIdx === null) return;
    const rows = rankedRows(selectedIdx, w).slice(0, Number(config.top_k_rows || 25));
    const body = rows.map(row => '<tr>' +
      '<td class="num">' + row.rank + '</td>' +
      '<td class="num">' + esc(row.track_number) + '</td>' +
      '<td>' + esc(row.mix_slug) + '</td>' +
      '<td>' + esc(row.title) + '</td>' +
      '<td>' + esc(row.artists) + '</td>' +
      '<td>' + esc(row.genre) + '</td>' +
      '<td>' + esc(row.key) + '</td>' +
      '<td class="num">' + esc(row.csv_bpm) + '</td>' +
      '<td class="num">' + fmt(row.est_bpm, 2) + '</td>' +
      '<td class="num">' + signedPct(row.bpm_delta_frac) + '</td>' +
      '<td class="num">' + fmt(row.score, 4) + '</td>' +
      '<td class="num">' + fmt(row.maest_similarity, 4) + '</td>' +
      '<td class="num">' + fmt(row.tempo_similarity, 4) + '</td>' +
      '<td class="num">' + pct(row.probability) + '</td>' +
      '</tr>').join('');
    panel.innerHTML = '<b>Top ' + Number(config.top_k_rows || 25) + ' MAEST/tempo matches</b>' +
      '<div class="muted">Score = ' + fmt(w, 2) + ' * MAEST + ' + fmt(1 - w, 2) + ' * tempo. {layout_description}</div>' +
      '<div class="table-wrap"><table><thead><tr>' +
      '<th class="num">#</th><th class="num">Track</th><th>Mix</th><th>Title</th><th>Artists</th><th>Genre</th><th>Key</th>' +
      '<th class="num">CSV BPM</th><th class="num">Est BPM</th><th class="num">dBPM%</th><th class="num">Score</th>' +
      '<th class="num">MAEST</th><th class="num">Tempo</th><th class="num">Prob</th>' +
      '</tr></thead><tbody>' + body + '</tbody></table></div>';
  }}

  function renderWeightState() {{
    const w = parsedWeight();
    currentPoints = interpolatedPoints(w);
    sliderVal.textContent = fmt(w, 2);
    summary.innerHTML = 'MAEST <b>' + pct(w) + '</b> / tempo <b>' + pct(1 - w) + '</b>. Nearest 10% precomputed key: <b>' + weightKey(w) + '</b>.';
    updatePointCoordinates();
    linkTraceIndices = clearTraceSet(linkTraceIndices);
    renderBackgroundLinks(w);
    renderSelectedLinks(w);
    renderPanel(w);
  }}

  slider.value = String(Number(config.default_maest_weight || 1.0));
  currentPoints = interpolatedPoints(parsedWeight());
  slider.addEventListener('input', renderWeightState);

  if (plot && plot.on) {{
    plot.on('plotly_click', ev => {{
      if (!ev || !ev.points || !ev.points.length) return;
      const c = ev.points[0].customdata || [];
      const idx = Number(c[13]);
      if (!Number.isFinite(idx)) return;
      selectedIdx = idx;
      meta.innerHTML = '<b>' + esc(c[0]) + '</b><br>' +
        'Artists: ' + esc(c[1]) + '<br>' +
        'Genre: ' + esc(c[2]) + '<br>' +
        'Mix: ' + esc(c[14]) + '<br>' +
        'Key: ' + esc(c[3]) + '<br>' +
        'CSV BPM: ' + esc(c[4]) + '<br>' +
        'Estimated BPM: ' + fmt(c[5], 2) + ' (confidence=' + fmt(c[6], 3) + ')<br>' +
        'Track #: ' + esc(c[7]) + '<br>' +
        'File: ' + esc(c[8]) + '<br>' +
        'Snippet: ' + fmt(c[10], 2) + 's -> ' + fmt(c[11], 2) + 's | RMS ' + fmt(c[12], 5);
      if (c[9]) {{
        audio.src = c[9];
        const p = audio.play();
        if (p && p.catch) p.catch(() => {{}});
      }} else {{
        audio.removeAttribute('src');
        audio.load();
      }}
      renderSelectedLinks(parsedWeight());
      renderPanel(parsedWeight());
    }});
  }}

  renderWeightState();
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


def _render_layout_gif(
    *,
    records: list[dict[str, Any]],
    layouts: dict[str, list[list[float]]],
    output_file: Path,
    title: str,
    renderer: str = "matplotlib",
    fps: int = 8,
    tween_frames: int = 6,
    hold_frames: int = 4,
) -> None:
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(layouts.keys(), key=float)
    if not keys:
        raise ValueError("No layouts available for GIF rendering.")

    genre_values = sorted({str(r["genre"]) for r in records}, key=lambda g: g.lower())
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939",
    ]
    genre_color = {g: palette[i % len(palette)] for i, g in enumerate(genre_values)}
    colors = [genre_color[str(r["genre"])] for r in records]

    arrays = {key: np.asarray(layouts[key], dtype=np.float32) for key in keys}
    all_points = np.vstack([arrays[key] for key in keys])
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)
    xpad = max(1e-3, float(xmax - xmin) * 0.08)
    ypad = max(1e-3, float(ymax - ymin) * 0.08)

    frames: list[tuple[float, np.ndarray]] = []
    for idx, key in enumerate(keys):
        weight = float(key)
        current = arrays[key]
        for _ in range(max(1, int(hold_frames))):
            frames.append((weight, current))
        if idx >= len(keys) - 1:
            continue
        next_key = keys[idx + 1]
        nxt = arrays[next_key]
        for step in range(1, max(1, int(tween_frames)) + 1):
            t = step / float(max(1, int(tween_frames)) + 1)
            interp_weight = weight + (float(next_key) - weight) * t
            frames.append((interp_weight, current + (nxt - current) * t))

    renderer_key = str(renderer).strip().lower()
    if renderer_key == "plotly":
        _render_layout_gif_plotly(
            records=records,
            frames=frames,
            genre_color=genre_color,
            output_file=output_file,
            title=title,
            fps=fps,
            x_range=[float(xmin - xpad), float(xmax + xpad)],
            y_range=[float(ymin - ypad), float(ymax + ypad)],
        )
        return
    if renderer_key != "matplotlib":
        raise ValueError("renderer must be one of: 'matplotlib', 'plotly'.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=120)
    fig.subplots_adjust(right=0.78)
    scatter = ax.scatter(
        frames[0][1][:, 0],
        frames[0][1][:, 1],
        s=46,
        c=colors,
        edgecolors="black",
        linewidths=0.35,
        alpha=0.88,
    )
    ax.set_xlim(float(xmin - xpad), float(xmax + xpad))
    ax.set_ylim(float(ymin - ypad), float(ymax + ypad))
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(alpha=0.18)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=genre,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=6,
        )
        for genre, color in genre_color.items()
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=True)

    def update(frame: int):
        weight, points = frames[frame]
        scatter.set_offsets(points)
        ax.set_title(f"{title}\nMAEST {weight:.2f} / tempo {1.0 - weight:.2f}")
        return (scatter,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / max(1, int(fps)), blit=False)
    anim.save(output_file, writer=PillowWriter(fps=max(1, int(fps))))
    plt.close(fig)


def _render_layout_gif_plotly(
    *,
    records: list[dict[str, Any]],
    frames: list[tuple[float, np.ndarray]],
    genre_color: dict[str, str],
    output_file: Path,
    title: str,
    fps: int,
    x_range: list[float],
    y_range: list[float],
) -> None:
    from io import BytesIO

    from PIL import Image

    try:
        import kaleido  # noqa: F401
    except ImportError as exc:
        raise ImportError("Plotly GIF rendering requires kaleido. Install with: uv add kaleido") from exc

    frame_images: list[Image.Image] = []
    genre_to_indices: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        genre_to_indices.setdefault(str(record["genre"]), []).append(idx)

    for weight, points in frames:
        fig = go.Figure()
        for genre in sorted(genre_to_indices, key=lambda g: g.lower()):
            idxs = genre_to_indices[genre]
            pts = points[idxs]
            fig.add_scatter(
                x=pts[:, 0],
                y=pts[:, 1],
                mode="markers",
                name=genre,
                marker={
                    "size": 9,
                    "color": genre_color[genre],
                    "line": {"color": "black", "width": 0.5},
                    "opacity": 0.88,
                },
                hoverinfo="skip",
            )
        fig.update_layout(
            title={
                "text": f"{title}<br><sup>MAEST {weight:.2f} / tempo {1.0 - weight:.2f}</sup>",
                "x": 0.5,
                "xanchor": "center",
            },
            width=1152,
            height=864,
            template="plotly_white",
            margin={"l": 60, "r": 210, "t": 90, "b": 60},
            legend={"x": 1.02, "y": 1.0, "xanchor": "left", "yanchor": "top", "bgcolor": "rgba(255,255,255,0.85)"},
            xaxis={"title": "Dimension 1", "range": x_range, "showgrid": True, "zeroline": False},
            yaxis={"title": "Dimension 2", "range": y_range, "showgrid": True, "zeroline": False},
        )
        png = fig.to_image(format="png", scale=1)
        frame_images.append(Image.open(BytesIO(png)).convert("P", palette=Image.Palette.ADAPTIVE))

    if not frame_images:
        raise ValueError("No GIF frames were rendered.")

    duration_ms = int(round(1000 / max(1, int(fps))))
    frame_images[0].save(
        output_file,
        save_all=True,
        append_images=frame_images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def export_interactive_umap(
    *,
    project_root: Path = PROJECT_ROOT,
    mix_slugs: list[str] | None = None,
    output_file: Path | None = None,
    maest_dir: Path | None = None,
    chroma_dir: Path | None = None,
    tempo_dir: Path | None = None,
    random_state: int = 7777,
    n_neighbors: int = 10,
    min_dist: float = 0.1,
    default_maest_weight: float = 1.0,
    align_layouts: bool = True,
    temperature: float = 0.08,
    tempo_bandwidth: float = 0.06,
    tempo_decay: float = 0.5,
    tempo_allow_octave: bool = True,
    tempo_octave_penalty: float = 0.5,
    tempo_similarity_shape: str = "gaussian",
    tempo_softflat_sharpness: float = 8.0,
    tempo_use_confidence: bool = False,
    snippet_seconds: float = 8.0,
    snippet_middle_fraction: float = 0.66,
    snippet_hop_seconds: float = 0.25,
    snippet_cache_overwrite: bool = False,
    top_k_rows: int = 25,
    background_links_per_song: int = 3,
    click_links_per_song: int = 5,
    bpm_color_scale_pct: float = 0.10,
    gif_output_file: Path | None = None,
    gif_renderer: str = "matplotlib",
    gif_fps: int = 8,
    gif_tween_frames: int = 6,
    gif_hold_frames: int = 4,
    open_browser: bool = False,
) -> Path:
    project_root = project_root.expanduser().resolve()
    mix_slugs = mix_slugs or ["aries-mix", "ara-mix"]
    maest_dir = (maest_dir or project_root / "data" / "maest_embeddings").expanduser().resolve()
    chroma_dir = (chroma_dir or project_root / "data" / "chroma_embeddings").expanduser().resolve()
    tempo_dir = (tempo_dir or project_root / "data" / "tempo_embeddings").expanduser().resolve()
    dataset_tag = "__".join(
        str((project_root / "music" / slug / f"{slug.replace('-', '_')}_tracks.csv").stem)
        for slug in mix_slugs
    )
    output_file = output_file or (
        project_root / "data" / "exports" / f"{dataset_tag}_interactive_umap_precomputed_maest_tempo.html"
    )
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records, features = _load_combined_records_and_features(
        project_root=project_root,
        mix_slugs=mix_slugs,
        maest_dir=maest_dir,
        chroma_dir=chroma_dir,
        tempo_dir=tempo_dir,
        html_output_dir=output_file.parent,
        snippet_seconds=snippet_seconds,
        snippet_middle_fraction=snippet_middle_fraction,
        snippet_hop_seconds=snippet_hop_seconds,
        snippet_cache_overwrite=snippet_cache_overwrite,
    )

    matrices = _component_matrices(
        features,
        tempo_bandwidth=tempo_bandwidth,
        tempo_decay=tempo_decay,
        tempo_allow_octave=tempo_allow_octave,
        tempo_octave_penalty=tempo_octave_penalty,
        tempo_similarity_shape=tempo_similarity_shape,
        tempo_softflat_sharpness=tempo_softflat_sharpness,
        tempo_use_confidence=tempo_use_confidence,
    )
    weight_values = [round(i / 10.0, 1) for i in range(0, 11)]
    layouts = _compute_umap_layouts(
        D_maest=matrices["maest_distance"],
        D_tempo=matrices["tempo_distance"],
        weight_values=weight_values,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        align=align_layouts,
    )
    initial_key = f"{round(float(np.clip(default_maest_weight, 0.0, 1.0)) * 10.0) / 10.0:.1f}"
    initial_coords = np.asarray(layouts[initial_key], dtype=np.float32)
    similarity_payload = _build_similarity_payload(
        records=records,
        maest_similarity=matrices["maest_similarity"],
        tempo_similarity=matrices["tempo_similarity"],
        temperature=temperature,
    )

    mix_title = " + ".join(mix_slugs)
    title = f"{mix_title}: Interactive UMAP from Precomputed MAEST/Tempo Distances"
    plot_div_id = f"{dataset_tag}_umap_precomputed_maest_tempo".replace("-", "_")
    plot_html = _build_plot(records, initial_coords, plot_div_id=plot_div_id, title=title)
    html = _build_html(
        plot_html=plot_html,
        records=records,
        layouts=layouts,
        similarity_payload=similarity_payload,
        plot_div_id=plot_div_id,
        title=title,
        default_maest_weight=float(np.clip(default_maest_weight, 0.0, 1.0)),
        top_k_rows=top_k_rows,
        background_links_per_song=background_links_per_song,
        click_links_per_song=click_links_per_song,
        bpm_color_scale_pct=bpm_color_scale_pct,
        temperature=temperature,
    )
    output_file.write_text(html, encoding="utf-8")
    if gif_output_file is not None:
        _render_layout_gif(
            records=records,
            layouts=layouts,
            output_file=gif_output_file,
            title=title,
            renderer=gif_renderer,
            fps=gif_fps,
            tween_frames=gif_tween_frames,
            hold_frames=gif_hold_frames,
        )

    print(f"Loaded aligned tracks: {len(records)}")
    print(f"Computed UMAP layouts: {len(layouts)} weight splits ({', '.join(layouts.keys())})")
    print(f"Standalone HTML saved to: {output_file}")
    if gif_output_file is not None:
        print(f"Animated GIF saved to: {gif_output_file.expanduser().resolve()}")
    if open_browser:
        webbrowser.open_new_tab(output_file.as_uri())
    return output_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive UMAP HTML from precomputed MAEST/tempo weighted distances."
    )
    parser.add_argument(
        "--mix-slug",
        action="append",
        dest="mix_slugs",
        help="Mix slug to include, e.g. aries-mix. May be passed multiple times. Defaults to aries-mix + ara-mix.",
    )
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=7777)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--default-maest-weight", type=float, default=1.0)
    parser.add_argument("--no-align-layouts", action="store_true")
    parser.add_argument("--gif-output", type=Path, default=None, help="Optional path for an animated GIF export.")
    parser.add_argument("--gif-renderer", choices=["matplotlib", "plotly"], default="matplotlib")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--gif-tween-frames", type=int, default=6)
    parser.add_argument("--gif-hold-frames", type=int, default=4)
    parser.add_argument("--open", action="store_true", help="Open the exported HTML in a browser.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_interactive_umap(
        mix_slugs=args.mix_slugs,
        output_file=args.output_file,
        random_state=args.random_state,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        default_maest_weight=args.default_maest_weight,
        align_layouts=not args.no_align_layouts,
        gif_output_file=args.gif_output,
        gif_renderer=args.gif_renderer,
        gif_fps=args.gif_fps,
        gif_tween_frames=args.gif_tween_frames,
        gif_hold_frames=args.gif_hold_frames,
        open_browser=bool(args.open),
    )


if __name__ == "__main__":
    main()
