"""Export an interactive PaCMAP energy sequence builder as standalone HTML."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from djprojectexploration.audio_snippets import ensure_cached_snippet
from djprojectexploration.multimodal_compatibility import (
    SongFeatureSet,
    SongMetadata,
    compatible_song_distribution,
    load_aries_mix_feature_set,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _norm_token(value: str) -> str:
    return Path(str(value).strip()).name.lower()


def _norm_mix(value: str) -> str:
    text = str(value).strip().lower()
    if text.endswith("-mix"):
        text = text[: -len("-mix")]
    return text


def _json_script_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _row_token(row: dict[str, str]) -> str | None:
    for key in ("mp3_name", "filename", "filepath", "location"):
        value = (row.get(key) or "").strip()
        if value:
            return _norm_token(value)
    return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def _load_energy_lookup(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Energy feature NPZ not found: {path}")

    npz = np.load(path, allow_pickle=False)
    required = {"mix_name", "mp3_name", "energy"}
    missing = required - set(npz.files)
    if missing:
        raise KeyError(f"Energy NPZ missing required arrays: {sorted(missing)}")

    mix_names = [str(v) for v in npz["mix_name"]]
    filenames = [str(v) for v in npz["mp3_name"]]
    human_energy = np.asarray(npz["energy"], dtype=np.float32)
    glm_energy = (
        np.asarray(npz["glm_energy_pred"], dtype=np.float32)
        if "glm_energy_pred" in npz.files
        else np.full_like(human_energy, np.nan, dtype=np.float32)
    )

    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for mix, filename, human, glm in zip(mix_names, filenames, human_energy, glm_energy, strict=False):
        lookup[(_norm_mix(mix), _norm_token(filename))] = {
            "human_energy": float(human) if np.isfinite(float(human)) else float("nan"),
            "glm_energy": float(glm) if np.isfinite(float(glm)) else float("nan"),
        }
    return lookup


def _load_combined_records_and_features(
    *,
    project_root: Path,
    mix_slugs: list[str],
    energy_npz_path: Path,
    maest_dir: Path,
    chroma_dir: Path,
    tempo_dir: Path,
    html_output_dir: Path,
) -> tuple[list[dict[str, Any]], SongFeatureSet]:
    energy_lookup = _load_energy_lookup(energy_npz_path)

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
            energy_values = energy_lookup.get((_norm_mix(mix_slug), token), {})

            title = (row.get("title") or meta.title or Path(meta.filename).stem).strip()
            artists = (row.get("artists") or meta.artist or "").strip()
            key_tag = (row.get("key") or "").strip()
            bpm_tag = (row.get("bpm") or "").strip()
            genre = (row.get("genre") or meta.genre or "Unknown").strip() or "Unknown"
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
                    snippet_seconds=8.0,
                    middle_fraction=0.66,
                    hop_seconds=0.25,
                    overwrite=False,
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
                    "key": key_tag,
                    "csv_bpm": bpm_tag,
                    "est_bpm": est_bpm,
                    "est_conf": est_conf,
                    "mix_slug": mix_slug,
                    "audio_path": "" if audio_path is None else str(audio_path),
                    "snippet_uri": snippet_uri,
                    "snippet_path": snippet_path,
                    "snippet_start": snippet_start,
                    "snippet_end": snippet_end,
                    "snippet_rms": snippet_rms,
                    "human_energy": float(energy_values.get("human_energy", 5.0)),
                    "glm_energy": float(energy_values.get("glm_energy", np.nan)),
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


def _build_similarity_payload(
    *,
    records: list[dict[str, Any]],
    features: SongFeatureSet,
    top_k: int | None,
    maest_weight: float,
    chroma_weight: float,
    tempo_weight: float,
    temperature: float,
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
) -> dict[str, dict[str, Any]]:
    global_track_number_to_idx = {
        int(meta.track_number): i
        for i, meta in enumerate(features.metadata)
    }
    payload: dict[str, dict[str, Any]] = {}

    for src_idx, meta in enumerate(features.metadata):
        dist_rows = compatible_song_distribution(
            features=features,
            seed_track_number=int(meta.track_number),
            maest_weight=maest_weight,
            chroma_weight=chroma_weight,
            harmonic_exact_weight=harmonic_exact_weight,
            harmonic_first_fifth_weight=harmonic_first_fifth_weight,
            harmonic_second_fifth_weight=harmonic_second_fifth_weight,
            harmonic_other_weight=harmonic_other_weight,
            harmonic_self_normalize=harmonic_self_normalize,
            tempo_weight=tempo_weight,
            temperature=temperature,
            candidate_top_k=top_k,
            tempo_bandwidth=tempo_bandwidth,
            tempo_decay=tempo_decay,
            tempo_allow_octave=tempo_allow_octave,
            tempo_octave_penalty=tempo_octave_penalty,
            tempo_similarity_shape=tempo_similarity_shape,
            tempo_softflat_sharpness=tempo_softflat_sharpness,
            tempo_use_confidence=tempo_use_confidence,
        )

        ranked_rows: list[dict[str, Any]] = []
        src_est_bpm = float(records[src_idx]["est_bpm"])
        for initial_rank, row in enumerate(dist_rows, start=1):
            cand_global_track_number = int(row.get("track_number", -1))
            cand_idx = global_track_number_to_idx.get(cand_global_track_number)
            if cand_idx is None:
                continue

            cand_record = records[cand_idx]
            cand_est_bpm = float(cand_record["est_bpm"])
            bpm_delta_frac = 0.0
            if src_est_bpm > 0.0 and cand_est_bpm > 0.0:
                bpm_delta_frac = float((cand_est_bpm - src_est_bpm) / src_est_bpm)

            ranked_rows.append(
                {
                    "initial_rank": initial_rank,
                    "idx": int(cand_idx),
                    "global_track_number": cand_global_track_number,
                    "track_number": str(cand_record["track_number"]),
                    "mix_slug": str(cand_record["mix_slug"]),
                    "title": str(cand_record["title"]),
                    "artists": str(cand_record["artists"]),
                    "genre": str(cand_record["genre"]),
                    "key": str(cand_record["key"]),
                    "csv_bpm": str(cand_record["csv_bpm"]),
                    "est_bpm": cand_est_bpm,
                    "est_conf": float(cand_record["est_conf"]),
                    "filename": str(cand_record["filename"]),
                    "compatibility_score": float(row.get("compatibility_logit", 0.0)),
                    "maest_similarity": float(row.get("maest_similarity", 0.0)),
                    "chroma_similarity": float(row.get("chroma_similarity", 0.0)),
                    "tempo_similarity": float(row.get("tempo_similarity", 0.0)),
                    "probability": float(row.get("probability", 0.0)),
                    "bpm_delta_frac": bpm_delta_frac,
                    "is_self": bool(cand_idx == src_idx),
                }
            )

        payload[str(src_idx)] = {"candidates": ranked_rows}
    return payload


def _compute_pacmap_coords(features: SongFeatureSet, *, random_state: int, n_neighbors: int) -> np.ndarray:
    try:
        import pacmap
    except ImportError as exc:
        raise ImportError("PaCMAP is not installed. Install with: pip install pacmap") from exc

    maest = np.asarray(features.maest, dtype=np.float32)
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=min(int(n_neighbors), max(2, maest.shape[0] - 1)),
        MN_ratio=0.5,
        FP_ratio=1.5,
        random_state=int(random_state),
    )
    return np.asarray(reducer.fit_transform(maest), dtype=np.float32)


def _build_plot(records: list[dict[str, Any]], coords: np.ndarray, *, plot_div_id: str, title: str) -> str:
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
                p["idx"], p["title"], p["artists"], p["genre"], p["mix_slug"], p["key"],
                p["csv_bpm"], p["est_bpm"], p["track_number"], p["filename"],
                p["human_energy"], p["glm_energy"],
            ]
            for p in pts
        ]
        fig.add_scatter(
            x=coords[idxs, 0],
            y=coords[idxs, 1],
            mode="markers",
            name=genre,
            marker={"size": 9, "color": genre_color[genre], "line": {"color": "black", "width": 0.5}},
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Artists: %{customdata[2]}<br>"
                "Genre: %{customdata[3]}<br>"
                "Mix: %{customdata[4]}<br>"
                "Key: %{customdata[5]}<br>"
                "CSV BPM: %{customdata[6]}<br>"
                "Estimated BPM: %{customdata[7]:.2f}<br>"
                "Track #: %{customdata[8]}<br>"
                "Human energy: %{customdata[10]:.2f}<br>"
                "GLM energy: %{customdata[11]:.2f}<br>"
                "File: %{customdata[9]}<extra></extra>"
            ),
        )

    fig.add_scatter(
        x=[],
        y=[],
        mode="markers",
        name="Selected",
        marker={"size": 18, "color": "rgba(255,255,255,0)", "line": {"color": "#111", "width": 3}},
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_scatter(
        x=[],
        y=[],
        mode="lines+markers+text",
        name="Sequence path",
        line={"color": "#0c6b58", "width": 3},
        marker={"size": 13, "color": "#0c6b58", "line": {"color": "white", "width": 1.5}},
        text=[],
        textposition="top center",
        textfont={"color": "#0c3f35", "size": 12},
        hoverinfo="skip",
        showlegend=True,
    )
    fig.add_scatter(
        x=[],
        y=[],
        mode="lines",
        name="Recommended next links",
        line={"color": "rgba(214,95,39,0.45)", "width": 2},
        hoverinfo="skip",
        showlegend=True,
    )
    fig.add_scatter(
        x=[],
        y=[],
        mode="markers+text",
        name="Recommended next",
        marker={"size": 16, "color": "rgba(214,95,39,0.22)", "line": {"color": "#d65f27", "width": 2}},
        text=[],
        textposition="bottom center",
        textfont={"color": "#9b3e18", "size": 11},
        customdata=[],
        hovertemplate="<b>Recommended %{text}</b><extra></extra>",
        showlegend=True,
    )

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        xaxis_title="PaCMAP-1 (MAEST)",
        yaxis_title="PaCMAP-2 (MAEST)",
        width=1220,
        height=650,
        margin={"t": 70, "r": 30, "b": 50, "l": 55},
        legend_title_text="Genre",
        template="plotly_white",
    )
    return fig.to_html(include_plotlyjs=True, full_html=False, div_id=plot_div_id)


def _build_html(
    *,
    plot_html: str,
    records: list[dict[str, Any]],
    coords: np.ndarray,
    similarity_payload: dict[str, dict[str, Any]],
    plot_div_id: str,
    title: str,
    default_length: int,
    default_weights: dict[str, float],
    temperature: float,
    top_k_rows: int,
) -> str:
    idx_to_point = {
        str(i): [float(coords[i, 0]), float(coords[i, 1])]
        for i in range(coords.shape[0])
    }
    record_payload = [
        {
            "idx": int(r["idx"]),
            "global_track_number": int(r["global_track_number"]),
            "track_number": str(r["track_number"]),
            "filename": str(r["filename"]),
            "title": str(r["title"]),
            "artists": str(r["artists"]),
            "genre": str(r["genre"]),
            "key": str(r["key"]),
            "csv_bpm": str(r["csv_bpm"]),
            "est_bpm": float(r["est_bpm"]),
            "est_conf": float(r["est_conf"]),
            "mix_slug": str(r["mix_slug"]),
            "audio_path": str(r["audio_path"]),
            "snippet_uri": str(r["snippet_uri"]),
            "snippet_path": str(r["snippet_path"]),
            "snippet_start": float(r["snippet_start"]),
            "snippet_end": float(r["snippet_end"]),
            "snippet_rms": float(r["snippet_rms"]),
            "human_energy": float(r["human_energy"]) if np.isfinite(float(r["human_energy"])) else None,
            "glm_energy": float(r["glm_energy"]) if np.isfinite(float(r["glm_energy"])) else None,
        }
        for r in records
    ]

    data_scripts = "\n".join(
        [
            f'<script id="seq-records-json" type="application/json">{_json_script_payload(record_payload)}</script>',
            f'<script id="seq-sim-json" type="application/json">{_json_script_payload(similarity_payload)}</script>',
            f'<script id="seq-points-json" type="application/json">{_json_script_payload(idx_to_point)}</script>',
            f'<script id="seq-config-json" type="application/json">{_json_script_payload({"default_length": default_length, "weights": default_weights, "temperature": temperature, "top_k_rows": top_k_rows})}</script>',
        ]
    )

    controls_html = """
<section class="builder">
  <div class="control-grid">
    <label>Sequence length <input id="sequence-length" type="number" min="2" max="40" step="1"></label>
    <label>Energy source
      <select id="energy-source">
        <option value="human">Human tag</option>
        <option value="glm">GLM prediction</option>
      </select>
    </label>
    <label>Point color
      <select id="color-mode">
        <option value="genre">Genre</option>
        <option value="energy">Energy</option>
        <option value="tempo">Tempo</option>
      </select>
    </label>
    <label>Energy penalty <input id="energy-penalty-scale" type="range" min="0" max="10" step="0.05" value="2"></label>
    <output id="energy-penalty-scale-val">2.00</output>
    <label>MAEST <input id="weight-maest" type="range" min="0" max="1" step="0.01"></label>
    <output id="weight-maest-val">0.000</output>
    <label>Chroma <input id="weight-chroma" type="range" min="0" max="1" step="0.01"></label>
    <output id="weight-chroma-val">0.000</output>
    <label>BPM <input id="weight-tempo" type="range" min="0" max="1" step="0.01"></label>
    <output id="weight-tempo-val">0.000</output>
  </div>
  <div class="selected-panel">
    <div>
      <div id="selected-track">Click a PaCMAP point to select a current track.</div>
      <audio id="track-audio" controls></audio>
    </div>
    <div class="actions">
      <button id="append-selected">Append selected</button>
      <button id="clear-last">Clear last</button>
      <button id="reset-sequence">Reset sequence</button>
      <button id="download-sequence">Download CSV</button>
    </div>
  </div>
  <div id="target-inputs" class="target-inputs"></div>
  <div id="energy-curve" class="energy-curve"></div>
  <div>
    <h2>Transition Diagnostics</h2>
    <div id="transition-diagnostics" class="transition-diagnostics"></div>
  </div>
  <div class="lower-grid">
    <div>
      <h2>Sequence</h2>
      <div id="sequence-list" class="sequence-list"></div>
    </div>
    <div>
      <h2>Recommendations</h2>
      <div id="recommendation-panel" class="recommendation-panel">Select a current track to score candidates.</div>
    </div>
  </div>
</section>
"""

    style = """
<style>
  :root { color-scheme: light; --line:#d9dee8; --ink:#17202a; --muted:#5b6678; --accent:#0c6b58; --warn:#a64219; }
  body { font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 18px; color: var(--ink); background: #f6f8fb; }
  h1 { font-size: 24px; margin: 0 0 14px; }
  h2 { font-size: 15px; margin: 0 0 8px; }
  .builder { margin-top: 14px; display: grid; gap: 12px; }
  .control-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; align-items: center; padding: 10px; border: 1px solid var(--line); background: #fff; }
  label { display: grid; gap: 4px; font-size: 12px; color: var(--muted); }
  input, select, button { font: inherit; }
  input[type="number"], select { padding: 6px 7px; border: 1px solid var(--line); border-radius: 4px; background: #fff; }
  output { font-size: 12px; text-align: right; color: var(--ink); }
  button { border: 1px solid #b9c3d3; background: #fff; border-radius: 4px; padding: 7px 10px; cursor: pointer; }
  button:hover { border-color: #7e8da3; }
  button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  button:disabled { opacity: .45; cursor: not-allowed; }
  audio { width: 100%; margin-top: 8px; display: block; }
  .selected-panel { display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: center; padding: 10px; border: 1px solid var(--line); background: #fff; }
  .actions { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
  .target-inputs { display: grid; grid-template-columns: repeat(auto-fit, minmax(88px, 1fr)); gap: 7px; padding: 10px; border: 1px solid var(--line); background: #fff; }
  .target-inputs label { min-width: 0; }
  .energy-curve { height: 320px; border: 1px solid var(--line); background: #fff; }
  .lower-grid { display: grid; grid-template-columns: minmax(360px, .9fr) minmax(520px, 1.4fr); gap: 12px; align-items: start; }
  .sequence-list, .recommendation-panel, .transition-diagnostics { border: 1px solid var(--line); background: #fff; max-height: 560px; overflow: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { border-bottom: 1px solid #edf0f5; padding: 6px 7px; vertical-align: top; }
  th { position: sticky; top: 0; background: #f9fafc; z-index: 1; text-align: left; color: #3c4758; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  tr.selected-slot { background: #eef8f5; }
  .muted { color: var(--muted); }
  .warn { color: var(--warn); }
  .pill { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 1px 6px; color: var(--muted); font-size: 11px; }
  @media (max-width: 1100px) {
    .control-grid, .selected-panel, .lower-grid { grid-template-columns: 1fr; }
    .actions { justify-content: flex-start; }
  }
</style>
"""

    script = r"""
<script>
(function() {
  const records = JSON.parse(document.getElementById('seq-records-json').textContent);
  const simMap = JSON.parse(document.getElementById('seq-sim-json').textContent);
  const idxToPoint = JSON.parse(document.getElementById('seq-points-json').textContent);
  const config = JSON.parse(document.getElementById('seq-config-json').textContent);
  const plot = document.getElementById('__PLOT_ID__');

  const byIdx = new Map(records.map(r => [Number(r.idx), r]));
  let selectedIdx = null;
  let selectedSlot = null;
  let sequenceLength = Number(config.default_length || 10);
  let sequence = Array.from({ length: sequenceLength }, () => null);
  let draggingEnergySlot = null;

  const els = {
    sequenceLength: document.getElementById('sequence-length'),
    energySource: document.getElementById('energy-source'),
    colorMode: document.getElementById('color-mode'),
    penaltyScale: document.getElementById('energy-penalty-scale'),
    penaltyScaleVal: document.getElementById('energy-penalty-scale-val'),
    weightMaest: document.getElementById('weight-maest'),
    weightChroma: document.getElementById('weight-chroma'),
    weightTempo: document.getElementById('weight-tempo'),
    weightMaestVal: document.getElementById('weight-maest-val'),
    weightChromaVal: document.getElementById('weight-chroma-val'),
    weightTempoVal: document.getElementById('weight-tempo-val'),
    selectedTrack: document.getElementById('selected-track'),
    audio: document.getElementById('track-audio'),
    appendSelected: document.getElementById('append-selected'),
    clearLast: document.getElementById('clear-last'),
    resetSequence: document.getElementById('reset-sequence'),
    downloadSequence: document.getElementById('download-sequence'),
    targetInputs: document.getElementById('target-inputs'),
    energyCurve: document.getElementById('energy-curve'),
    sequenceList: document.getElementById('sequence-list'),
    recommendationPanel: document.getElementById('recommendation-panel'),
    transitionDiagnostics: document.getElementById('transition-diagnostics'),
  };

  const baseTraceIndices = [];
  const originalBaseMarkers = new Map();
  if (plot && Array.isArray(plot.data)) {
    for (let i = 0; i < plot.data.length; i += 1) {
      const trace = plot.data[i] || {};
      const custom = Array.isArray(trace.customdata) ? trace.customdata : [];
      const isBaseTrackTrace = custom.length > 0 && Array.isArray(custom[0]) && custom[0].length >= 12;
      if (isBaseTrackTrace) {
        baseTraceIndices.push(i);
        originalBaseMarkers.set(i, JSON.parse(JSON.stringify(trace.marker || {})));
      }
    }
  }

  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
  function finiteOr(v, fallback) { const n = Number(v); return Number.isFinite(n) ? n : fallback; }
  function fmt(v, d=3) { return Number.isFinite(Number(v)) ? Number(v).toFixed(d) : ''; }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  }
  function traceIndexByName(name) {
    if (!plot || !plot.data) return -1;
    for (let i = 0; i < plot.data.length; i += 1) {
      if (plot.data[i] && plot.data[i].name === name) return i;
    }
    return -1;
  }
  function restyleTrace(name, update) {
    if (!window.Plotly || !plot) return;
    const idx = traceIndexByName(name);
    if (idx < 0) return;
    Plotly.restyle(plot, update, [idx]);
  }
  function colorScaleExtent(mode) {
    if (mode === 'energy') return { min: 1, max: 9, title: els.energySource.value === 'glm' ? 'GLM energy' : 'Human energy' };
    const vals = records.map(r => Number(r.est_bpm)).filter(Number.isFinite);
    if (!vals.length) return { min: 0, max: 1, title: 'Tempo BPM' };
    let min = Math.min(...vals);
    let max = Math.max(...vals);
    if (Math.abs(max - min) < 1e-9) max = min + 1;
    return { min, max, title: 'Estimated BPM' };
  }
  function pointColorValue(idx, mode) {
    const record = byIdx.get(Number(idx));
    if (!record) return NaN;
    if (mode === 'energy') return energyOf(record);
    if (mode === 'tempo') return Number(record.est_bpm);
    return NaN;
  }
  function applyPointColorMode() {
    if (!window.Plotly || !plot || !baseTraceIndices.length) return;
    const mode = els.colorMode ? els.colorMode.value : 'genre';

    if (mode === 'genre') {
      for (const traceIdx of baseTraceIndices) {
        const marker = JSON.parse(JSON.stringify(originalBaseMarkers.get(traceIdx) || {}));
        Plotly.restyle(plot, { marker: [marker], showlegend: [true] }, [traceIdx]);
      }
      Plotly.relayout(plot, { 'legend.title.text': 'Genre' });
      return;
    }

    const extent = colorScaleExtent(mode);
    baseTraceIndices.forEach((traceIdx, traceOrder) => {
      const trace = plot.data[traceIdx] || {};
      const custom = Array.isArray(trace.customdata) ? trace.customdata : [];
      const values = custom.map(row => pointColorValue(Array.isArray(row) ? row[0] : NaN, mode));
      const marker = {
        size: 9,
        color: values,
        colorscale: mode === 'energy' ? 'Viridis' : 'Turbo',
        cmin: extent.min,
        cmax: extent.max,
        showscale: traceOrder === 0,
        colorbar: { title: { text: extent.title }, thickness: 14 },
        line: { color: 'black', width: 0.5 },
      };
      Plotly.restyle(plot, { marker: [marker], showlegend: [false] }, [traceIdx]);
    });
    Plotly.relayout(plot, { 'legend.title.text': mode === 'energy' ? 'Energy' : 'Tempo' });
  }
  function weights() {
    const ma = Math.max(0, Number(els.weightMaest.value || 0));
    const ch = Math.max(0, Number(els.weightChroma.value || 0));
    const te = Math.max(0, Number(els.weightTempo.value || 0));
    const s = ma + ch + te;
    if (s <= 1e-12) return { maest: 1/3, chroma: 1/3, tempo: 1/3 };
    return { maest: ma/s, chroma: ch/s, tempo: te/s };
  }
  function updateWeightLabels() {
    const w = weights();
    els.weightMaestVal.textContent = fmt(w.maest, 3);
    els.weightChromaVal.textContent = fmt(w.chroma, 3);
    els.weightTempoVal.textContent = fmt(w.tempo, 3);
    els.penaltyScaleVal.textContent = fmt(Number(els.penaltyScale.value || 0), 2);
  }
  function energyOf(record) {
    if (!record) return NaN;
    const source = els.energySource.value;
    const primary = source === 'glm' ? record.glm_energy : record.human_energy;
    const secondary = source === 'glm' ? record.human_energy : record.glm_energy;
    const p = Number(primary);
    if (Number.isFinite(p)) return p;
    const s = Number(secondary);
    if (Number.isFinite(s)) return s;
    return 5;
  }
  function selectedIndices() {
    return new Set(sequence.filter(v => v !== null).map(Number));
  }
  function selectedIndicesExcept(slot) {
    const used = new Set();
    for (let i = 0; i < sequence.length; i += 1) {
      if (i === slot) continue;
      if (sequence[i] !== null) used.add(Number(sequence[i]));
    }
    return used;
  }
  function nextEmptySlot() {
    if (selectedSlot !== null && selectedSlot >= 0 && selectedSlot < sequence.length && sequence[selectedSlot] === null) return selectedSlot;
    const i = sequence.findIndex(v => v === null);
    return i >= 0 ? i : -1;
  }
  function targetSlot() {
    if (selectedSlot !== null && selectedSlot >= 0 && selectedSlot < sequence.length) return selectedSlot;
    return nextEmptySlot();
  }
  function canPlaceTrack(idx, slot) {
    idx = Number(idx);
    if (!Number.isFinite(idx) || !byIdx.has(idx) || slot < 0) return false;
    return !selectedIndicesExcept(slot).has(idx);
  }
  function setSequenceLength(n) {
    n = clamp(Math.round(Number(n || 10)), 2, 40);
    const old = sequence.slice();
    sequenceLength = n;
    sequence = Array.from({ length: n }, (_, i) => i < old.length ? old[i] : null);
    selectedSlot = selectedSlot !== null && selectedSlot < n ? selectedSlot : null;
    renderTargetInputs();
    renderAll();
  }
  function renderTargetInputs() {
    const oldValues = Array.from(els.targetInputs.querySelectorAll('input')).map(i => i.value);
    els.targetInputs.innerHTML = '';
    for (let i = 0; i < sequenceLength; i += 1) {
      const label = document.createElement('label');
      label.innerHTML = 'Slot ' + (i + 1) + '<input class="target-energy" data-slot="' + i + '" type="number" min="1" max="9" step="0.1" placeholder="auto">';
      const input = label.querySelector('input');
      if (oldValues[i] !== undefined) input.value = oldValues[i];
      input.addEventListener('input', renderAll);
      els.targetInputs.appendChild(label);
    }
  }
  function setTargetSlotValue(slot, value, shouldRender=true) {
    slot = Math.round(Number(slot));
    value = clamp(Number(value), 1, 9);
    if (!Number.isFinite(slot) || !Number.isFinite(value)) return;
    const input = els.targetInputs.querySelector('input[data-slot="' + slot + '"]');
    if (!input) return;
    input.value = value.toFixed(2);
    selectedSlot = slot;
    if (shouldRender) renderAll();
  }
  function targetAnchors() {
    const inputs = Array.from(els.targetInputs.querySelectorAll('input'));
    const anchors = [];
    for (const input of inputs) {
      const slot = Number(input.dataset.slot);
      const val = Number(input.value);
      if (Number.isFinite(val)) anchors.push({ slot, value: clamp(val, 1, 9) });
    }
    return anchors.sort((a,b) => a.slot - b.slot);
  }
  function targetCurve() {
    const anchors = targetAnchors();
    const out = [];
    if (!anchors.length) {
      for (let i = 0; i < sequenceLength; i += 1) {
        out.push(sequenceLength === 1 ? 5 : 3 + (4 * i / (sequenceLength - 1)));
      }
      return out;
    }
    if (anchors.length === 1) return Array.from({ length: sequenceLength }, () => anchors[0].value);
    for (let i = 0; i < sequenceLength; i += 1) {
      if (i <= anchors[0].slot) { out.push(anchors[0].value); continue; }
      if (i >= anchors[anchors.length - 1].slot) { out.push(anchors[anchors.length - 1].value); continue; }
      let left = anchors[0], right = anchors[anchors.length - 1];
      for (let j = 0; j < anchors.length - 1; j += 1) {
        if (anchors[j].slot <= i && i <= anchors[j+1].slot) { left = anchors[j]; right = anchors[j+1]; break; }
      }
      const t = (i - left.slot) / Math.max(1, right.slot - left.slot);
      out.push(left.value + t * (right.value - left.value));
    }
    return out;
  }
  function scoreCandidate(c, slot) {
    const w = weights();
    const baseline = w.maest * Number(c.maest_similarity || 0) + w.chroma * Number(c.chroma_similarity || 0) + w.tempo * Number(c.tempo_similarity || 0);
    const record = byIdx.get(Number(c.idx));
    const e = energyOf(record);
    const target = targetCurve()[slot] ?? 5;
    const rawError = Number.isFinite(e) ? e - target : 0;
    const normSq = Math.pow(rawError / 8, 2);
    const penalty = Number(els.penaltyScale.value || 0) * normSq;
    const energyScore = 1 - normSq;
    return { baseline, energy: e, target, rawError, normSq, penalty, energyScore, finalScore: baseline - penalty };
  }
  function scoreTransition(sourceIdx, destIdx, slot) {
    const entry = simMap[String(sourceIdx)] || {};
    const candidates = Array.isArray(entry.candidates) ? entry.candidates : [];
    const found = candidates.find(c => Number(c.idx) === Number(destIdx));
    if (found) return { ...found, ...scoreCandidate(found, slot), missing: false };

    const record = byIdx.get(Number(destIdx));
    const e = energyOf(record);
    const target = targetCurve()[slot] ?? 5;
    const rawError = Number.isFinite(e) ? e - target : 0;
    const normSq = Math.pow(rawError / 8, 2);
    const penalty = Number(els.penaltyScale.value || 0) * normSq;
    const energyScore = 1 - normSq;
    return {
      idx: Number(destIdx),
      maest_similarity: NaN,
      chroma_similarity: NaN,
      tempo_similarity: NaN,
      baseline: NaN,
      energy: e,
      target,
      rawError,
      normSq,
      penalty,
      energyScore,
      finalScore: NaN,
      missing: true,
    };
  }
  function rankedRecommendations() {
    if (selectedIdx === null) return [];
    const slot = targetSlot();
    if (slot < 0) return [];
    const used = selectedIndicesExcept(slot);
    const entry = simMap[String(selectedIdx)] || {};
    const candidates = Array.isArray(entry.candidates) ? entry.candidates : [];
    const rows = [];
    for (const c of candidates) {
      const idx = Number(c.idx);
      if (!Number.isFinite(idx) || idx === selectedIdx || used.has(idx)) continue;
      const s = scoreCandidate(c, slot);
      rows.push({ ...c, ...s, slot });
    }
    rows.sort((a,b) => b.finalScore - a.finalScore);
    return rows;
  }
  function updatePacmapOverlays() {
    const selectedPt = selectedIdx === null ? null : idxToPoint[String(selectedIdx)];
    restyleTrace('Selected', {
      x: [selectedPt ? [selectedPt[0]] : []],
      y: [selectedPt ? [selectedPt[1]] : []],
    });

    const pathX = [];
    const pathY = [];
    const pathText = [];
    const pathCustom = [];
    for (let i = 0; i < sequence.length; i += 1) {
      const idx = sequence[i];
      if (idx === null) continue;
      const pt = idxToPoint[String(idx)];
      if (!pt) continue;
      pathX.push(pt[0]);
      pathY.push(pt[1]);
      pathText.push(String(i + 1));
      pathCustom.push([idx]);
    }
    restyleTrace('Sequence path', {
      x: [pathX],
      y: [pathY],
      text: [pathText],
      customdata: [pathCustom],
    });

    const rows = rankedRecommendations().slice(0, Math.min(10, Number(config.top_k_rows || 25)));
    const linkX = [];
    const linkY = [];
    const recX = [];
    const recY = [];
    const recText = [];
    const recCustom = [];
    const src = selectedIdx === null ? null : idxToPoint[String(selectedIdx)];
    if (src) {
      rows.forEach((row, i) => {
        const dst = idxToPoint[String(row.idx)];
        if (!dst) return;
        linkX.push(src[0], dst[0], null);
        linkY.push(src[1], dst[1], null);
        recX.push(dst[0]);
        recY.push(dst[1]);
        recText.push(String(i + 1));
        recCustom.push([Number(row.idx)]);
      });
    }
    restyleTrace('Recommended next links', { x: [linkX], y: [linkY] });
    restyleTrace('Recommended next', {
      x: [recX],
      y: [recY],
      text: [recText],
      customdata: [recCustom],
      hovertext: [rows.map((row, i) => '#' + (i + 1) + ' ' + row.title + '<br>Final: ' + fmt(row.finalScore, 4))],
      hovertemplate: ['%{hovertext}<extra></extra>'],
    });
  }
  function selectTrack(idx) {
    idx = Number(idx);
    if (!byIdx.has(idx)) return;
    selectedIdx = idx;
    const r = byIdx.get(idx);
    els.selectedTrack.innerHTML = '<b>' + esc(r.title) + '</b> <span class="pill">' + esc(r.mix_slug) + '</span><br>' +
      '<span class="muted">' + esc(r.artists) + ' | ' + esc(r.genre) + ' | key ' + esc(r.key) + ' | BPM ' + fmt(r.est_bpm, 2) + '</span><br>' +
      'Human energy: <b>' + fmt(r.human_energy, 2) + '</b> | GLM energy: <b>' + fmt(r.glm_energy, 2) + '</b><br>' +
      '<span class="muted">Snippet: ' + fmt(r.snippet_start, 2) + 's -> ' + fmt(r.snippet_end, 2) + 's | RMS ' + fmt(r.snippet_rms, 5) + '</span>';
    if (els.audio) {
      if (r.snippet_uri) {
        els.audio.src = r.snippet_uri;
        const playPromise = els.audio.play();
        if (playPromise && playPromise.catch) playPromise.catch(() => {});
      } else {
        els.audio.removeAttribute('src');
        els.audio.load();
      }
    }
    renderAll();
  }
  function appendTrack(idx) {
    const slot = targetSlot();
    if (slot < 0) return;
    idx = Number(idx);
    if (!canPlaceTrack(idx, slot)) return;
    sequence[slot] = idx;
    selectedSlot = null;
    selectTrack(idx);
  }
  function removeSlot(slot) {
    slot = Number(slot);
    if (slot >= 0 && slot < sequence.length) sequence[slot] = null;
    renderAll();
  }
  function renderSequence() {
    const targets = targetCurve();
    let html = '<table><thead><tr><th class="num">Slot</th><th>Track</th><th class="num">Target</th><th class="num">Actual</th><th></th></tr></thead><tbody>';
    for (let i = 0; i < sequence.length; i += 1) {
      const idx = sequence[i];
      const r = idx === null ? null : byIdx.get(idx);
      const cls = selectedSlot === i ? ' class="selected-slot"' : '';
      html += '<tr' + cls + '><td class="num">' + (i + 1) + '</td><td>';
      if (r) html += '<b>' + esc(r.title) + '</b><br><span class="muted">' + esc(r.artists) + ' | ' + esc(r.genre) + '</span>';
      else html += '<span class="muted">empty</span>';
      html += '</td><td class="num">' + fmt(targets[i], 2) + '</td><td class="num">' + (r ? fmt(energyOf(r), 2) : '') + '</td><td>';
      html += '<button data-seq-slot="' + i + '">' + (selectedSlot === i ? 'Selected' : 'Select slot') + '</button> ';
      if (selectedIdx !== null && canPlaceTrack(selectedIdx, i)) {
        html += '<button class="primary" data-place-slot="' + i + '">' + (r ? 'Replace' : 'Place') + '</button> ';
      }
      if (r) html += '<button data-remove-slot="' + i + '">Remove</button>';
      html += '</td></tr>';
    }
    html += '</tbody></table>';
    els.sequenceList.innerHTML = html;
  }
  function renderTransitionDiagnostics() {
    const filled = [];
    for (let i = 0; i < sequence.length; i += 1) {
      if (sequence[i] !== null) filled.push({ slot: i, idx: Number(sequence[i]) });
    }
    if (filled.length < 2) {
      els.transitionDiagnostics.innerHTML = '<div class="muted" style="padding:10px;">Add at least two tracks to inspect transition scores.</div>';
      return;
    }

    let html = '<table><thead><tr>' +
      '<th class="num">From</th><th class="num">To</th><th>Transition</th>' +
      '<th class="num">Final</th><th class="num">Base</th><th class="num">Target</th><th class="num">Energy</th>' +
      '<th class="num">Err</th><th class="num">Penalty</th><th class="num">MAEST</th><th class="num">Chroma</th><th class="num">BPM</th>' +
      '</tr></thead><tbody>';
    for (let i = 0; i < filled.length - 1; i += 1) {
      const from = filled[i];
      const to = filled[i + 1];
      const src = byIdx.get(from.idx);
      const dst = byIdx.get(to.idx);
      const score = scoreTransition(from.idx, to.idx, to.slot);
      html += '<tr>' +
        '<td class="num">' + (from.slot + 1) + '</td>' +
        '<td class="num">' + (to.slot + 1) + '</td>' +
        '<td><b>' + esc(src ? src.title : from.idx) + '</b> -> <b>' + esc(dst ? dst.title : to.idx) + '</b>' +
        (score.missing ? '<br><span class="warn">missing similarity row</span>' : '') + '</td>' +
        '<td class="num">' + fmt(score.finalScore, 4) + '</td>' +
        '<td class="num">' + fmt(score.baseline, 4) + '</td>' +
        '<td class="num">' + fmt(score.target, 2) + '</td>' +
        '<td class="num">' + fmt(score.energy, 2) + '</td>' +
        '<td class="num">' + fmt(score.rawError, 2) + '</td>' +
        '<td class="num">' + fmt(score.penalty, 4) + '</td>' +
        '<td class="num">' + fmt(score.maest_similarity, 4) + '</td>' +
        '<td class="num">' + fmt(score.chroma_similarity, 4) + '</td>' +
        '<td class="num">' + fmt(score.tempo_similarity, 4) + '</td>' +
        '</tr>';
    }
    html += '</tbody></table>';
    els.transitionDiagnostics.innerHTML = html;
  }
  function renderRecommendations() {
    if (selectedIdx === null) {
      els.recommendationPanel.textContent = 'Select a current track to score candidates.';
      return;
    }
    const slot = targetSlot();
    if (slot < 0) {
      els.recommendationPanel.innerHTML = '<div class="muted" style="padding:10px;">Sequence is full. Select a slot to replace a track.</div>';
      return;
    }
    const rows = rankedRecommendations().slice(0, Number(config.top_k_rows || 25));
    const actionLabel = sequence[slot] === null ? 'Append' : 'Replace';
    let html = '<div class="muted" style="padding:8px 8px 0;">Scoring candidates for slot <b>' + (slot + 1) + '</b> (' + actionLabel.toLowerCase() + ').</div>' +
      '<table><thead><tr><th class="num">#</th><th></th><th>Track</th><th class="num">Final</th><th class="num">Base</th><th class="num">Target</th><th class="num">Energy</th><th class="num">Err</th><th class="num">NormSq</th><th class="num">Penalty</th><th class="num">E score</th><th class="num">MAEST</th><th class="num">Chroma</th><th class="num">BPM</th></tr></thead><tbody>';
    rows.forEach((r, i) => {
      html += '<tr><td class="num">' + (i + 1) + '</td><td><button class="primary" data-append-idx="' + r.idx + '">' + actionLabel + '</button></td>' +
        '<td><b>' + esc(r.title) + '</b><br><span class="muted">' + esc(r.artists) + ' | ' + esc(r.genre) + ' | ' + esc(r.mix_slug) + '</span></td>' +
        '<td class="num">' + fmt(r.finalScore, 4) + '</td><td class="num">' + fmt(r.baseline, 4) + '</td>' +
        '<td class="num">' + fmt(r.target, 2) + '</td><td class="num">' + fmt(r.energy, 2) + '</td><td class="num">' + fmt(r.rawError, 2) + '</td>' +
        '<td class="num">' + fmt(r.normSq, 4) + '</td><td class="num">' + fmt(r.penalty, 4) + '</td><td class="num">' + fmt(r.energyScore, 4) + '</td>' +
        '<td class="num">' + fmt(r.maest_similarity, 4) + '</td><td class="num">' + fmt(r.chroma_similarity, 4) + '</td><td class="num">' + fmt(r.tempo_similarity, 4) + '</td></tr>';
    });
    html += '</tbody></table>';
    els.recommendationPanel.innerHTML = html;
  }
  function renderEnergyCurve() {
    const targets = targetCurve();
    const x = Array.from({ length: sequence.length }, (_, i) => i + 1);
    const actual = sequence.map(idx => idx === null ? null : energyOf(byIdx.get(idx)));
    const traces = [
      { x, y: targets, type: 'scatter', mode: 'lines+markers', name: 'Target energy', line: { color: '#111', width: 2 }, marker: { size: 10 } },
      { x, y: actual, type: 'scatter', mode: 'lines+markers', name: 'Selected actual energy', line: { color: '#0c6b58', width: 2 }, marker: { size: 9 } }
    ];
    const slot = targetSlot();
    if (slot >= 0) traces.push({ x: [slot + 1], y: [targets[slot]], type: 'scatter', mode: 'markers', name: 'Next slot', marker: { size: 14, color: '#d62728', symbol: 'x' } });
    Plotly.react('energy-curve', traces, {
      title: { text: 'Drag target points to edit the energy curve', font: { size: 13 } },
      margin: { t: 48, r: 20, b: 46, l: 48 },
      template: 'plotly_white',
      xaxis: { title: 'Sequence slot', dtick: 1, range: [0.5, sequence.length + 0.5] },
      yaxis: { title: 'Energy', range: [0.5, 9.5] },
      legend: { orientation: 'h' }
    }, { displayModeBar: false, responsive: true });
  }
  function energyValueFromMouse(ev) {
    const gd = els.energyCurve;
    if (!gd || !gd._fullLayout || !gd._fullLayout._size) return null;
    const rect = gd.getBoundingClientRect();
    const size = gd._fullLayout._size;
    const xPixel = ev.clientX - rect.left - size.l;
    const yPixel = ev.clientY - rect.top - size.t;
    if (xPixel < 0 || yPixel < 0 || xPixel > size.w || yPixel > size.h) return null;
    const xValue = gd._fullLayout.xaxis.p2d(xPixel);
    const yValue = gd._fullLayout.yaxis.p2d(yPixel);
    const slot = clamp(Math.round(Number(xValue)) - 1, 0, sequence.length - 1);
    const value = clamp(Number(yValue), 1, 9);
    if (!Number.isFinite(slot) || !Number.isFinite(value)) return null;
    return { slot, value };
  }
  function beginEnergyDrag(ev) {
    if (ev.button !== 0) return;
    const point = energyValueFromMouse(ev);
    if (!point) return;
    draggingEnergySlot = point.slot;
    setTargetSlotValue(point.slot, point.value, true);
    ev.preventDefault();
  }
  function updateEnergyDrag(ev) {
    if (draggingEnergySlot === null) return;
    const point = energyValueFromMouse(ev);
    if (!point) return;
    setTargetSlotValue(draggingEnergySlot, point.value, true);
  }
  function endEnergyDrag() {
    draggingEnergySlot = null;
  }
  function renderAll() {
    updateWeightLabels();
    const slot = targetSlot();
    const actionLabel = slot >= 0 && sequence[slot] !== null ? 'Replace selected slot' : 'Append selected';
    els.appendSelected.textContent = actionLabel;
    els.appendSelected.disabled = selectedIdx === null || !canPlaceTrack(selectedIdx, slot);
    renderSequence();
    renderTransitionDiagnostics();
    renderRecommendations();
    renderEnergyCurve();
    applyPointColorMode();
    updatePacmapOverlays();
  }
  function downloadCsv() {
    const targets = targetCurve();
    const header = ['slot','target_energy','actual_energy','mix','track_number','title','artists','genre','key','bpm','filename'];
    const lines = [header.join(',')];
    for (let i = 0; i < sequence.length; i += 1) {
      const r = sequence[i] === null ? null : byIdx.get(sequence[i]);
      const vals = [
        i + 1, fmt(targets[i], 3), r ? fmt(energyOf(r), 3) : '', r ? r.mix_slug : '', r ? r.track_number : '',
        r ? r.title : '', r ? r.artists : '', r ? r.genre : '', r ? r.key : '', r ? fmt(r.est_bpm, 3) : '', r ? r.filename : ''
      ];
      lines.push(vals.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(','));
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'energy_sequence.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  els.sequenceLength.value = String(sequenceLength);
  els.weightMaest.value = String((config.weights && config.weights.maest) || 0.6);
  els.weightChroma.value = String((config.weights && config.weights.chroma) || 0.25);
  els.weightTempo.value = String((config.weights && config.weights.tempo) || 0.15);
  renderTargetInputs();

  els.sequenceLength.addEventListener('change', () => setSequenceLength(els.sequenceLength.value));
  els.energySource.addEventListener('change', renderAll);
  els.colorMode.addEventListener('change', renderAll);
  els.penaltyScale.addEventListener('input', renderAll);
  els.weightMaest.addEventListener('input', renderAll);
  els.weightChroma.addEventListener('input', renderAll);
  els.weightTempo.addEventListener('input', renderAll);
  els.appendSelected.addEventListener('click', () => { if (selectedIdx !== null) appendTrack(selectedIdx); });
  els.clearLast.addEventListener('click', () => {
    for (let i = sequence.length - 1; i >= 0; i -= 1) { if (sequence[i] !== null) { sequence[i] = null; break; } }
    renderAll();
  });
  els.resetSequence.addEventListener('click', () => { sequence = Array.from({ length: sequenceLength }, () => null); selectedSlot = null; renderAll(); });
  els.downloadSequence.addEventListener('click', downloadCsv);

  document.body.addEventListener('click', ev => {
    const appendIdx = ev.target && ev.target.getAttribute && ev.target.getAttribute('data-append-idx');
    const removeSlotValue = ev.target && ev.target.getAttribute && ev.target.getAttribute('data-remove-slot');
    const seqSlotValue = ev.target && ev.target.getAttribute && ev.target.getAttribute('data-seq-slot');
    const placeSlotValue = ev.target && ev.target.getAttribute && ev.target.getAttribute('data-place-slot');
    if (appendIdx !== null) appendTrack(Number(appendIdx));
    if (removeSlotValue !== null) removeSlot(Number(removeSlotValue));
    if (seqSlotValue !== null) { selectedSlot = Number(seqSlotValue); renderAll(); }
    if (placeSlotValue !== null && selectedIdx !== null) { selectedSlot = Number(placeSlotValue); appendTrack(selectedIdx); }
  });

  if (els.energyCurve) {
    els.energyCurve.addEventListener('mousedown', beginEnergyDrag);
    document.addEventListener('mousemove', updateEnergyDrag);
    document.addEventListener('mouseup', endEnergyDrag);
  }

  if (plot && plot.on) {
    plot.on('plotly_click', ev => {
      if (!ev || !ev.points || !ev.points.length) return;
      const c = ev.points[0].customdata || [];
      const idx = Number(c[0]);
      if (Number.isFinite(idx)) selectTrack(idx);
    });
  }
  renderAll();
})();
</script>
""".replace("__PLOT_ID__", plot_div_id)

    return "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            f"<title>{title}</title>",
            style,
            "</head>",
            "<body>",
            f"<h1>{title}</h1>",
            plot_html,
            controls_html,
            data_scripts,
            script,
            "</body>",
            "</html>",
        ]
    )


def export_energy_sequence_builder(
    *,
    project_root: Path = PROJECT_ROOT,
    output_file: Path | None = None,
    mix_slugs: list[str] | None = None,
    energy_npz_path: Path | None = None,
    default_length: int = 10,
    open_browser: bool = False,
) -> Path:
    project_root = project_root.expanduser().resolve()
    mix_slugs = mix_slugs or ["aries-mix", "ara-mix"]
    output_file = output_file or (project_root / "data" / "exports" / "energy_sequence_builder.html")
    output_file = output_file.expanduser().resolve()
    energy_npz_path = energy_npz_path or (project_root / "data" / "energy_embeddings" / "aries_ara_energy_features.npz")

    records, features = _load_combined_records_and_features(
        project_root=project_root,
        mix_slugs=mix_slugs,
        energy_npz_path=energy_npz_path,
        maest_dir=project_root / "data" / "maest_embeddings",
        chroma_dir=project_root / "data" / "chroma_embeddings",
        tempo_dir=project_root / "data" / "tempo_embeddings",
        html_output_dir=output_file.parent,
    )
    coords = _compute_pacmap_coords(features, random_state=7777, n_neighbors=10)

    default_weights = {"maest": 0.60, "chroma": 0.25, "tempo": 0.15}
    similarity_payload = _build_similarity_payload(
        records=records,
        features=features,
        top_k=None,
        maest_weight=default_weights["maest"],
        chroma_weight=default_weights["chroma"],
        tempo_weight=default_weights["tempo"],
        temperature=0.08,
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

    title = "Energy Sequence Builder"
    plot_div_id = "energy_sequence_builder_pacmap"
    plot_html = _build_plot(records, coords, plot_div_id=plot_div_id, title="PaCMAP Track Selector")
    html = _build_html(
        plot_html=plot_html,
        records=records,
        coords=coords,
        similarity_payload=similarity_payload,
        plot_div_id=plot_div_id,
        title=title,
        default_length=default_length,
        default_weights=default_weights,
        temperature=0.08,
        top_k_rows=25,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")

    if open_browser:
        webbrowser.open_new_tab(output_file.as_uri())
    return output_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export an interactive energy sequence builder HTML view.")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--energy-npz", type=Path, default=None)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--mix", action="append", dest="mix_slugs", default=None, help="Mix slug to include; repeatable.")
    parser.add_argument("--open", action="store_true", help="Open the exported HTML in a browser.")
    args = parser.parse_args(argv)

    output_file = export_energy_sequence_builder(
        output_file=args.output_file,
        energy_npz_path=args.energy_npz,
        default_length=args.sequence_length,
        mix_slugs=args.mix_slugs,
        open_browser=bool(args.open),
    )
    print(f"Saved energy sequence builder HTML: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
