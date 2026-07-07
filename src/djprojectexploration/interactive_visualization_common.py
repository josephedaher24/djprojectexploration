"""Shared helpers for interactive embedding visualizations."""

from __future__ import annotations

import csv
import json
import os
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
    project_root: Path = PROJECT_ROOT,
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
        project_root / rel,
        music_dir / rel.name,
        project_root / "music" / rel.name,
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
                project_root=project_root,
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


def _neighbor_pairs_from_distance(D: np.ndarray, *, n_neighbors: int) -> np.ndarray:
    distances = np.asarray(D, dtype=np.float32)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError(f"Expected square distance matrix, got shape {distances.shape}.")

    n = distances.shape[0]
    k = min(max(1, int(n_neighbors)), n - 1)
    pairs = np.empty((n * k, 2), dtype=np.int32)

    row = 0
    for i in range(n):
        order = np.argsort(distances[i], kind="stable")
        order = order[order != i]
        for j in order[:k]:
            pairs[row, 0] = i
            pairs[row, 1] = int(j)
            row += 1

    return pairs


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
