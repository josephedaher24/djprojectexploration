"""Export an interactive PaCMAP energy sequence builder as standalone HTML."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from djprojectexploration.audio_snippets import ensure_cached_snippet
from djprojectexploration.interactive_pacmap_knn_simplex import (
    _build_similarity_payload_4way,
    _component_matrices_4way,
    _compute_pacmap_knn_4way_layouts,
    _load_combined_groove_embeddings,
    _simplex_grid_4way,
)
from djprojectexploration.interactive_visualization_common import simplify_genre
from djprojectexploration.multimodal_compatibility import (
    SongFeatureSet,
    SongMetadata,
    load_aries_mix_feature_set,
)
from djprojectexploration.waveform_features import (
    default_waveform_npz_path,
    load_waveform_feature_lookup,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTROL_MODE_CHOICES = ("genre-mixability", "legacy-weights")


def _norm_token(value: str) -> str:
    return Path(str(value).strip()).name.lower()


def _norm_mix(value: str) -> str:
    text = str(value).strip().lower()
    if text.endswith("-mix"):
        text = text[: -len("-mix")]
    return text


def _json_script_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _camelot_key_color(value: str) -> str:
    token = str(value or "").strip().upper()
    number = ""
    for ch in token:
        if ch.isdigit():
            number += ch
        else:
            break
    if not number:
        return "#cbd5e1"
    try:
        n = int(number)
    except ValueError:
        return "#cbd5e1"
    palette = {
        1: "#ef4444",
        2: "#f97316",
        3: "#f59e0b",
        4: "#eab308",
        5: "#84cc16",
        6: "#22c55e",
        7: "#14b8a6",
        8: "#06b6d4",
        9: "#3b82f6",
        10: "#6366f1",
        11: "#a855f7",
        12: "#ec4899",
    }
    return palette.get(n, "#cbd5e1")


def _format_duration(seconds: float | None) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value) or value <= 0:
        return ""
    total = int(round(value))
    return f"{total // 60}:{total % 60:02d}"


def _app_asset_uri(path: str) -> str:
    if not path:
        return ""
    resolved = Path(path).expanduser().resolve()
    try:
        rel = resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return ""
    return "/assets/" + rel.as_posix()


def _extract_mp3_metadata(
    *,
    audio_path: Path | None,
    artwork_dir: Path,
    artwork_key: str,
) -> dict[str, Any]:
    if audio_path is None or not audio_path.exists():
        return {"duration_seconds": float("nan"), "artwork_path": "", "artwork_mime": ""}
    duration_seconds = float("nan")
    artwork_path = ""
    artwork_mime = ""
    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(str(audio_path))
        if audio is not None and getattr(audio, "info", None) is not None:
            length = getattr(audio.info, "length", None)
            if length is not None:
                duration_seconds = float(length)
    except Exception:
        audio = None

    try:
        from mutagen.id3 import ID3
        from PIL import Image, ImageOps

        tags = ID3(str(audio_path))
        apic_frames = tags.getall("APIC")
        if apic_frames:
            frame = apic_frames[0]
            digest = hashlib.sha1((str(audio_path.resolve()) + "|" + artwork_key).encode("utf-8")).hexdigest()[:12]
            artwork_dir.mkdir(parents=True, exist_ok=True)
            out_path = artwork_dir / f"{Path(artwork_key).stem}_{digest}.jpg"
            if not out_path.exists():
                image = Image.open(io.BytesIO(frame.data)).convert("RGB")
                image = ImageOps.fit(image, (192, 192), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
                image.save(out_path, format="JPEG", quality=86, optimize=True)
            artwork_path = str(out_path.resolve())
            artwork_mime = "image/jpeg"
    except Exception:
        pass

    return {
        "duration_seconds": duration_seconds,
        "artwork_path": artwork_path,
        "artwork_mime": artwork_mime,
    }


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

        waveform_lookup: dict[str, dict[str, Any]] = {}
        waveform_npz = default_waveform_npz_path(
            tracklist_csv,
            output_dir=project_root / "data" / "waveform_features",
        )
        if waveform_npz.exists():
            waveform_lookup = load_waveform_feature_lookup(waveform_npz)
        else:
            print(
                f"Warning: waveform feature collection not found: {waveform_npz}. "
                "Run `uv run djprojectexploration-waveforms <tracklist_csv>` for detailed waveforms.",
                file=sys.stderr,
            )

        for local_i, meta in enumerate(mix_features.metadata):
            global_idx = len(records)
            global_track_number = global_idx + 1
            token = _norm_token(meta.filename)
            row = csv_by_token.get(token, {})
            energy_values = energy_lookup.get((_norm_mix(mix_slug), token), {})
            waveform_meta = waveform_lookup.get(token, {})

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
            audio_uri = (
                Path(os.path.relpath(audio_path.resolve(), html_output_dir.resolve())).as_posix()
                if audio_path is not None
                else ""
            )
            media_meta = _extract_mp3_metadata(
                audio_path=audio_path,
                artwork_dir=project_root / "data" / "artwork" / mix_slug,
                artwork_key=meta.filename,
            )
            artwork_path_value = str(media_meta.get("artwork_path") or "")
            artwork_uri = (
                Path(os.path.relpath(Path(artwork_path_value), html_output_dir.resolve())).as_posix()
                if artwork_path_value
                else ""
            )
            waveform_path_value = str(waveform_meta.get("waveform_path") or "")
            waveform_uri = (
                Path(os.path.relpath(Path(waveform_path_value), html_output_dir.resolve())).as_posix()
                if waveform_path_value
                else ""
            )
            duration_seconds = float(media_meta.get("duration_seconds", np.nan))
            if not np.isfinite(duration_seconds) and waveform_meta.get("duration_seconds") is not None:
                duration_seconds = float(waveform_meta["duration_seconds"])

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
                    "track_id": f"{mix_slug}:{track_num_tag}",
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
                    "mix_slug": mix_slug,
                    "audio_path": "" if audio_path is None else str(audio_path),
                    "audio_uri": audio_uri,
                    "duration_seconds": duration_seconds,
                    "duration_text": _format_duration(duration_seconds),
                    "artwork_path": str(media_meta.get("artwork_path") or ""),
                    "artwork_uri": artwork_uri,
                    "artwork_mime": str(media_meta.get("artwork_mime") or ""),
                    "waveform_path": waveform_path_value,
                    "waveform_uri": waveform_uri,
                    "waveform_peaks": waveform_meta.get("waveform_peaks") or [],
                    "waveform_preview_peaks": waveform_meta.get("waveform_preview_peaks") or [],
                    "waveform_preview_min": waveform_meta.get("waveform_preview_min") or [],
                    "waveform_preview_max": waveform_meta.get("waveform_preview_max") or [],
                    "waveform_detail_bins": int(waveform_meta.get("waveform_detail_bins") or 0),
                    "waveform_band_bins": int(waveform_meta.get("waveform_band_bins") or 0),
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
    temperature: float,
    maest_similarity: np.ndarray,
    tempo_similarity: np.ndarray,
    groove_similarity: np.ndarray,
    chroma_similarity: np.ndarray,
    maest_distance: np.ndarray,
    tempo_distance: np.ndarray,
    groove_distance: np.ndarray,
    chroma_distance: np.ndarray,
) -> dict[str, dict[str, Any]]:
    payload = _build_similarity_payload_4way(
        records=records,
        maest_similarity=maest_similarity,
        tempo_similarity=tempo_similarity,
        groove_similarity=groove_similarity,
        chroma_similarity=chroma_similarity,
        maest_distance=maest_distance,
        tempo_distance=tempo_distance,
        groove_distance=groove_distance,
        chroma_distance=chroma_distance,
        temperature=temperature,
    )
    for src_idx, group in payload.items():
        for initial_rank, row in enumerate(group["candidates"], start=1):
            cand_record = records[int(row["idx"])]
            row["initial_rank"] = initial_rank
            row["global_track_number"] = int(cand_record["global_track_number"])
            row["filename"] = str(cand_record["filename"])
            row["raw_genre"] = str(cand_record.get("raw_genre", cand_record["genre"]))
            row["duration_seconds"] = float(cand_record["duration_seconds"]) if np.isfinite(float(cand_record["duration_seconds"])) else None
            row["duration_text"] = str(cand_record.get("duration_text") or "")
            row["artwork_uri"] = str(cand_record.get("artwork_uri") or "")
            row["artwork_path"] = str(cand_record.get("artwork_path") or "")
            row["is_self"] = int(row["idx"]) == int(src_idx)
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


def _default_genre_mixability_weights(default_weights: dict[str, float]) -> tuple[float, float, float, float]:
    style = float(np.clip(default_weights.get("maest", 0.45), 0.0, 1.0))
    tempo = max(0.0, float(default_weights.get("mix_tempo", 0.34)))
    groove = max(0.0, float(default_weights.get("mix_groove", 0.33)))
    chroma = max(0.0, float(default_weights.get("mix_chroma", 0.33)))
    total = max(tempo + groove + chroma, 1e-12)
    mix = 1.0 - style
    return style, mix * tempo / total, mix * groove / total, mix * chroma / total


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
                p["human_energy"], p["glm_energy"], p.get("raw_genre", p["genre"]),
                p.get("duration_text", ""), _camelot_key_color(str(p["key"])),
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
            hoverinfo="none",
        )

    fig.add_scatter(
        x=[],
        y=[],
        mode="markers",
        name="Hovered",
        marker={"size": 18, "color": "rgba(255,255,255,0.02)", "line": {"color": "rgba(255,255,255,0.92)", "width": 2}},
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_scatter(
        x=[],
        y=[],
        mode="markers",
        name="Selected",
        marker={"size": 24, "color": "rgba(255,255,255,0.16)", "line": {"color": "#e5e7eb", "width": 3}},
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_scatter(
        x=[],
        y=[],
        mode="lines",
        name="Transition pair",
        line={"color": "rgba(245,158,11,0.0)", "width": 0},
        hoverinfo="skip",
        showlegend=True,
    )
    fig.add_scatter(
        x=[],
        y=[],
        mode="markers+text",
        name="Track 1",
        marker={"size": 20, "color": "rgba(16,185,129,0.18)", "line": {"color": "#10b981", "width": 3}},
        text=[],
        textposition="top center",
        textfont={"color": "#6ee7b7", "size": 12},
        hoverinfo="skip",
        showlegend=True,
    )
    fig.add_scatter(
        x=[],
        y=[],
        mode="markers+text",
        name="Track 2",
        marker={"size": 20, "color": "rgba(245,158,11,0.18)", "line": {"color": "#f59e0b", "width": 3}},
        text=[],
        textposition="top center",
        textfont={"color": "#fbbf24", "size": 12},
        hoverinfo="skip",
        showlegend=True,
    )
    fig.add_scatter(
        x=[],
        y=[],
        mode="markers+text",
        name="Sequence path",
        marker={"size": 13, "color": "#14b8a6", "line": {"color": "#ecfeff", "width": 1.5}},
        text=[],
        textposition="top center",
        textfont={"color": "#99f6e4", "size": 12},
        hoverinfo="skip",
        showlegend=True,
    )
    fig.add_scatter(
        x=[],
        y=[],
        mode="lines",
        name="Recommended next links",
        line={"color": "rgba(245,158,11,0.0)", "width": 0},
        hoverinfo="skip",
        showlegend=True,
    )
    fig.add_scatter(
        x=[],
        y=[],
        mode="markers+text",
        name="Recommended next",
        marker={"size": 15, "color": "rgba(245,158,11,0.10)", "line": {"color": "rgba(245,158,11,0.62)", "width": 1.5}},
        text=[],
        textposition="bottom center",
        textfont={"color": "#fbbf24", "size": 11},
        customdata=[],
        hoverinfo="skip",
        hovertemplate=None,
        showlegend=True,
    )

    fig.update_layout(
        title={"text": "", "x": 0.5, "xanchor": "center"},
        autosize=True,
        height=650,
        margin={"t": 10, "r": 10, "b": 72, "l": 10},
        legend_title_text="Genre",
        legend={"orientation": "h", "x": 0.0, "y": -0.08, "xanchor": "left", "yanchor": "top"},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
        hoverlabel={
            "bgcolor": "#0f172a",
            "bordercolor": "#475569",
            "font": {"color": "#f8fafc", "size": 12},
            "namelength": -1,
        },
    )
    fig.update_xaxes(title_text="", showticklabels=False, showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(title_text="", showticklabels=False, showgrid=False, zeroline=False, visible=False)
    return fig.to_html(include_plotlyjs=True, full_html=False, div_id=plot_div_id, config={"responsive": True})


def _build_html(
    *,
    plot_html: str,
    records: list[dict[str, Any]],
    coords: np.ndarray,
    layouts: dict[str, list[list[float]]] | None,
    similarity_payload: dict[str, dict[str, Any]],
    plot_div_id: str,
    title: str,
    default_length: int,
    default_weights: dict[str, float],
    control_mode: str,
    static_layout: bool,
    temperature: float,
    top_k_rows: int,
    app_mode: bool = False,
) -> str:
    idx_to_point = {
        str(i): [float(coords[i, 0]), float(coords[i, 1])]
        for i in range(coords.shape[0])
    }
    layout_entries = []
    if layouts:
        for key, points in layouts.items():
            weights = [float(part) for part in key.split(",")]
            layout_entries.append({"key": key, "weights": weights, "points": points})
    record_payload = [
        {
            "idx": int(r["idx"]),
            "track_id": str(r.get("track_id") or f'{r["mix_slug"]}:{r["track_number"]}'),
            "global_track_number": int(r["global_track_number"]),
            "track_number": str(r["track_number"]),
            "filename": str(r["filename"]),
            "title": str(r["title"]),
            "artists": str(r["artists"]),
            "genre": str(r["genre"]),
            "raw_genre": str(r.get("raw_genre", r["genre"])),
            "key": str(r["key"]),
            "csv_bpm": str(r["csv_bpm"]),
            "est_bpm": float(r["est_bpm"]),
            "est_conf": float(r["est_conf"]),
            "mix_slug": str(r["mix_slug"]),
            "audio_path": str(r["audio_path"]),
            "audio_uri": _app_asset_uri(str(r["audio_path"])) if app_mode else str(r.get("audio_uri") or ""),
            "duration_seconds": float(r["duration_seconds"]) if np.isfinite(float(r["duration_seconds"])) else None,
            "duration_text": str(r.get("duration_text") or ""),
            "artwork_uri": _app_asset_uri(str(r["artwork_path"])) if app_mode else str(r.get("artwork_uri") or ""),
            "artwork_path": str(r.get("artwork_path") or ""),
            "artwork_mime": str(r.get("artwork_mime") or ""),
            "waveform_uri": _app_asset_uri(str(r.get("waveform_path") or "")) if app_mode else str(r.get("waveform_uri") or ""),
            "waveform_detail_uri": _app_asset_uri(str(r.get("waveform_path") or "")) if app_mode else str(r.get("waveform_uri") or ""),
            "waveform_path": str(r.get("waveform_path") or ""),
            "waveform_peaks": list(r.get("waveform_peaks") or []),
            "waveform_preview_peaks": list(r.get("waveform_preview_peaks") or []),
            "waveform_preview_min": list(r.get("waveform_preview_min") or []),
            "waveform_preview_max": list(r.get("waveform_preview_max") or []),
            "waveform_detail_bins": int(r.get("waveform_detail_bins") or 0),
            "waveform_band_bins": int(r.get("waveform_band_bins") or 0),
            "snippet_uri": _app_asset_uri(str(r["snippet_path"])) if app_mode else str(r["snippet_uri"]),
            "snippet_path": str(r["snippet_path"]),
            "snippet_start": float(r["snippet_start"]),
            "snippet_end": float(r["snippet_end"]),
            "snippet_rms": float(r["snippet_rms"]),
            "human_energy": float(r["human_energy"]) if np.isfinite(float(r["human_energy"])) else None,
            "glm_energy": float(r["glm_energy"]) if np.isfinite(float(r["glm_energy"])) else None,
        }
        for r in records
    ]
    record_payload_by_idx = {int(r["idx"]): r for r in record_payload}
    similarity_payload_out = dict(similarity_payload)
    for group in similarity_payload_out.values():
        candidates = group.get("candidates") if isinstance(group, dict) else None
        if not isinstance(candidates, list):
            continue
        for row in candidates:
            if not isinstance(row, dict):
                continue
            canonical = record_payload_by_idx.get(int(row.get("idx", -1)))
            if not canonical:
                continue
            row["raw_genre"] = canonical.get("raw_genre", row.get("genre", ""))
            row["duration_seconds"] = canonical.get("duration_seconds")
            row["duration_text"] = canonical.get("duration_text", "")
            row["artwork_uri"] = canonical.get("artwork_uri", "")
            row["artwork_path"] = canonical.get("artwork_path", "")

    data_scripts = "\n".join(
        [
            f'<script id="seq-records-json" type="application/json">{_json_script_payload(record_payload)}</script>',
            f'<script id="seq-sim-json" type="application/json">{_json_script_payload(similarity_payload_out)}</script>',
            f'<script id="seq-points-json" type="application/json">{_json_script_payload(idx_to_point)}</script>',
            f'<script id="seq-layout-entries-json" type="application/json">{_json_script_payload(layout_entries)}</script>',
            f'<script id="seq-config-json" type="application/json">{_json_script_payload({"default_length": default_length, "weights": default_weights, "control_mode": control_mode, "static_layout": static_layout, "temperature": temperature, "top_k_rows": top_k_rows, "app_mode": bool(app_mode)})}</script>',
        ]
    )

    if control_mode == "legacy-weights":
        weight_controls_html = """
    <label>MAEST <input id="weight-maest" type="range" min="0" max="1" step="0.01"></label>
    <output id="weight-maest-val">0.00</output>
    <label>Chroma <input id="weight-chroma" type="range" min="0" max="1" step="0.01"></label>
    <output id="weight-chroma-val">0.00</output>
    <label>BPM <input id="weight-tempo" type="range" min="0" max="1" step="0.01"></label>
    <output id="weight-tempo-val">0.00</output>
"""
    else:
        weight_controls_html = """
    <div class="mixability-controls">
      <div class="weights">
        <div>Genre/style</div><input id="weight-style" type="range" min="0" max="1" step="0.01" value="0.45"><output id="weight-style-val">0.45</output>
      </div>
      <svg id="simplex-control" class="simplex-control" viewBox="0 0 360 320" role="img" aria-label="Tempo groove key mixability triangle">
        <polygon class="simplex-area" points="180,34 44,270 316,270"></polygon>
        <line class="simplex-grid-line" x1="166.4" y1="57.6" x2="71.2" y2="270"></line>
        <line class="simplex-grid-line" x1="193.6" y1="57.6" x2="288.8" y2="270"></line>
        <line class="simplex-grid-line" x1="153.0" y1="80.8" x2="98.4" y2="270"></line>
        <line class="simplex-grid-line" x1="207.0" y1="80.8" x2="261.6" y2="270"></line>
        <line class="simplex-grid-line" x1="112.0" y1="151.6" x2="248.0" y2="151.6"></line>
        <line class="simplex-grid-line" x1="98.4" y1="175.2" x2="261.6" y2="175.2"></line>
        <line class="simplex-grid-line" x1="84.8" y1="198.8" x2="275.2" y2="198.8"></line>
        <line class="simplex-grid-line" x1="71.2" y1="222.4" x2="288.8" y2="222.4"></line>
        <line class="simplex-grid-line" x1="57.6" y1="246.0" x2="302.4" y2="246.0"></line>
        <text class="simplex-label" x="180" y="22">Tempo</text>
        <text class="simplex-label" x="44" y="294">Groove</text>
        <text class="simplex-label" x="316" y="294">Key</text>
        <circle id="simplex-handle" class="simplex-handle" cx="180" cy="128.4" r="10"></circle>
      </svg>
      <div class="weights">
        <div>Tempo</div><input id="weight-tempo" type="range" min="0" max="1" step="0.01" value="0.34"><output id="weight-tempo-val">0.34</output>
        <div>Groove</div><input id="weight-groove" type="range" min="0" max="1" step="0.01" value="0.33"><output id="weight-groove-val">0.33</output>
        <div>Key</div><input id="weight-chroma" type="range" min="0" max="1" step="0.01" value="0.33"><output id="weight-chroma-val">0.33</output>
      </div>
    </div>
"""

    tabs_html = """
<nav class="pane-tabs" aria-label="Builder panes">
  <button class="tab-button active" data-pane-tab="explore">Explore</button>
  <button class="tab-button" data-pane-tab="library">Library</button>
  <button class="tab-button" data-pane-tab="diagnostics">Diagnostics</button>
  <button class="tab-button" data-pane-tab="preview">Transition Preview</button>
</nav>
"""

    controls_html = f"""
<section class="builder">
  <section id="pane-explore" class="pane active">
    <div class="top-workspace">
      <div class="plot-column">
        <div class="plot-pane">
          {plot_html}
          <canvas id="map-effects-canvas" class="map-effects-canvas" aria-hidden="true"></canvas>
          <div id="song-hover-card" class="song-hover-card hidden"></div>
        </div>
        <div id="song-popover" class="song-panel hidden">
          <div id="selected-track" class="song-panel-body"></div>
          <div id="song-wave-player" class="wave-player">
            <button id="song-player-toggle" class="player-toggle" type="button" aria-label="Play">▶</button>
            <div class="wave-main">
              <div class="wave-topline">
                <span id="song-player-status" class="muted">Preview</span>
                <span id="song-player-time" class="muted">0:00 / 0:00</span>
              </div>
              <canvas id="song-waveform" class="song-waveform" height="54"></canvas>
            </div>
          </div>
          <audio id="track-audio" preload="metadata"></audio>
          <div class="actions">
            <button id="set-outgoing">Set Track 1</button>
            <button id="set-incoming">Set Track 2</button>
            <button id="append-selected">Append</button>
            <button id="clear-transition">Clear</button>
          </div>
        </div>
      </div>
      <aside class="side-controls">
        <div id="transition-badges" class="transition-badges"></div>
        <div class="control-grid">
          <label>Sequence length <input id="sequence-length" type="number" min="2" max="40" step="1"></label>
          <label>Energy source
            <select id="energy-source">
              <option value="human">Tagged energy</option>
              <option value="glm">Auto energy</option>
            </select>
          </label>
          <label>Point color
            <select id="color-mode">
              <option value="genre">Genre</option>
              <option value="energy">Energy</option>
              <option value="tempo">Tempo</option>
            </select>
          </label>
          <label>Map FX <input id="map-effects-enabled" type="checkbox" checked></label>
          <label>Energy penalty <input id="energy-penalty-scale" type="range" min="0" max="10" step="0.05" value="2"></label>
          <output id="energy-penalty-scale-val">2.00</output>
          {weight_controls_html}
        </div>
        <div class="side-actions">
          <button id="clear-last">Clear last</button>
          <button id="reset-sequence">Reset sequence</button>
          <button id="download-sequence">Download CSV</button>
        </div>
      </aside>
    </div>
    <div class="explore-lower-grid">
      <div class="energy-column">
        <h2>Energy Curve</h2>
        <div id="energy-curve" class="energy-curve"></div>
        <h2>Recommendations</h2>
        <div id="recommendation-panel" class="recommendation-panel">Select a current track to score candidates.</div>
      </div>
      <div class="sequence-panel">
        <h2>Sequence</h2>
        <div id="sequence-list" class="sequence-list"></div>
      </div>
    </div>
  </section>
  <section id="pane-library" class="pane">
      <div class="library-panel">
      <div class="library-toolbar">
        <label>Search <input id="library-search" type="search" placeholder="title, artist, key, genre, mix"></label>
      </div>
      <div id="library-table" class="library-table"></div>
    </div>
  </section>
  <section id="pane-diagnostics" class="pane">
    <div>
      <h2>Selected Transition</h2>
      <div id="current-transition-score" class="current-transition-score"></div>
    </div>
    <div>
      <h2>Transition Diagnostics</h2>
      <div id="transition-diagnostics" class="transition-diagnostics"></div>
    </div>
  </section>
  <section id="pane-preview" class="pane">
    <div id="transition-preview-panel" class="transition-preview-panel"></div>
  </section>
</section>
<div id="global-player" class="global-player idle">
  <div id="global-art" class="global-art art-placeholder">art</div>
  <div class="global-track">
    <div id="global-title" class="track-title">No track playing</div>
    <div id="global-artist" class="track-artist muted">Select a track to preview</div>
  </div>
  <button id="global-toggle" class="player-toggle" type="button" aria-label="Play" disabled>▶</button>
  <input id="global-scrub" class="scrub-range global-scrub" type="range" min="0" max="1" step="0.1" value="0">
  <div id="global-time" class="scrub-time">0:00 / 0:00</div>
  <label class="global-volume"><span>Vol</span><input id="global-volume" class="scrub-range" type="range" min="0" max="1" step="0.01" value="0.85"></label>
  <button id="global-track1" type="button">Track 1</button>
  <button id="global-track2" type="button">Track 2</button>
</div>
"""

    style = """
<style>
  :root { color-scheme: dark; --bg:#0b1020; --panel:#111827; --panel2:#151f32; --line:#2d374c; --ink:#e5e7eb; --muted:#9ca3af; --accent:#14b8a6; --accent2:#f59e0b; --warn:#fb923c; }
  body { font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 18px; padding-bottom: 88px; color: var(--ink); background: var(--bg); }
  .app-header { display:flex; align-items:center; justify-content:space-between; gap:14px; margin:0 0 14px; }
  h1 { font-size: 24px; margin: 0; }
  h2 { font-size: 15px; margin: 0 0 8px; }
  .builder { margin-top: 14px; display: grid; gap: 12px; }
  .pane-tabs { display: flex; gap: 8px; flex-wrap: wrap; }
  .tab-button { border: 1px solid var(--line); background: var(--panel); color: var(--muted); border-radius: 6px; padding: 6px 10px; }
  .tab-button.active { background: #0f766e; border-color: #2dd4bf; color: #ecfeff; }
  .pane { display: none; }
  .pane.active { display: grid; gap: 12px; }
  .top-workspace { display: grid; grid-template-columns: minmax(0, 1fr) 350px; gap: 12px; align-items: start; }
  .plot-column { min-width: 0; display:grid; gap:10px; }
  .plot-pane { min-width: 0; border: 1px solid var(--line); background: var(--panel); position: relative; overflow: hidden; }
  .map-effects-canvas { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; z-index:1; mix-blend-mode:screen; }
  .plot-pane .js-plotly-plot, .plot-pane .plot-container { position:relative; z-index:2; }
  .song-hover-card { position:absolute; z-index:5; pointer-events:none; max-width:330px; min-width:260px; border:1px solid #334155; background:rgba(15,23,42,.96); box-shadow:0 18px 40px rgba(0,0,0,.34); border-radius:7px; padding:10px; backdrop-filter:blur(8px); }
  .song-hover-card.hidden { display:none; }
  .side-controls { display: grid; gap: 10px; position: sticky; top: 12px; }
  .transition-badges { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .transition-badge { border: 1px solid var(--line); background: var(--panel); padding: 8px; border-radius: 6px; min-height: 48px; min-width:0; overflow:hidden; }
  .transition-badge b { display:block; color: var(--ink); font-size: 12px; margin-bottom: 3px; }
  .transition-badge.out b { color:#6ee7b7; }
  .transition-badge.in b { color:#fbbf24; }
  .transition-mini-score { grid-column: 1 / -1; border: 1px solid var(--line); background: var(--panel); border-radius: 6px; padding: 8px; }
  .transition-mini-score table { font-size: 11px; }
  .transition-mini-score th, .transition-mini-score td { padding: 4px 5px; }
  .control-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(90px, 1fr)); gap: 7px; align-items: center; padding: 9px; border: 1px solid var(--line); background: var(--panel); }
  .side-controls .control-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .mixability-controls { grid-column: 1 / -1; display: grid; grid-template-columns: minmax(260px, 360px) minmax(280px, 1fr); gap: 10px; align-items: center; }
  .side-controls .mixability-controls { grid-template-columns: 1fr; }
  .weights { display: grid; grid-template-columns: 76px 1fr 42px; gap: 6px; align-items: center; }
  .simplex-control { width: 100%; max-width: 250px; justify-self:center; display: block; touch-action: none; user-select: none; }
  .simplex-area { fill: #0f172a; stroke: #64748b; stroke-width: 1.5; }
  .simplex-grid-line { stroke: #334155; stroke-width: .8; }
  .simplex-label { fill: #cbd5e1; font-size: 13px; font-weight: 600; text-anchor: middle; }
  .simplex-handle { fill: #2dd4bf; stroke: #0f172a; stroke-width: 4; cursor: grab; }
  .simplex-handle:active { cursor: grabbing; }
  label { display: grid; gap: 4px; font-size: 11px; color: var(--muted); }
  input, select, button { font: inherit; }
  input[type="number"], select { padding: 5px 6px; border: 1px solid var(--line); border-radius: 4px; background: #0f172a; color: var(--ink); min-width:0; }
  input[type="range"] { accent-color: var(--accent); }
  output { font-size: 11px; text-align: right; color: var(--ink); }
  button { border: 1px solid var(--line); background: #0f172a; color: var(--ink); border-radius: 4px; padding: 6px 9px; cursor: pointer; }
  button:hover { border-color: #38bdf8; }
  button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  button:disabled { opacity: .45; cursor: not-allowed; }
  .song-panel audio { display: none; }
  .actions { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
  .side-controls .actions { justify-content: flex-start; }
  .side-actions { display:flex; gap:8px; flex-wrap:wrap; border:1px solid var(--line); background:var(--panel); padding:9px; }
  .energy-curve { height: 260px; border: 1px solid var(--line); background: var(--panel); }
  .explore-lower-grid { display: grid; grid-template-columns: minmax(420px, 1.15fr) minmax(360px, .85fr); gap: 12px; align-items: start; }
  .energy-column, .sequence-panel { min-width: 0; display: grid; gap: 8px; align-content: start; }
  .sequence-panel { align-self: start; }
  .sequence-list, .recommendation-panel, .transition-diagnostics, .transition-preview-panel { border: 1px solid var(--line); background: var(--panel); max-height: 560px; overflow: auto; }
  .sequence-list { height: 260px; max-height: 260px; overflow-y: auto; }
  .recommendation-panel { max-height: 310px; }
  .recommendation-panel table { font-size: 11px; }
  .recommendation-panel th, .recommendation-panel td { padding: 5px 6px; }
  .transition-preview-panel { max-height: none; min-height: 760px; padding: 0; overflow:auto; background:#0b1020; }
  #pane-diagnostics .transition-diagnostics { min-height: 620px; max-height: 720px; }
  .transition-workbench { min-height:760px; display:grid; grid-template-columns:1fr; align-content:start; background:#0b1020; }
  .transition-controls { border-top:1px solid var(--line); border-bottom:1px solid var(--line); background:#101827; padding:12px 14px; overflow:visible; }
  .transition-main { min-width:0; display:grid; grid-template-rows:auto auto auto; }
  .transition-topbar { border-bottom:1px solid var(--line); background:var(--panel2); padding:10px 12px; display:flex; gap:10px; align-items:center; justify-content:space-between; }
  .transition-meta { min-width:0; color:var(--muted); font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .transition-links { display:flex; gap:8px; flex-wrap:wrap; }
  .transition-links a { color:#67e8f9; text-decoration:none; font-size:12px; }
  .transition-loading { display:none; width:100%; max-width:260px; height:6px; border-radius:999px; overflow:hidden; background:rgba(51,65,85,.72); position:relative; }
  .transition-loading.active { display:block; }
  .transition-loading span { position:absolute; top:0; bottom:0; width:46%; border-radius:999px; background:linear-gradient(90deg, #67e8f9, #1db954); animation:transition-loading 1.05s ease-in-out infinite; }
  @keyframes transition-loading {
    0% { left:-52%; }
    100% { left:106%; }
  }
  .transition-scrubbar { position:relative; z-index:3; display:grid; grid-template-columns:1fr; gap:7px; align-items:center; padding:8px 10px; border:1px solid #2d2d2d; border-radius:7px; background:#101827; }
  .transition-scrubbar-empty { min-height:54px; color:var(--muted); font-size:12px; align-content:center; }
  .transition-transport-controls { display:flex; gap:9px; align-items:center; justify-content:space-between; min-width:0; }
  .transition-transport-title { min-width:0; color:#cbd5e1; font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .transition-scrubbar audio { display:none; }
  .transition-editor { border-bottom:1px solid var(--line); background:#101010; padding:12px; display:grid; gap:10px; }
  .transition-editor-head { display:flex; gap:10px; align-items:center; justify-content:space-between; flex-wrap:wrap; }
  .transition-editor-head h2 { margin:0; }
  .transition-editor-tools { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .transition-editor-tools label { display:flex; grid-template-columns:none; flex-direction:row; gap:6px; align-items:center; color:#cbd5e1; }
  .transition-editor-hint { font-size:11px; color:var(--muted); }
  .transition-editor-dirty { color:#fbbf24; }
  .transition-editor-lanes { display:grid; grid-template-columns:1fr; gap:10px; }
  .transition-native-stage { display:grid; gap:9px; min-width:0; }
  .transition-overview-card, .transition-window-card, .transition-lane-card { min-width:0; border:1px solid #2d2d2d; background:#171717; border-radius:7px; padding:10px; display:grid; gap:8px; box-shadow:inset 0 1px 0 rgba(255,255,255,.03); }
  .transition-overview-card { padding:8px 10px 10px; background:#141414; }
  .transition-window-stack { display:grid; grid-template-columns:1fr; gap:10px; }
  .transition-transport-row { display:grid; grid-template-columns:minmax(0,1fr) minmax(230px,300px); gap:8px; align-items:center; }
  .transition-transport-main { min-width:0; }
  .transition-transport-side { min-width:0; color:var(--muted); font-size:12px; padding:0 0 0 10px; border-left:1px solid #2d2d2d; }
  .transition-window-card { background:#151515; grid-template-columns:minmax(0,1fr) minmax(230px,300px); align-items:stretch; }
  .transition-window-main { min-width:0; display:grid; align-content:stretch; }
  .transition-track-side { min-width:0; border-left:1px solid #2d2d2d; padding-left:10px; display:grid; grid-template-rows:auto minmax(0,1fr); gap:10px; align-content:start; }
  .transition-track-meta { min-width:0; padding:8px; border:1px solid #263244; border-radius:6px; background:#101827; overflow:hidden; }
  .transition-track-label { color:#cbd5e1; font-size:11px; font-weight:700; margin-bottom:7px; letter-spacing:0; }
  .transition-lane-head { display:grid; grid-template-columns:auto minmax(0,1fr); gap:8px; align-items:start; }
  .transition-lane-head b { color:#f2f2f2; font-size:12px; }
  .transition-lane-title { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#f2f2f2; font-size:12px; }
  .transition-lane-meta { grid-column:1 / -1; display:flex; gap:8px; flex-wrap:wrap; color:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }
  .transition-lane-tools { display:grid; grid-template-columns:1fr; gap:8px; align-items:end; align-content:start; }
  .transition-lane-tools label { min-width:0; }
  .transition-cue-control { display:grid; grid-template-columns:42px minmax(0,1fr); gap:6px; align-items:center; }
  .transition-cue-control span { color:var(--muted); font-size:11px; }
  .transition-cue-control select { width:100%; }
  .transition-overview-row { display:grid; grid-template-columns:auto minmax(0,1fr); gap:8px; align-items:center; }
  .transition-track-play { width:34px; height:34px; border-radius:999px; display:grid; place-items:center; padding:0; font-size:14px; font-weight:800; }
  .transition-overview-canvas, .transition-section-canvas { width:100%; display:block; background:#1d1d1d; border:1px solid #2d2d2d; cursor:grab; touch-action:none; }
  .transition-overview-canvas { height:76px; border-radius:7px; }
  .transition-section-canvas { height:245px; border-radius:7px; }
  .transition-overview-canvas:active, .transition-section-canvas:active { cursor:grabbing; }
  .transition-editor-empty { border:1px solid var(--line); background:rgba(15,23,42,.62); border-radius:7px; padding:12px; color:var(--muted); font-size:12px; }
  .current-transition-score, .library-panel { border:1px solid var(--line); background:var(--panel); }
  .current-transition-score { padding:10px; overflow:auto; }
  .library-panel { padding:10px; display:grid; gap:10px; }
  .library-toolbar { display:grid; grid-template-columns:minmax(260px,1fr); gap:10px; align-items:end; }
  .library-table { max-height:690px; overflow:auto; border:1px solid var(--line); }
  .library-table th[data-library-sort] { cursor:pointer; user-select:none; }
  .library-table th[data-library-sort]::after { content: attr(data-sort-indicator); color:#67e8f9; margin-left:6px; font-size:10px; }
  .library-table tr.now-playing { background: rgba(29,185,84,.12); box-shadow: inset 3px 0 0 #1db954; }
  .library-table tr.now-playing-paused { background: rgba(148,163,184,.08); box-shadow: inset 3px 0 0 #64748b; }
  .library-actions { display:flex; gap:6px; flex-wrap:wrap; align-items:center; min-width:0; }
  .row-player { display:grid; grid-template-columns:auto minmax(170px, 260px) 62px; gap:7px; align-items:center; min-width:260px; }
  .row-player .play-button { border-radius:999px; min-width:42px; }
  .row-waveform { width:100%; height:30px; border:1px solid #263244; background:#0b1020; border-radius:999px; cursor:pointer; display:block; }
  .scrub-range { --progress:0%; width:100%; min-width:80px; accent-color:#1db954; background:linear-gradient(to right, #1db954 var(--progress), #334155 var(--progress)); border-radius:999px; }
  .scrub-time { color:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; white-space:nowrap; text-align:right; }
  .recommendation-filterbar { display:grid; grid-template-columns: repeat(auto-fit, minmax(112px, 1fr)); gap:7px; padding:8px; border-bottom:1px solid var(--line); background:#0f172a; align-items:end; }
  .recommendation-filterbar label { color:#cbd5e1; }
  .recommendation-filterbar input[type="number"] { width:100%; }
  .automation-summary { border:1px solid var(--line); background:#0f172a; border-radius:6px; padding:8px; margin-top:10px; color:var(--muted); font-size:12px; }
  .preview-grid { display:grid; grid-template-columns: 1fr; gap: 10px; margin-bottom:12px; }
  .preview-card { border: 1px solid var(--line); background: var(--panel2); border-radius: 6px; padding: 10px; min-height: 0; overflow:hidden; }
  .preview-card h3 { margin: 0 0 7px; font-size: 13px; }
  .preview-actions { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:0; }
  .preview-actions a { color:#67e8f9; text-decoration:none; border:1px solid var(--line); border-radius:4px; padding:6px 9px; }
  .transition-settings-panel { display:grid; grid-template-columns:minmax(260px,.72fr) minmax(360px,1fr) auto; gap:12px; align-items:end; }
  .preview-render-layout { display:grid; grid-template-columns:minmax(240px,.7fr) minmax(340px,1fr); gap:12px; align-items:start; }
  .preview-render-section { display:grid; gap:8px; padding:9px; border:1px solid var(--line); border-radius:7px; background:#0f172a; }
  .preview-render-section h3 { margin:0; font-size:12px; color:#dbeafe; }
  .preview-render-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:10px; }
  .preview-timing-grid { grid-template-columns: repeat(3, minmax(86px, 1fr)); gap:8px; }
  .preview-mode-controls { display:grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap:8px; }
  .preview-render-actions { display:grid; gap:8px; align-content:end; min-width:220px; }
  .preview-status { min-height:20px; color:var(--muted); font-size:12px; padding-top:5px; }
  .preview-result { border:1px solid var(--line); background:var(--panel2); border-radius:6px; margin-top:12px; padding:10px; }
  .song-panel { border: 1px solid var(--line); background: linear-gradient(180deg, #151f32 0%, #101827 100%); padding: 10px; display:grid; grid-template-columns:minmax(220px,.88fr) minmax(280px,1.22fr) auto; gap:12px; align-items:center; box-shadow:0 14px 36px rgba(0,0,0,.26); }
  .song-panel.hidden { display:none; }
  .song-panel-body { min-width:0; }
  .wave-player { display:grid; grid-template-columns:auto minmax(0,1fr); gap:10px; align-items:center; min-width:0; }
  .player-toggle { width:44px; height:44px; border-radius:999px; display:grid; place-items:center; background:#1db954; border-color:#1db954; color:#06130a; font-weight:800; font-size:18px; line-height:1; padding:0; }
  .player-toggle:hover { border-color:#7ddf9b; }
  .wave-main { min-width:0; display:grid; gap:5px; }
  .wave-topline { display:flex; justify-content:space-between; gap:8px; font-size:11px; font-variant-numeric:tabular-nums; min-width:0; }
  .song-waveform { width:100%; height:54px; border:1px solid #243244; background:#0b1020; border-radius:7px; cursor:pointer; display:block; }
  .global-player { position:fixed; left:18px; right:18px; bottom:14px; z-index:30; display:grid; grid-template-columns:48px minmax(140px, .9fr) auto minmax(180px, 1.6fr) 82px minmax(112px,.42fr) auto auto; gap:10px; align-items:center; padding:9px 11px; border:1px solid #334155; border-radius:8px; background:rgba(15,23,42,.96); box-shadow:0 18px 48px rgba(0,0,0,.42); backdrop-filter:blur(12px); }
  .global-player.idle { opacity:.82; }
  .global-art { width:48px; height:48px; border-radius:5px; object-fit:cover; border:1px solid var(--line); background:#0f172a; display:grid; place-items:center; color:#64748b; font-size:10px; overflow:hidden; }
  .global-art img { width:100%; height:100%; object-fit:cover; display:block; }
  .global-track { min-width:0; }
  .global-scrub { min-width:140px; }
  .global-volume { display:grid; grid-template-columns:auto minmax(76px,1fr); gap:7px; align-items:center; color:var(--muted); font-size:11px; min-width:112px; }
  .global-volume input { min-width:76px; }
  .track-summary { display:grid; grid-template-columns:48px minmax(0,1fr); gap:10px; align-items:center; min-width:0; max-width:100%; }
  .track-summary.large { grid-template-columns:60px minmax(0,1fr); align-items:start; }
  .track-summary.compact { grid-template-columns:34px minmax(0,1fr); gap:8px; }
  .art-thumb { width:48px; height:48px; border-radius:5px; object-fit:cover; object-position:center; background:#0f172a; border:1px solid var(--line); display:block; }
  .track-summary.large .art-thumb { width:60px; height:60px; }
  .track-summary.compact .art-thumb { width:34px; height:34px; border-radius:4px; }
  .art-placeholder { display:grid; place-items:center; color:#64748b; font-size:11px; }
  .track-title { font-weight:700; color:var(--ink); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; }
  .track-artist, .track-meta-line { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; }
  .preview-card .track-title, .preview-card .track-artist, .preview-card .track-meta-line,
  .transition-badge .track-title, .transition-badge .track-artist, .transition-badge .track-meta-line { white-space:normal; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; }
  .camelot-key { font-weight:700; }
  .target-edit { width:70px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { border-bottom: 1px solid #263244; padding: 6px 7px; vertical-align: top; }
  th { position: sticky; top: 0; background: #0f172a; z-index: 1; text-align: left; color: #cbd5e1; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  tr.selected-slot { background: #12312e; }
  .muted { color: var(--muted); }
  .warn { color: var(--warn); }
  .pill { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 1px 6px; color: var(--muted); font-size: 11px; }
  .play-button { min-height:26px; min-width:34px; padding:2px 7px; font-size:14px; line-height:1; font-weight:800; }
  @media (max-width: 1100px) {
    .app-header { align-items:flex-start; flex-direction:column; }
    .top-workspace, .control-grid, .mixability-controls, .selected-panel, .lower-grid, .explore-lower-grid, .preview-grid, .preview-render-layout, .transition-settings-panel, .transition-window-card, .transition-transport-row, .transition-workbench, .transition-editor-lanes, .transition-lane-tools, .song-panel, .library-toolbar { grid-template-columns: 1fr; }
    .transition-track-side, .transition-transport-side { border-left:0; border-top:1px solid #2d2d2d; padding-left:0; padding-top:10px; }
    .preview-mode-controls { border-left:0; padding-left:0; }
    .side-controls .control-grid { grid-template-columns: 1fr; }
    .side-controls { position: static; }
    .actions { justify-content: flex-start; }
    .global-player { grid-template-columns:42px minmax(0,1fr) auto; left:10px; right:10px; bottom:10px; }
    .global-scrub, #global-time, .global-volume, #global-track1, #global-track2 { grid-column: 1 / -1; }
  }
</style>
"""

    script = r"""
<script>
(function() {
  const records = JSON.parse(document.getElementById('seq-records-json').textContent);
  const simMap = JSON.parse(document.getElementById('seq-sim-json').textContent);
  const idxToPoint = JSON.parse(document.getElementById('seq-points-json').textContent);
  const layoutEntries = JSON.parse(document.getElementById('seq-layout-entries-json').textContent);
  const config = JSON.parse(document.getElementById('seq-config-json').textContent);
  const plot = document.getElementById('__PLOT_ID__');

  const byIdx = new Map(records.map(r => [Number(r.idx), r]));
  let selectedIdx = null;
  let hoveredIdx = null;
  let selectedSlot = null;
  let sequenceLength = Number(config.default_length || 10);
  let sequence = Array.from({ length: sequenceLength }, () => null);
  let targetValues = Array.from({ length: sequenceLength }, () => null);
  let draggingEnergySlot = null;
  let currentPoints = [];
  let activePane = 'explore';
  let transitionFromIdx = null;
  let transitionToIdx = null;
  let lastClickedIdx = null;
  let lastClickMs = 0;
  let lastPlotPointClickMs = 0;
  let libraryQuery = '';
  let librarySort = null;
  let libraryDir = null;
  let appTracks = [];
  let appTracksById = new Map();
  let appOptions = null;
  let appLoadStarted = false;
  let appLoadDone = false;
  let transitionRender = null;
  let transitionRenderedState = null;
  let transitionRenderPending = false;
  let transitionRenderMessage = '';
  let transitionRenderWarn = false;
  let transitionRenderRequestId = 0;
  let mapTrailSegments = [];
  let mapEffectsFrame = null;
  let strongestPairsCache = null;
  let previewAudioIdx = null;
  let previewAudioPlaying = false;
  let previewAudioContext = '';
  let previewAudioUrl = '';
  let previewAudioStart = 0;
  let masterVolume = 0.85;
  let rowScrubPositions = new Map();
  let transitionScrubFrame = null;
  let transitionEditorFrame = null;
  let transitionScrubDragging = false;
  let transitionSnapToBeat = true;
  let transitionEditorDrag = null;
  let recFilters = { sameKey: false, bpmRange: '', energyRange: '', excludeUsed: true, genreMode: 'any' };
  let waveformState = { idx: null, url: '', payload: null, loading: false, error: '' };
  const waveformCache = new Map();
  let waveformPointerActive = false;
  let rowWaveformPointerIdx = null;
  let audioProgressFrame = null;
  let lastPointDoubleClickMs = 0;
  let lastPointDoubleClickIdx = null;
  let previewFormState = {
    from_cue: '',
    to_cue: '',
    overlap_bars: 16,
    front_padding_bars: 2,
    back_padding_bars: 2,
    from_nudge_beats: 0,
    to_nudge_beats: 0,
    from_pitch_shift: 0,
    to_pitch_shift: 0,
    preset: 'auto',
    volume_mode: 'smooth-crossfade',
    eq_mode: 'center-bass-swap',
    filter_mode: 'none',
  };

  const els = {
    sequenceLength: document.getElementById('sequence-length'),
    energySource: document.getElementById('energy-source'),
    colorMode: document.getElementById('color-mode'),
    mapEffectsEnabled: document.getElementById('map-effects-enabled'),
    penaltyScale: document.getElementById('energy-penalty-scale'),
    penaltyScaleVal: document.getElementById('energy-penalty-scale-val'),
    weightStyle: document.getElementById('weight-style'),
    weightMaest: document.getElementById('weight-maest'),
    weightChroma: document.getElementById('weight-chroma'),
    weightTempo: document.getElementById('weight-tempo'),
    weightGroove: document.getElementById('weight-groove'),
    weightStyleVal: document.getElementById('weight-style-val'),
    weightMaestVal: document.getElementById('weight-maest-val'),
    weightChromaVal: document.getElementById('weight-chroma-val'),
    weightTempoVal: document.getElementById('weight-tempo-val'),
    weightGrooveVal: document.getElementById('weight-groove-val'),
    simplex: document.getElementById('simplex-control'),
    simplexHandle: document.getElementById('simplex-handle'),
    tabButtons: Array.from(document.querySelectorAll('[data-pane-tab]')),
    panes: {
      explore: document.getElementById('pane-explore'),
      library: document.getElementById('pane-library'),
      diagnostics: document.getElementById('pane-diagnostics'),
      preview: document.getElementById('pane-preview'),
    },
    transitionBadges: document.getElementById('transition-badges'),
    transitionPreview: document.getElementById('transition-preview-panel'),
    setOutgoing: document.getElementById('set-outgoing'),
    setIncoming: document.getElementById('set-incoming'),
    songPopover: document.getElementById('song-popover'),
    selectedTrack: document.getElementById('selected-track'),
    audio: document.getElementById('track-audio'),
    songPlayerToggle: document.getElementById('song-player-toggle'),
    songPlayerStatus: document.getElementById('song-player-status'),
    songPlayerTime: document.getElementById('song-player-time'),
    songWaveform: document.getElementById('song-waveform'),
    appendSelected: document.getElementById('append-selected'),
    clearTransition: document.getElementById('clear-transition'),
    clearLast: document.getElementById('clear-last'),
    resetSequence: document.getElementById('reset-sequence'),
    downloadSequence: document.getElementById('download-sequence'),
    energyCurve: document.getElementById('energy-curve'),
    sequenceList: document.getElementById('sequence-list'),
    recommendationPanel: document.getElementById('recommendation-panel'),
    transitionDiagnostics: document.getElementById('transition-diagnostics'),
    currentTransitionScore: document.getElementById('current-transition-score'),
    mapEffects: document.getElementById('map-effects-canvas'),
    songHoverCard: document.getElementById('song-hover-card'),
    librarySearch: document.getElementById('library-search'),
    libraryTable: document.getElementById('library-table'),
    globalPlayer: document.getElementById('global-player'),
    globalArt: document.getElementById('global-art'),
    globalTitle: document.getElementById('global-title'),
    globalArtist: document.getElementById('global-artist'),
    globalToggle: document.getElementById('global-toggle'),
    globalScrub: document.getElementById('global-scrub'),
    globalVolume: document.getElementById('global-volume'),
    globalTime: document.getElementById('global-time'),
    globalTrack1: document.getElementById('global-track1'),
    globalTrack2: document.getElementById('global-track2'),
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
  function setRangeProgress(el, value, maxValue) {
    if (!el) return;
    const max = Number(maxValue || el.max || 0);
    const pct = max > 0 ? clamp((Number(value) || 0) / max, 0, 1) * 100 : 0;
    el.style.setProperty('--progress', pct.toFixed(2) + '%');
  }
  function applyMasterVolume() {
    masterVolume = clamp(Number(masterVolume), 0, 1);
    if (els.audio) els.audio.volume = masterVolume;
    const transitionAudio = transitionAudioElement();
    if (transitionAudio) transitionAudio.volume = masterVolume;
    if (els.globalVolume) {
      els.globalVolume.value = String(masterVolume);
      setRangeProgress(els.globalVolume, masterVolume, 1);
      els.globalVolume.setAttribute('aria-valuetext', Math.round(masterVolume * 100) + '%');
      els.globalVolume.setAttribute('title', 'Volume ' + Math.round(masterVolume * 100) + '%');
    }
  }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  }
  function reportUiError(scope, err) {
    const message = err && err.message ? err.message : String(err || 'Unknown error');
    console.error(scope + ': ' + message, err);
    const status = document.getElementById('transition-render-status');
    if (status && String(scope || '').toLowerCase().includes('transition')) {
      status.textContent = scope + ': ' + message;
      status.classList.add('warn');
    }
  }
  function safeUi(scope, fn, fallback=null) {
    try {
      return fn();
    } catch (err) {
      reportUiError(scope, err);
      return fallback;
    }
  }
  function durationText(record) {
    if (!record) return '';
    if (record.duration_text) return String(record.duration_text);
    const n = Number(record.duration_seconds);
    if (!Number.isFinite(n) || n <= 0) return '';
    const total = Math.round(n);
    return Math.floor(total / 60) + ':' + String(total % 60).padStart(2, '0');
  }
  function roundedBpm(record) {
    const n = Number(record && record.est_bpm);
    return Number.isFinite(n) ? String(Math.round(n)) : '';
  }
  function camelotKeyColor(value) {
    const m = String(value || '').trim().toUpperCase().match(/^(\d{1,2})[AB]?/);
    if (!m) return '#cbd5e1';
    const palette = {
      1:'#ef4444', 2:'#f97316', 3:'#f59e0b', 4:'#eab308',
      5:'#84cc16', 6:'#22c55e', 7:'#14b8a6', 8:'#06b6d4',
      9:'#3b82f6', 10:'#6366f1', 11:'#a855f7', 12:'#ec4899',
    };
    return palette[Number(m[1])] || '#cbd5e1';
  }
  function keyHtml(key) {
    return '<span class="camelot-key" style="color:' + esc(camelotKeyColor(key)) + '">' + esc(key || '') + '</span>';
  }
  function canonicalRecord(record) {
    if (!record) return null;
    const idx = Number(record.idx);
    if (Number.isFinite(idx) && byIdx.has(idx)) {
      return Object.assign({}, record, byIdx.get(idx));
    }
    return record;
  }
  function artworkHtml(record, cls='') {
    record = canonicalRecord(record);
    const extra = cls ? ' ' + cls : '';
    if (record && record.artwork_uri) {
      return '<img class="art-thumb' + extra + '" src="' + esc(record.artwork_uri) + '" alt="">';
    }
    return '<div class="art-thumb art-placeholder' + extra + '">art</div>';
  }
  function rawGenre(record) {
    record = canonicalRecord(record);
    return titleCaseGenre(String((record && (record.raw_genre || record.genre)) || 'Unknown'));
  }
  function titleCaseGenre(value) {
    return String(value || '').split(/(\s+|\/|-)/).map(part => {
      if (/^\s+$|^\/$|^-$/.test(part)) return part;
      if (!part) return part;
      const lower = part.toLowerCase();
      if (lower === 'dj') return 'DJ';
      if (lower === 'edm') return 'EDM';
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    }).join('');
  }
  function trackSummaryHtml(record, { size='compact', showArt=true } = {}) {
    record = canonicalRecord(record);
    if (!record) return '<span class="muted">None selected</span>';
    const bits = [rawGenre(record), record.key ? keyHtml(record.key) : '', roundedBpm(record), durationText(record)].filter(Boolean);
    const body =
      '<div class="track-title">' + esc(record.title) + '</div>' +
      '<div class="track-artist muted">' + esc(record.artists) + '</div>' +
      '<div class="track-meta-line muted">' + bits.join(' &sdot; ') + '</div>';
    if (!showArt) return body;
    return '<div class="track-summary ' + esc(size) + '">' + artworkHtml(record) + '<div>' + body + '</div></div>';
  }
  function closestAttr(target, attr) {
    const el = target && target.closest ? target.closest('[' + attr + ']') : null;
    return el ? el.getAttribute(attr) : null;
  }
  function closestEl(target, selector) {
    return target && target.closest ? target.closest(selector) : null;
  }
  function isPlotPointTarget(target) {
    return !!closestEl(target, '.point, .points');
  }
  function trackId(record) {
    return record ? String(record.track_id || (record.mix_slug + ':' + record.track_number)) : '';
  }
  function resetTransitionRenderState() {
    transitionRender = null;
    transitionRenderedState = null;
    transitionRenderPending = false;
    transitionRenderMessage = '';
    transitionRenderWarn = false;
    transitionRenderRequestId += 1;
  }
  function appTrack(record) {
    const id = trackId(record);
    return id ? appTracksById.get(id) || null : null;
  }
  function optionHtml(value, label, selectedValue) {
    const selected = String(value) === String(selectedValue) ? ' selected' : '';
    return '<option value="' + esc(value) + '"' + selected + '>' + esc(label) + '</option>';
  }
  function numericInput(id, label, value, min, max, step) {
    return '<label>' + esc(label) + '<input id="' + id + '" data-preview-field="' + id + '" type="number" min="' + min + '" max="' + max + '" step="' + step + '" value="' + esc(value) + '"></label>';
  }
  function selectInput(id, label, values, selectedValue, labelFn) {
    const labels = labelFn || (v => v);
    return '<label>' + esc(label) + '<select id="' + id + '" data-preview-field="' + id + '">' +
      values.map(v => optionHtml(v, labels(v), selectedValue)).join('') +
      '</select></label>';
  }
  function cueOptions(record, role, selectedValue) {
    const track = appTrack(record);
    const cues = track && Array.isArray(track.cues) ? track.cues : [];
    if (!cues.length) return optionHtml('', '(no imported cues)', '');
    const preferredRole = String(role || '').toLowerCase();
    let selected = selectedValue;
    if (!selected) {
      const preferred = cues.find(cue => String(cue.role || '').toLowerCase() === preferredRole) || cues[0];
      selected = preferred ? preferred.name : '';
    }
    return cues.map(cue => {
      const time = cue.start_seconds == null ? '' : ' @ ' + Number(cue.start_seconds).toFixed(2) + 's';
      const roleText = cue.role ? ' / ' + cue.role : '';
      return optionHtml(cue.name, cue.name + roleText + time, selected);
    }).join('');
  }
  function cuesForRecord(record) {
    const track = appTrack(record);
    return track && Array.isArray(track.cues) ? track.cues : [];
  }
  function defaultCueFor(record, role) {
    const cues = cuesForRecord(record);
    if (!cues.length) return null;
    const preferredRole = String(role || '').toLowerCase();
    return cues.find(cue => String(cue.role || '').toLowerCase() === preferredRole) || cues[0] || null;
  }
  function cueByName(record, cueName, role) {
    const cues = cuesForRecord(record);
    const normalized = String(cueName || '').trim().toUpperCase();
    if (normalized) {
      const match = cues.find(cue => String(cue.name || '').trim().toUpperCase() === normalized);
      if (match) return match;
    }
    return defaultCueFor(record, role);
  }
  function cueSelectHtml(record, role, field, label='Cue') {
    return '<label class="transition-cue-control"><span>' + esc(label) + '</span><select data-preview-field="' + esc(field) + '">' +
      cueOptions(record, role, previewFormState[field]) +
      '</select></label>';
  }
  function trackBpm(record) {
    const track = appTrack(record);
    const bpm = Number(track && track.bpm);
    if (Number.isFinite(bpm) && bpm > 0) return bpm;
    const est = Number(record && record.est_bpm);
    if (Number.isFinite(est) && est > 0) return est;
    const csv = Number(record && record.csv_bpm);
    return Number.isFinite(csv) && csv > 0 ? csv : 120;
  }
  function transitionDeckConfig(deck) {
    const isFrom = deck === 'from';
    return {
      deck,
      label: isFrom ? 'Track 1' : 'Track 2',
      role: isFrom ? 'out' : 'in',
      cueField: isFrom ? 'from_cue' : 'to_cue',
      nudgeField: isFrom ? 'from_nudge_beats' : 'to_nudge_beats',
      record: isFrom ? (transitionFromIdx === null ? null : byIdx.get(transitionFromIdx)) : (transitionToIdx === null ? null : byIdx.get(transitionToIdx)),
      color: isFrom ? '#10b981' : '#f59e0b',
    };
  }
  function transitionWindowInfo(deck) {
    const cfg = transitionDeckConfig(deck);
    const record = cfg.record;
    if (!record) return null;
    const bpm = trackBpm(record);
    const beatSec = 60 / Math.max(1, bpm);
    const beatsPerBar = 4;
    const frontBars = Math.max(0, Number(previewFormState.front_padding_bars) || 0);
    const overlapBars = Math.max(1, Number(previewFormState.overlap_bars) || 1);
    const backBars = Math.max(0, Number(previewFormState.back_padding_bars) || 0);
    const totalBars = frontBars + overlapBars + backBars;
    const cue = cueByName(record, previewFormState[cfg.cueField], cfg.role);
    const cueSeconds = Number(cue && cue.start_seconds);
    const anchor = Number.isFinite(cueSeconds) ? cueSeconds : previewStart(record);
    const nudgeBeats = Number(previewFormState[cfg.nudgeField]) || 0;
    const transitionStart = anchor + (nudgeBeats * beatSec);
    const contextStart = transitionStart - (frontBars * beatsPerBar * beatSec);
    const duration = Math.max(beatSec, totalBars * beatsPerBar * beatSec);
    const trackDuration = Math.max(beatSec, Number(record.duration_seconds) || audioDuration(record) || duration);
    return {
      cfg,
      record,
      bpm,
      beatSec,
      beatsPerBar,
      frontBars,
      overlapBars,
      backBars,
      totalBars,
      cue,
      anchor,
      nudgeBeats,
      transitionStart,
      transitionEnd: transitionStart + (overlapBars * beatsPerBar * beatSec),
      contextStart,
      contextEnd: contextStart + duration,
      duration,
      trackDuration,
    };
  }
  function transitionNudgeBounds(info) {
    if (!info) return { min: -999, max: 999 };
    const overlapDuration = Math.max(info.beatSec, info.overlapBars * info.beatsPerBar * info.beatSec);
    const min = (-overlapDuration - info.anchor) / info.beatSec;
    const max = (info.trackDuration - info.anchor) / info.beatSec;
    if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: -999, max: 999 };
    return { min: Math.min(min, max), max: Math.max(min, max) };
  }
  function clampTransitionNudge(deck, value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    const info = transitionWindowInfo(deck);
    const bounds = transitionNudgeBounds(info);
    return clamp(n, bounds.min, bounds.max);
  }
  function presetDetails(name) {
    return (appOptions && appOptions.preset_details && appOptions.preset_details[name]) || {};
  }
  function applyPresetToPreviewState(name) {
    previewFormState.preset = name || 'auto';
    const preset = presetDetails(previewFormState.preset);
    previewFormState.volume_mode = preset.volume_mode || previewFormState.volume_mode || 'crossfade';
    previewFormState.eq_mode = preset.eq_mode || previewFormState.eq_mode || 'none';
    previewFormState.filter_mode = preset.filter_mode || previewFormState.filter_mode || 'none';
    ['volume_mode', 'eq_mode', 'filter_mode'].forEach(id => {
      const el = document.getElementById(id);
      if (el && previewFormState[id] != null) el.value = previewFormState[id];
    });
    const presetEl = document.getElementById('preset');
    if (presetEl) presetEl.value = previewFormState.preset;
  }
  function resolvedPreviewModes() {
    return {
      volume: previewFormState.volume_mode || 'crossfade',
      eq: previewFormState.eq_mode || 'none',
      filter: previewFormState.filter_mode || 'none',
    };
  }
  function previewSignature(state) {
    const s = state || {};
    return JSON.stringify({
      from_cue: s.from_cue || '',
      to_cue: s.to_cue || '',
      overlap_bars: Number(s.overlap_bars) || 0,
      front_padding_bars: Number(s.front_padding_bars) || 0,
      back_padding_bars: Number(s.back_padding_bars) || 0,
      from_nudge_beats: Number(s.from_nudge_beats) || 0,
      to_nudge_beats: Number(s.to_nudge_beats) || 0,
      from_pitch_shift: Number(s.from_pitch_shift) || 0,
      to_pitch_shift: Number(s.to_pitch_shift) || 0,
      volume_mode: s.volume_mode || 'crossfade',
      eq_mode: s.eq_mode || 'none',
      filter_mode: s.filter_mode || 'none',
    });
  }
  function previewSettingsDirty() {
    return !!(transitionRender && transitionRenderedState && previewSignature(previewFormState) !== previewSignature(transitionRenderedState));
  }
  function transitionLoadingHtml(active=false, id='transition-loading') {
    const idAttr = id ? ' id="' + esc(id) + '"' : '';
    return '<div' + idAttr + ' class="transition-loading' + (active ? ' active' : '') + '" aria-hidden="' + (active ? 'false' : 'true') + '"><span></span></div>';
  }
  function transitionRenderStatusInfo() {
    const dirty = previewSettingsDirty();
    if (transitionRenderPending) return { text: 'Rendering transition...', warn: false };
    if (transitionRender) {
      if (transitionRenderWarn && transitionRenderMessage) return { text: transitionRenderMessage, warn: true };
      return {
        text: dirty ? 'Settings changed. Render again to update audio and visualizer.' : (transitionRenderMessage || 'Rendered preview is current.'),
        warn: dirty,
      };
    }
    if (transitionRenderMessage) return { text: transitionRenderMessage, warn: transitionRenderWarn };
    return { text: 'Ready to render selected transition.', warn: false };
  }
  function automationSummaryHtml() {
    return '<div id="automation-summary" class="automation-summary">' + automationSummaryInnerHtml() + '</div>';
  }
  function syncPreviewFieldElements(field, value, sourceEl=null) {
    document.querySelectorAll('[data-preview-field="' + field + '"]').forEach(el => {
      if (el === sourceEl) return;
      if (el.type === 'number') el.value = String(Number(value) || 0);
      else el.value = String(value == null ? '' : value);
    });
  }
  function updateTransitionDirtyUi() {
    const status = document.getElementById('transition-render-status');
    const button = document.getElementById('transition-render-button');
    const loading = document.getElementById('transition-loading');
    const dirty = previewSettingsDirty();
    if (button) {
      button.textContent = transitionRenderPending ? 'Rendering...' : (transitionRender && dirty ? 'Render again' : 'Render transition');
      button.disabled = transitionRenderPending;
    }
    if (loading) {
      loading.classList.toggle('active', transitionRenderPending);
      loading.setAttribute('aria-hidden', transitionRenderPending ? 'false' : 'true');
    }
    if (status) {
      const info = transitionRenderStatusInfo();
      status.textContent = info.text;
      status.classList.toggle('warn', !!info.warn);
    }
    const editorDirty = document.getElementById('transition-editor-dirty');
    if (editorDirty) {
      editorDirty.textContent = transitionRenderPending ? 'Rendering' : (dirty ? 'Pending render' : (transitionRender ? 'Rendered' : 'Not rendered'));
      editorDirty.classList.toggle('transition-editor-dirty', transitionRenderPending || dirty);
    }
  }
  function showTransitionPendingUi() {
    updateTransitionDirtyUi();
    const transport = document.querySelector('.transition-transport-row');
    if (transport && !transitionRender) transport.outerHTML = transitionTransportHtml();
    const meta = document.querySelector('.transition-meta');
    if (meta && !transitionRender) meta.textContent = 'Rendering transition...';
  }
  function updatePreviewEffectDisplay({ dirty=true } = {}) {
    const summary = document.getElementById('automation-summary');
    if (summary) summary.innerHTML = automationSummaryInnerHtml();
    safeUi('transition editor draw', drawTransitionEditor);
    if (dirty) safeUi('transition dirty ui update', updateTransitionDirtyUi);
  }
  function automationSummaryInnerHtml() {
    const modes = resolvedPreviewModes();
    return '<b>Resolved automation</b><br>' +
      'preset ' + esc(previewFormState.preset || 'auto') +
      ' / volume ' + esc(modes.volume || 'none') +
      ' / EQ ' + esc(modes.eq || 'none') +
      ' / filter ' + esc(modes.filter || 'none');
  }
  async function loadAppRuntime() {
    if (!config.app_mode || appLoadStarted) return;
    appLoadStarted = true;
    try {
      const [tracksRes, optionsRes] = await Promise.all([fetch('/api/tracks'), fetch('/api/options')]);
      const tracksPayload = await tracksRes.json();
      const optionsPayload = await optionsRes.json();
      if (!tracksPayload.ok) throw new Error(tracksPayload.error || 'Track load failed');
      if (!optionsPayload.ok) throw new Error(optionsPayload.error || 'Option load failed');
      appTracks = tracksPayload.tracks || [];
      appTracksById = new Map(appTracks.map(track => [String(track.id), track]));
      appOptions = optionsPayload;
      applyPresetToPreviewState(previewFormState.preset);
      appLoadDone = true;
    } catch (err) {
      appOptions = { error: err.message || String(err) };
    }
    renderTransitionPreview();
  }
  function setActivePane(pane, { render=true } = {}) {
    activePane = pane || 'explore';
    els.tabButtons.forEach(btn => btn.classList.toggle('active', btn.getAttribute('data-pane-tab') === activePane));
    Object.keys(els.panes).forEach(key => {
      if (els.panes[key]) els.panes[key].classList.toggle('active', key === activePane);
    });
    if (activePane === 'explore' && window.Plotly && plot) {
      setTimeout(() => Plotly.Plots.resize(plot), 30);
      setTimeout(ensureMapEffectsLoop, 40);
    }
    if (activePane === 'explore' && window.Plotly && els.energyCurve) {
      setTimeout(() => Plotly.Plots.resize(els.energyCurve), 30);
    }
    if (activePane === 'preview') {
      setTimeout(() => safeUi('transition editor bind', bindTransitionEditor), 20);
      setTimeout(() => safeUi('transition editor draw', drawTransitionEditor), 30);
    }
    if (render) setTimeout(() => safeUi('pane render', renderAll), 0);
  }
  function shortTrack(record) {
    return trackSummaryHtml(record, { size: 'compact', showArt: false });
  }
  function trackMetaHtml(record) {
    if (!record) return '<div class="muted">No track assigned.</div>';
    return trackSummaryHtml(record, { size: 'large', showArt: true });
  }
  function timeText(seconds) {
    const n = Number(seconds);
    if (!Number.isFinite(n) || n < 0) return '0:00';
    const total = Math.floor(n);
    return Math.floor(total / 60) + ':' + String(total % 60).padStart(2, '0');
  }
  function audioUri(record) {
    record = canonicalRecord(record);
    return String((record && (record.audio_uri || record.snippet_uri)) || '');
  }
  function previewStart(record) {
    record = canonicalRecord(record);
    const start = Number(record && record.snippet_start);
    const duration = Number(record && record.duration_seconds);
    if (!Number.isFinite(start) || start < 0) return 0;
    if (Number.isFinite(duration) && duration > 1) return clamp(start, 0, Math.max(0, duration - 1));
    return start;
  }
  function audioDuration(record) {
    const activeDuration = els.audio && Number.isFinite(Number(els.audio.duration)) ? Number(els.audio.duration) : NaN;
    if (record && previewAudioIdx === Number(record.idx) && Number.isFinite(activeDuration) && activeDuration > 0) return activeDuration;
    const duration = Number(record && record.duration_seconds);
    return Number.isFinite(duration) && duration > 0 ? duration : activeDuration;
  }
  function rowScrubValue(record) {
    record = canonicalRecord(record);
    if (!record) return 0;
    const idx = Number(record.idx);
    const stored = rowScrubPositions.get(idx);
    if (Number.isFinite(Number(stored))) return Number(stored);
    if (previewAudioIdx === idx && els.audio && Number.isFinite(Number(els.audio.currentTime))) return Number(els.audio.currentTime);
    return 0;
  }
  function rowPlayerHtml(record) {
    record = canonicalRecord(record);
    if (!record) return '';
    const duration = audioDuration(record);
    const max = Number.isFinite(duration) && duration > 0 ? duration : Math.max(1, Number(record.duration_seconds) || 1);
    const value = clamp(rowScrubValue(record), 0, max);
    return '<div class="row-player">' +
      '<button class="play-button" data-play-idx="' + record.idx + '" data-play-mode="library" aria-label="Play">▶</button>' +
      '<canvas class="row-waveform" data-library-waveform-idx="' + record.idx + '" data-duration="' + esc(max) + '" height="30" aria-label="Playback scrubber"></canvas>' +
      '<span class="scrub-time" data-library-time-idx="' + record.idx + '">' + esc(timeText(value)) + ' / ' + esc(durationText(record)) + '</span>' +
      '</div>';
  }
  function currentAudioRecord() {
    if (previewAudioIdx !== null && byIdx.has(Number(previewAudioIdx))) return byIdx.get(Number(previewAudioIdx));
    if (selectedIdx !== null && byIdx.has(Number(selectedIdx))) return byIdx.get(Number(selectedIdx));
    return null;
  }
  function updatePlayButtons() {
    document.querySelectorAll('[data-play-idx]').forEach(btn => {
      const idx = Number(btn.getAttribute('data-play-idx'));
      const playing = Number.isFinite(idx) && idx === previewAudioIdx && previewAudioPlaying;
      btn.textContent = playing ? '⏸' : '▶';
      const label = (playing ? 'Pause preview for ' : 'Play preview for ') + (byIdx.get(idx)?.title || 'track');
      btn.setAttribute('aria-label', label);
      btn.setAttribute('title', label);
    });
    if (els.songPlayerToggle) {
      const mainPlaying = selectedIdx !== null && previewAudioIdx === selectedIdx && previewAudioPlaying;
      els.songPlayerToggle.textContent = mainPlaying ? '⏸' : '▶';
      els.songPlayerToggle.setAttribute('aria-label', mainPlaying ? 'Pause' : 'Play');
      els.songPlayerToggle.setAttribute('title', mainPlaying ? 'Pause' : 'Play');
      els.songPlayerToggle.disabled = selectedIdx === null || !audioUri(byIdx.get(selectedIdx));
    }
  }
  function updateGlobalPlayer() {
    const record = currentAudioRecord();
    const hasRecord = !!(record && audioUri(record));
    const activeRecord = record && previewAudioIdx === Number(record.idx);
    const duration = audioDuration(record);
    const current = activeRecord && els.audio && Number.isFinite(Number(els.audio.currentTime))
      ? Number(els.audio.currentTime)
      : rowScrubValue(record);
    if (els.globalPlayer) els.globalPlayer.classList.toggle('idle', !hasRecord);
    if (els.globalArt) {
      if (record && record.artwork_uri) {
        els.globalArt.className = 'global-art';
        els.globalArt.innerHTML = '<img src="' + esc(record.artwork_uri) + '" alt="">';
      } else {
        els.globalArt.className = 'global-art art-placeholder';
        els.globalArt.textContent = 'art';
      }
    }
    if (els.globalTitle) els.globalTitle.textContent = record ? String(record.title || 'Untitled') : 'No track playing';
    if (els.globalArtist) els.globalArtist.textContent = record ? String(record.artists || '') : 'Select a track to preview';
    if (els.globalToggle) {
      els.globalToggle.disabled = !hasRecord;
      els.globalToggle.textContent = previewAudioPlaying ? '⏸' : '▶';
      els.globalToggle.setAttribute('aria-label', previewAudioPlaying ? 'Pause' : 'Play');
      els.globalToggle.setAttribute('title', previewAudioPlaying ? 'Pause' : 'Play');
    }
    if (els.globalScrub) {
      const max = Number.isFinite(duration) && duration > 0 ? duration : 1;
      els.globalScrub.disabled = !hasRecord;
      els.globalScrub.max = String(max);
      els.globalScrub.value = String(clamp(current || 0, 0, max));
      setRangeProgress(els.globalScrub, current || 0, max);
    }
    if (els.globalTime) els.globalTime.textContent = timeText(current || 0) + ' / ' + timeText(duration);
    if (els.globalTrack1) els.globalTrack1.disabled = !record;
    if (els.globalTrack2) els.globalTrack2.disabled = !record;
  }
  function updateLibraryPlaybackState() {
    document.querySelectorAll('[data-library-row-idx]').forEach(row => {
      const idx = Number(row.getAttribute('data-library-row-idx'));
      const isCurrent = Number.isFinite(idx) && idx === previewAudioIdx;
      row.classList.toggle('now-playing', isCurrent && previewAudioPlaying);
      row.classList.toggle('now-playing-paused', isCurrent && !previewAudioPlaying);
    });
  }
  function updatePlayerTimeLabels() {
    const record = previewAudioIdx === null ? (selectedIdx === null ? null : byIdx.get(selectedIdx)) : byIdx.get(previewAudioIdx);
    const duration = audioDuration(record);
    const current = els.audio && Number.isFinite(Number(els.audio.currentTime)) ? Number(els.audio.currentTime) : 0;
    if (els.songPlayerTime) {
      const displayCurrent = selectedIdx !== null && previewAudioIdx === selectedIdx ? current : previewStart(byIdx.get(selectedIdx));
      const displayDuration = audioDuration(selectedIdx === null ? null : byIdx.get(selectedIdx));
      els.songPlayerTime.textContent = timeText(displayCurrent) + ' / ' + timeText(displayDuration);
    }
    if (els.songPlayerStatus) {
      const activeWaveform = selectedIdx !== null && waveformState.idx === selectedIdx ? waveformState : null;
      if (activeWaveform && activeWaveform.loading) els.songPlayerStatus.textContent = 'Loading waveform';
      else if (activeWaveform && activeWaveform.error) els.songPlayerStatus.textContent = activeWaveform.error;
      else els.songPlayerStatus.textContent = selectedIdx === null ? 'Preview' : 'Full track preview';
    }
    document.querySelectorAll('[data-library-time-idx]').forEach(el => {
      const idx = Number(el.getAttribute('data-library-time-idx'));
      const row = byIdx.get(idx);
      if (!row) return;
      const rowCurrent = rowScrubValue(row);
      el.textContent = timeText(rowCurrent) + ' / ' + durationText(row);
    });
  }
  function updateRowScrubbers() {
    drawLibraryWaveforms();
    updatePlayerTimeLabels();
    updateLibraryPlaybackState();
  }
  function updateTransitionTrackPlayButtons() {
    document.querySelectorAll('[data-transition-track-play]').forEach(button => {
      const deck = button.getAttribute('data-transition-track-play') || '';
      const info = transitionWindowInfo(deck);
      const active = !!(info && previewAudioIdx === Number(info.record.idx) && previewAudioContext === 'transition-' + deck && previewAudioPlaying);
      button.textContent = active ? '⏸' : '▶';
      button.setAttribute('aria-label', (active ? 'Pause ' : 'Play ') + (info && info.cfg ? info.cfg.label : 'track'));
    });
  }
  function syncAudioUi() {
    previewAudioPlaying = !!(els.audio && !els.audio.paused && !els.audio.ended);
    updatePlayButtons();
    updateRowScrubbers();
    updateTransitionTrackPlayButtons();
    updateGlobalPlayer();
    drawMainWaveform();
  }
  function requestAudioProgressFrame() {
    if (audioProgressFrame !== null || !window.requestAnimationFrame) return;
    audioProgressFrame = window.requestAnimationFrame(() => {
      audioProgressFrame = null;
      syncAudioUi();
      if (previewAudioPlaying) requestAudioProgressFrame();
    });
  }
  function seekSharedAudio(seconds) {
    if (!els.audio) return;
    const duration = Number(els.audio.duration);
    const max = Number.isFinite(duration) && duration > 0 ? duration : Number.POSITIVE_INFINITY;
    const target = clamp(Number(seconds) || 0, 0, max);
    try { els.audio.currentTime = target; } catch (err) {}
  }
  function playRecordAt(record, startSeconds, context='point') {
    record = canonicalRecord(record);
    if (!record || !els.audio) return;
    const url = audioUri(record);
    if (!url) return;
    const start = Math.max(0, Number(startSeconds) || 0);
    const idx = Number(record.idx);
    const sourceChanged = previewAudioUrl !== url;
    previewAudioIdx = idx;
    previewAudioContext = context;
    previewAudioStart = start;
    if (sourceChanged) {
      previewAudioUrl = url;
      els.audio.src = url;
      try { els.audio.load(); } catch (err) {}
    }
    applyMasterVolume();
    const doPlay = () => {
      seekSharedAudio(start);
      rowScrubPositions.set(idx, start);
      const promise = els.audio.play();
      if (promise && promise.catch) promise.catch(() => {});
      syncAudioUi();
      requestAudioProgressFrame();
    };
    if (els.audio.readyState >= 1) doPlay();
    else els.audio.addEventListener('loadedmetadata', doPlay, { once: true });
  }
  function pauseSharedAudio() {
    if (!els.audio) return;
    try { els.audio.pause(); } catch (err) {}
    previewAudioPlaying = false;
    syncAudioUi();
  }
  function stopSharedAudio({ clear=false } = {}) {
    if (!els.audio) return;
    pauseSharedAudio();
    if (clear) {
      els.audio.removeAttribute('src');
      try { els.audio.load(); } catch (err) {}
      previewAudioIdx = null;
      previewAudioUrl = '';
      previewAudioContext = '';
      previewAudioStart = 0;
    }
    syncAudioUi();
  }
  function hideSongPopover({ stopAudio=true } = {}) {
    if (els.songPopover) els.songPopover.classList.add('hidden');
    if (stopAudio && previewAudioContext === 'main') stopSharedAudio({ clear: false });
    updatePlayButtons();
  }
  function clearCurrentTrackSelection() {
    selectedIdx = null;
    lastClickedIdx = null;
    lastClickMs = 0;
    hideSongPopover({ stopAudio: false });
    renderSelectionOnly();
  }
  function showSongPopover(record, { autoplay=true } = {}) {
    if (!els.songPopover || !record) return;
    els.songPopover.classList.remove('hidden');
    if (els.selectedTrack) {
      els.selectedTrack.innerHTML = trackSummaryHtml(record, { size: 'large', showArt: true });
    }
    loadMainWaveform(record);
    const start = previewStart(record);
    if (autoplay) {
      playRecordAt(record, start, 'main');
    } else {
      previewAudioIdx = Number(record.idx);
      previewAudioContext = 'main';
      previewAudioStart = start;
      if (els.audio && audioUri(record) && previewAudioUrl !== audioUri(record)) {
        previewAudioUrl = audioUri(record);
        els.audio.src = previewAudioUrl;
        try { els.audio.load(); } catch (err) {}
      }
      if (els.audio) seekSharedAudio(start);
    }
    syncAudioUi();
  }
  function showSongHover(record, nativeEvent) {
    if (record && Number.isFinite(Number(record.idx))) {
      const idx = Number(record.idx);
      if (hoveredIdx !== idx) {
        hoveredIdx = idx;
        safeUi('hover overlay', updatePacmapOverlays);
      }
    }
    if (!els.songHoverCard || !record) return;
    els.songHoverCard.innerHTML = trackSummaryHtml(record, { size: 'large', showArt: true });
    els.songHoverCard.classList.remove('hidden');
    const pane = document.querySelector('.plot-pane');
    if (!pane || !nativeEvent) return;
    const paneRect = pane.getBoundingClientRect();
    const cardRect = els.songHoverCard.getBoundingClientRect();
    let x = Number(nativeEvent.clientX) - paneRect.left + 14;
    let y = Number(nativeEvent.clientY) - paneRect.top + 14;
    x = clamp(x, 8, Math.max(8, paneRect.width - cardRect.width - 8));
    y = clamp(y, 8, Math.max(8, paneRect.height - cardRect.height - 8));
    els.songHoverCard.style.left = x + 'px';
    els.songHoverCard.style.top = y + 'px';
  }
  function hideSongHover() {
    if (hoveredIdx !== null) {
      hoveredIdx = null;
      safeUi('hover overlay', updatePacmapOverlays);
    }
    if (els.songHoverCard) els.songHoverCard.classList.add('hidden');
  }
  function toggleTrackPreview(idx, { mode='point' } = {}) {
    idx = Number(idx);
    if (!byIdx.has(idx)) return;
    const record = byIdx.get(idx);
    const context = mode === 'library' ? 'library' : 'point';
    if (previewAudioIdx === idx && previewAudioContext === context && els.audio && !els.audio.paused) {
      pauseSharedAudio();
      return;
    }
    const start = context === 'library' ? rowScrubValue(record) : previewStart(record);
    playRecordAt(record, start, context);
  }
  function toggleTransitionTrackPreview(deck) {
    const info = transitionWindowInfo(deck);
    if (!info || !info.record) return;
    const idx = Number(info.record.idx);
    const context = 'transition-' + String(deck || '');
    if (previewAudioIdx === idx && previewAudioContext === context && els.audio && !els.audio.paused) {
      pauseSharedAudio();
      return;
    }
    playRecordAt(info.record, Math.max(0, Number(info.transitionStart) || 0), context);
  }
  function normalizeWaveformValues(values) {
    if (!Array.isArray(values)) return [];
    const out = [];
    let maxValue = 0;
    values.forEach(value => {
      const n = Number(value);
      if (!Number.isFinite(n)) return;
      if (n > maxValue) maxValue = n;
      out.push(n);
    });
    const scale = maxValue > 1.001 ? 255 : 1;
    return out.map(value => clamp(value / scale, 0, 1));
  }
  function normalizeSignedWaveformValues(values) {
    if (!Array.isArray(values)) return [];
    const out = [];
    let maxAbs = 0;
    values.forEach(value => {
      const n = Number(value);
      if (!Number.isFinite(n)) return;
      maxAbs = Math.max(maxAbs, Math.abs(n));
      out.push(n);
    });
    const scale = maxAbs > 1.001 ? 127 : 1;
    return out.map(value => clamp(value / scale, -1, 1));
  }
  function waveformDetailUri(record) {
    record = canonicalRecord(record);
    return String((record && (record.waveform_detail_uri || record.waveform_uri)) || '');
  }
  function baseWaveformPayload(record) {
    record = canonicalRecord(record);
    return {
      row: normalizeWaveformValues(record && record.waveform_peaks),
      preview: normalizeWaveformValues(record && record.waveform_preview_peaks),
      detail: [],
      previewMin: normalizeSignedWaveformValues(record && record.waveform_preview_min),
      previewMax: normalizeSignedWaveformValues(record && record.waveform_preview_max),
      detailMin: [],
      detailMax: [],
      bands: null,
      drawCache: new Map(),
      loaded: false,
      loading: false,
      error: '',
      promise: null,
    };
  }
  function waveformPayload(record) {
    record = canonicalRecord(record);
    if (!record) return baseWaveformPayload(null);
    const key = trackId(record);
    if (!waveformCache.has(key)) waveformCache.set(key, baseWaveformPayload(record));
    return waveformCache.get(key);
  }
  function applyWaveformDetail(payload, data) {
    if (!payload || !data) return payload;
    const row = normalizeWaveformValues(data.row_peaks);
    const preview = normalizeWaveformValues(data.preview_peaks);
    const detail = normalizeWaveformValues(data.detail_peaks);
    const envelope = data.envelope || {};
    const bands = data.bands || {};
    if (row.length) payload.row = row;
    if (preview.length) payload.preview = preview;
    if (detail.length) payload.detail = detail;
    const previewMin = normalizeSignedWaveformValues(envelope.preview_min || data.preview_min);
    const previewMax = normalizeSignedWaveformValues(envelope.preview_max || data.preview_max);
    const detailMin = normalizeSignedWaveformValues(envelope.detail_min || data.detail_min);
    const detailMax = normalizeSignedWaveformValues(envelope.detail_max || data.detail_max);
    if (previewMin.length && previewMax.length) {
      payload.previewMin = previewMin;
      payload.previewMax = previewMax;
    }
    if (detailMin.length && detailMax.length) {
      payload.detailMin = detailMin;
      payload.detailMax = detailMax;
    }
    const low = normalizeWaveformValues(bands.low);
    const mid = normalizeWaveformValues(bands.mid);
    const high = normalizeWaveformValues(bands.high);
    payload.bands = (low.length || mid.length || high.length) ? { low, mid, high } : null;
    payload.drawCache = new Map();
    payload.loaded = true;
    payload.error = '';
    return payload;
  }
  function waveformSeries(payload, preference='detail') {
    if (!payload) return [];
    if (preference === 'row') return payload.row || [];
    if (preference === 'preview') return (payload.preview && payload.preview.length ? payload.preview : payload.row) || [];
    return (payload.detail && payload.detail.length ? payload.detail : (payload.preview && payload.preview.length ? payload.preview : payload.row)) || [];
  }
  function waveformEnvelopeSeries(payload, preference='detail') {
    if (!payload) return null;
    const detailReady = preference !== 'row' && payload.detailMin && payload.detailMax && payload.detailMin.length && payload.detailMax.length;
    if (detailReady) return { min: payload.detailMin, max: payload.detailMax };
    if (preference !== 'row' && payload.previewMin && payload.previewMax && payload.previewMin.length && payload.previewMax.length) {
      return { min: payload.previewMin, max: payload.previewMax };
    }
    return null;
  }
  function resampleSeries(series, count) {
    const source = normalizeWaveformValues(series);
    const n = Math.max(2, Math.round(count || 2));
    if (!source.length) return [];
    if (source.length === n) return source;
    if (source.length === 1) return Array.from({ length: n }, () => source[0]);
    const out = [];
    const scale = (source.length - 1) / Math.max(1, n - 1);
    for (let i = 0; i < n; i += 1) {
      const pos = i * scale;
      const lo = Math.floor(pos);
      const hi = Math.min(source.length - 1, lo + 1);
      const frac = pos - lo;
      out.push(source[lo] + (source[hi] - source[lo]) * frac);
    }
    return out;
  }
  function resampleSignedSeries(series, count) {
    const source = normalizeSignedWaveformValues(series);
    const n = Math.max(2, Math.round(count || 2));
    if (!source.length) return [];
    if (source.length === n) return source;
    if (source.length === 1) return Array.from({ length: n }, () => source[0]);
    const out = [];
    const scale = (source.length - 1) / Math.max(1, n - 1);
    for (let i = 0; i < n; i += 1) {
      const pos = i * scale;
      const lo = Math.floor(pos);
      const hi = Math.min(source.length - 1, lo + 1);
      const frac = pos - lo;
      out.push(source[lo] + (source[hi] - source[lo]) * frac);
    }
    return out;
  }
  function drawWaveformEnvelope(ctx, samples, width, height, color, scale=1, pow=1) {
    if (!samples || !samples.length) return;
    const mid = height * 0.5;
    const amp = height * 0.47 * scale;
    const denom = Math.max(1, samples.length - 1);
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    samples.forEach((value, i) => {
      const x = (i / denom) * width;
      const y = mid - Math.pow(clamp(Number(value) || 0, 0, 1), pow) * amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    for (let i = samples.length - 1; i >= 0; i -= 1) {
      const x = (i / denom) * width;
      const y = mid + Math.pow(clamp(Number(samples[i]) || 0, 0, 1), pow) * amp;
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
  function cachedDrawSeries(payload, count) {
    if (!payload) return { peaks: [], low: [], mid: [], high: [], min: [], max: [] };
    if (!payload.drawCache) payload.drawCache = new Map();
    const key = String(Math.max(2, Math.round(count || 2))) + ':' + (payload.loaded ? 'detail' : 'base');
    if (payload.drawCache.has(key)) return payload.drawCache.get(key);
    const peaks = resampleSeries(waveformSeries(payload, 'detail'), count);
    const envelope = waveformEnvelopeSeries(payload, 'detail');
    const bands = payload.bands || null;
    const series = {
      peaks,
      low: bands && bands.low && bands.low.length ? resampleSeries(bands.low, count) : peaks,
      mid: bands && bands.mid && bands.mid.length ? resampleSeries(bands.mid, count) : peaks,
      high: bands && bands.high && bands.high.length ? resampleSeries(bands.high, count) : peaks,
      min: envelope && envelope.min && envelope.min.length ? resampleSignedSeries(envelope.min, count) : [],
      max: envelope && envelope.max && envelope.max.length ? resampleSignedSeries(envelope.max, count) : [],
    };
    payload.drawCache.set(key, series);
    return series;
  }
  function drawDetailedWaveform(ctx, payload, width, height, progressX) {
    const count = Math.max(180, Math.floor(width * 1.35));
    const series = cachedDrawSeries(payload, count);
    const peaks = series.peaks;
    const low = series.low;
    const mid = series.mid;
    const high = series.high;
    if (!peaks.length) {
      drawWaveformBars(ctx, [], width, height, progressX);
      return;
    }
    drawWaveformEnvelope(ctx, low, width, height, 'rgba(31,90,240,.68)', 1.0, 1.03);
    drawWaveformEnvelope(ctx, mid, width, height, 'rgba(211,128,39,.68)', 0.74, 0.92);
    drawWaveformEnvelope(ctx, high, width, height, 'rgba(248,242,226,.82)', 0.38, 0.72);
    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.beginPath();
    ctx.rect(0, 0, clamp(progressX, 0, width), height);
    ctx.clip();
    drawWaveformEnvelope(ctx, low, width, height, 'rgba(29,185,84,.42)', 1.0, 1.02);
    drawWaveformEnvelope(ctx, mid, width, height, 'rgba(94,234,212,.32)', 0.74, 0.9);
    ctx.restore();
  }
  async function ensureDetailedWaveform(record, { redrawMain=false, redrawTransition=false } = {}) {
    record = canonicalRecord(record);
    if (!record || !window.fetch) return null;
    const payload = waveformPayload(record);
    const url = waveformDetailUri(record);
    if (!url || payload.loaded) return payload;
    if (payload.loading && payload.promise) return payload.promise;
    payload.loading = true;
    payload.error = '';
    payload.promise = fetch(url)
      .then(res => {
        if (!res.ok) throw new Error('Waveform fetch failed');
        return res.json();
      })
      .then(data => applyWaveformDetail(payload, data))
      .catch(err => {
        payload.error = 'Detailed waveform unavailable';
        return payload;
      })
      .finally(() => {
        payload.loading = false;
        if (waveformState.idx === Number(record.idx)) {
          waveformState.loading = false;
          waveformState.error = payload.error || '';
          waveformState.payload = payload;
          if (redrawMain) drawMainWaveform();
        }
        if (redrawTransition) drawTransitionEditor();
      });
    return payload.promise;
  }
  async function loadMainWaveform(record) {
    record = canonicalRecord(record);
    if (!record || !els.songWaveform) return;
    const url = waveformDetailUri(record);
    const payload = waveformPayload(record);
    waveformState = { idx: Number(record.idx), url, payload, loading: false, error: payload.error || '' };
    drawMainWaveform();
    if (url && !payload.loaded) {
      waveformState.loading = true;
      drawMainWaveform();
      ensureDetailedWaveform(record, { redrawMain: true, redrawTransition: true });
    }
  }
  function drawWaveformBars(ctx, peaks, width, height, progressX) {
    const normalized = normalizeWaveformValues(peaks);
    const bars = normalized.length ? normalized : Array.from({ length: 120 }, (_, i) => 0.18 + 0.12 * Math.sin(i * 0.43));
    const gap = width < 260 ? 1 : 2;
    const barW = Math.max(1, Math.floor(width / bars.length) - gap);
    const step = width / bars.length;
    for (let i = 0; i < bars.length; i += 1) {
      const x = i * step;
      const h = Math.max(4, bars[i] * (height - 14));
      const y = (height - h) * 0.5;
      const active = x <= progressX;
      ctx.fillStyle = active ? '#1db954' : 'rgba(148, 163, 184, 0.52)';
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(x, y, barW, h, Math.min(3, barW * 0.5));
        ctx.fill();
      } else {
        ctx.fillRect(x, y, barW, h);
      }
    }
  }
  function drawMainWaveform() {
    const canvas = els.songWaveform;
    if (!canvas) return;
    const record = selectedIdx === null ? null : byIdx.get(selectedIdx);
    const rect = canvas.getBoundingClientRect();
    const widthCss = Math.max(1, rect.width || canvas.clientWidth || 320);
    const heightCss = Math.max(1, rect.height || 54);
    const dpr = window.devicePixelRatio || 1;
    const width = Math.floor(widthCss * dpr);
    const height = Math.floor(heightCss * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, widthCss, heightCss);
    ctx.fillStyle = '#0b1020';
    ctx.fillRect(0, 0, widthCss, heightCss);
    if (!record) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText('Select a track', 12, Math.round(heightCss / 2) + 4);
      return;
    }
    const duration = audioDuration(record);
    const current = previewAudioIdx === Number(record.idx) && els.audio ? Number(els.audio.currentTime || 0) : previewStart(record);
    const progress = Number.isFinite(duration) && duration > 0 ? clamp(current / duration, 0, 1) : 0;
    const progressX = progress * widthCss;
    const activeWaveform = waveformState.idx === Number(record.idx) ? waveformState : { payload: waveformPayload(record), loading: false, error: '' };
    drawDetailedWaveform(ctx, activeWaveform.payload || waveformPayload(record), widthCss, heightCss, progressX);
    const markerTime = previewStart(record);
    if (Number.isFinite(duration) && duration > 0 && markerTime > 0) {
      const x = clamp(markerTime / duration, 0, 1) * widthCss;
      ctx.save();
      ctx.strokeStyle = 'rgba(251, 191, 36, .9)';
      ctx.lineWidth = 1.4;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, 5);
      ctx.lineTo(x, heightCss - 5);
      ctx.stroke();
      ctx.restore();
    }
    ctx.strokeStyle = 'rgba(248, 250, 252, .88)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(progressX, 4);
    ctx.lineTo(progressX, heightCss - 4);
    ctx.stroke();
    if (activeWaveform.loading || activeWaveform.error) {
      ctx.fillStyle = 'rgba(15, 23, 42, .78)';
      ctx.fillRect(0, 0, widthCss, heightCss);
      ctx.fillStyle = activeWaveform.error ? '#fbbf24' : '#cbd5e1';
      ctx.font = '12px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText(activeWaveform.error || 'Loading waveform...', 12, Math.round(heightCss / 2) + 4);
    }
  }
  function drawRowWaveform(canvas, record) {
    record = canonicalRecord(record);
    if (!canvas || !record) return;
    const rect = canvas.getBoundingClientRect();
    const widthCss = Math.max(1, rect.width || canvas.clientWidth || 220);
    const heightCss = Math.max(1, rect.height || 30);
    const dpr = window.devicePixelRatio || 1;
    const width = Math.floor(widthCss * dpr);
    const height = Math.floor(heightCss * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, widthCss, heightCss);
    ctx.fillStyle = '#0b1020';
    ctx.fillRect(0, 0, widthCss, heightCss);
    const duration = audioDuration(record);
    const current = clamp(rowScrubValue(record), 0, Number.isFinite(duration) && duration > 0 ? duration : Number(record.duration_seconds) || 1);
    const progress = Number.isFinite(duration) && duration > 0 ? clamp(current / duration, 0, 1) : 0;
    const peaks = waveformSeries(waveformPayload(record), 'row');
    drawWaveformBars(ctx, peaks, widthCss, heightCss, progress * widthCss);
    const markerTime = previewStart(record);
    if (Number.isFinite(duration) && duration > 0 && markerTime > 0) {
      const x = clamp(markerTime / duration, 0, 1) * widthCss;
      ctx.save();
      ctx.strokeStyle = 'rgba(251, 191, 36, .78)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, 4);
      ctx.lineTo(x, heightCss - 4);
      ctx.stroke();
      ctx.restore();
    }
    if (previewAudioIdx === Number(record.idx)) {
      const x = progress * widthCss;
      ctx.strokeStyle = previewAudioPlaying ? 'rgba(248,250,252,.96)' : 'rgba(203,213,225,.72)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(x, 4);
      ctx.lineTo(x, heightCss - 4);
      ctx.stroke();
    }
  }
  function drawLibraryWaveforms() {
    document.querySelectorAll('[data-library-waveform-idx]').forEach(canvas => {
      const idx = Number(canvas.getAttribute('data-library-waveform-idx'));
      const record = byIdx.get(idx);
      if (record) drawRowWaveform(canvas, record);
    });
  }
  function seekLibraryWaveformFromEvent(ev, idx, { play=false } = {}) {
    idx = Number(idx);
    const record = byIdx.get(idx);
    if (!record) return;
    const duration = audioDuration(record);
    if (!Number.isFinite(duration) || duration <= 0) return;
    const canvas = ev.currentTarget && ev.currentTarget.getAttribute && ev.currentTarget.getAttribute('data-library-waveform-idx')
      ? ev.currentTarget
      : document.querySelector('[data-library-waveform-idx="' + idx + '"]');
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = clamp((ev.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const target = x * duration;
    rowScrubPositions.set(idx, target);
    if (play) {
      playRecordAt(record, target, 'library');
    } else {
      if (previewAudioIdx === idx && els.audio) seekSharedAudio(target);
      syncAudioUi();
    }
  }
  function seekMainWaveformFromEvent(ev, shouldPlay=true) {
    if (selectedIdx === null || !els.songWaveform) return;
    const record = byIdx.get(selectedIdx);
    const duration = audioDuration(record);
    if (!record || !Number.isFinite(duration) || duration <= 0) return;
    const rect = els.songWaveform.getBoundingClientRect();
    const x = clamp((ev.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const target = x * duration;
    if (shouldPlay) playRecordAt(record, target, 'main');
    else {
      seekSharedAudio(target);
      syncAudioUi();
    }
  }
  function smoothstep(x) {
    return x * x * (3 - (2 * x));
  }
  function previewValueFor(mode, deck, x) {
    const half = x < 0.5;
    if (mode === 'volume:crossfade') return deck === 'a' ? Math.cos(x * Math.PI * 0.5) : Math.sin(x * Math.PI * 0.5);
    if (mode === 'volume:overlap-crossfade') {
      const floor = Math.pow(10, -6 / 20);
      const curve = deck === 'a' ? Math.cos(x * Math.PI * 0.5) : Math.sin(x * Math.PI * 0.5);
      return floor + ((1 - floor) * curve);
    }
    if (mode === 'volume:smooth-crossfade') {
      const t = smoothstep(x);
      return deck === 'a' ? Math.sqrt(Math.max(0, 1 - t)) : Math.sqrt(Math.max(0, t));
    }
    if (mode === 'volume:overlap') return 1;
    if (mode === 'volume:fade-in-fade-out') return deck === 'a' ? (half ? 1 : 1 - ((x - 0.5) * 2)) : (half ? x * 2 : 1);
    if (mode === 'volume:cut-in-fade-out') return deck === 'a' ? (half ? 1 : 1 - ((x - 0.5) * 2)) : 1;
    if (mode === 'volume:fade-in-cut-out') return deck === 'a' ? 1 : (half ? x * 2 : 1);
    if (mode === 'volume:center-cut') return deck === 'a' ? (half ? 1 : 0) : (half ? 0 : 1);
    if (mode === 'eq:none') return 1;
    if (mode === 'eq:start-bass-swap') return deck === 'a' ? 0.15 : 1;
    if (mode === 'eq:end-bass-swap') return deck === 'b' ? 0.15 : 1;
    if (mode === 'eq:center-bass-swap') return deck === 'a' ? (half ? 1 : 0.15) : (half ? 0.15 : 1);
    if (mode === 'eq:long-bass-cut') return 0.15;
    if (mode === 'filter:none') return null;
    if (mode === 'filter:low-pass-filter-out') return deck === 'a' ? 1 - x : null;
    if (mode === 'filter:low-pass-filter-in') return deck === 'b' ? x : null;
    if (mode === 'filter:high-pass-filter-out') return deck === 'a' ? x : null;
    if (mode === 'filter:high-pass-filter-in') return deck === 'b' ? 1 - x : null;
    return null;
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
    if (mode === 'energy') return { min: 1, max: 9, title: els.energySource.value === 'glm' ? 'Auto energy' : 'Tagged energy' };
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
  const simplexVertices = {
    tempo: { x: 180, y: 34 },
    groove: { x: 44, y: 270 },
    chroma: { x: 316, y: 270 },
  };
  function mixWeightsRaw() {
    const te = Math.max(0, Number(els.weightTempo ? els.weightTempo.value : 0));
    const gr = Math.max(0, Number(els.weightGroove ? els.weightGroove.value : 0));
    const ch = Math.max(0, Number(els.weightChroma ? els.weightChroma.value : 0));
    const s = te + gr + ch;
    if (s <= 1e-12) return { tempo: 1/3, groove: 1/3, chroma: 1/3 };
    return { tempo: te / s, groove: gr / s, chroma: ch / s };
  }
  function setMixSliders(mix) {
    if (els.weightTempo) els.weightTempo.value = String(clamp(mix.tempo, 0, 1));
    if (els.weightGroove) els.weightGroove.value = String(clamp(mix.groove, 0, 1));
    if (els.weightChroma) els.weightChroma.value = String(clamp(mix.chroma, 0, 1));
  }
  function simplexPoint(mix) {
    return {
      x: mix.tempo * simplexVertices.tempo.x + mix.groove * simplexVertices.groove.x + mix.chroma * simplexVertices.chroma.x,
      y: mix.tempo * simplexVertices.tempo.y + mix.groove * simplexVertices.groove.y + mix.chroma * simplexVertices.chroma.y,
    };
  }
  function updateSimplexHandle(mix) {
    if (!els.simplexHandle) return;
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
    return { tempo: te / s, groove: gr / s, chroma: ch / s };
  }
  function eventToSvgPoint(ev) {
    const rect = els.simplex.getBoundingClientRect();
    return { x: (ev.clientX - rect.left) * 360 / Math.max(1, rect.width), y: (ev.clientY - rect.top) * 320 / Math.max(1, rect.height) };
  }
  function setWeightsFromSimplexEvent(ev) {
    const p = eventToSvgPoint(ev);
    setMixSliders(simplexWeightsFromPoint(p.x, p.y));
    renderAll();
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
        colorbar: { title: { text: extent.title }, thickness: 14, x: 1.02, y: 0.52, len: 0.72 },
        line: { color: 'black', width: 0.5 },
      };
      Plotly.restyle(plot, { marker: [marker], showlegend: [false] }, [traceIdx]);
    });
    Plotly.relayout(plot, { 'legend.title.text': mode === 'energy' ? 'Energy' : 'Tempo' });
  }
  function weights() {
    if (config.control_mode === 'genre-mixability') {
      const style = clamp(Number(els.weightStyle ? els.weightStyle.value : 0.45), 0, 1);
      const mix = mixWeightsRaw();
      const m = 1 - style;
      return { maest: style, tempo: m * mix.tempo, groove: m * mix.groove, chroma: m * mix.chroma, mix };
    }
    const ma = Math.max(0, Number(els.weightMaest ? els.weightMaest.value : 0));
    const ch = Math.max(0, Number(els.weightChroma ? els.weightChroma.value : 0));
    const te = Math.max(0, Number(els.weightTempo ? els.weightTempo.value : 0));
    const s = ma + ch + te;
    if (s <= 1e-12) return { maest: 1/3, chroma: 1/3, tempo: 1/3, groove: 0 };
    return { maest: ma/s, chroma: ch/s, tempo: te/s, groove: 0 };
  }
  function layoutInterpolatedPoints(w) {
    if (!Array.isArray(layoutEntries) || !layoutEntries.length) {
      return records.map(r => idxToPoint[String(r.idx)] || [0, 0]);
    }
    const target = [w.maest, w.tempo, w.groove, w.chroma];
    const ranked = layoutEntries.map(entry => {
      const d2 = entry.weights.reduce((acc, val, idx) => acc + Math.pow(Number(val) - target[idx], 2), 0);
      return { entry, d2 };
    }).sort((a,b) => a.d2 - b.d2).slice(0, 12);
    if (!ranked.length) return records.map(r => idxToPoint[String(r.idx)] || [0, 0]);
    if (ranked[0].d2 <= 1e-12) return ranked[0].entry.points;
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
  function currentPoint(idx) {
    return currentPoints[Number(idx)] || idxToPoint[String(idx)] || null;
  }
  function recordMapTrails(previousPoints, nextPoints) {
    if (els.mapEffectsEnabled && !els.mapEffectsEnabled.checked) return;
    if (!Array.isArray(previousPoints) || !Array.isArray(nextPoints) || previousPoints.length !== nextPoints.length) return;
    const now = performance.now();
    const fresh = [];
    for (let i = 0; i < nextPoints.length; i += 1) {
      const a = previousPoints[i];
      const b = nextPoints[i];
      if (!a || !b) continue;
      const dx = Number(b[0]) - Number(a[0]);
      const dy = Number(b[1]) - Number(a[1]);
      if (!Number.isFinite(dx) || !Number.isFinite(dy)) continue;
      if ((dx * dx + dy * dy) < 1e-6) continue;
      fresh.push({ from: [Number(a[0]), Number(a[1])], to: [Number(b[0]), Number(b[1])], t0: now, life: 780 });
      if (fresh.length >= 120) break;
    }
    if (!fresh.length) return;
    mapTrailSegments = mapTrailSegments.concat(fresh).slice(-220);
    ensureMapEffectsLoop();
  }
  function plotPointPx(pt) {
    if (!plot || !els.mapEffects || !plot._fullLayout || !pt) return null;
    const xa = plot._fullLayout.xaxis;
    const ya = plot._fullLayout.yaxis;
    const size = plot._fullLayout._size || { l: 0, t: 0 };
    if (!xa || !ya || typeof xa.l2p !== 'function' || typeof ya.l2p !== 'function') return null;
    const plotRect = plot.getBoundingClientRect();
    const canvasRect = els.mapEffects.getBoundingClientRect();
    const x = plotRect.left - canvasRect.left + Number(size.l || 0) + xa.l2p(Number(pt[0]));
    const y = plotRect.top - canvasRect.top + Number(size.t || 0) + ya.l2p(Number(pt[1]));
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    return { x, y };
  }
  function plotClickNearTrackPoint(ev, radius=18) {
    if (!ev || !els.mapEffects) return false;
    const canvasRect = els.mapEffects.getBoundingClientRect();
    const x = Number(ev.clientX) - canvasRect.left;
    const y = Number(ev.clientY) - canvasRect.top;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
    const r2 = radius * radius;
    for (let i = 0; i < records.length; i += 1) {
      const p = plotPointPx(currentPoint(i));
      if (!p) continue;
      const dx = p.x - x;
      const dy = p.y - y;
      if ((dx * dx + dy * dy) <= r2) return true;
    }
    return false;
  }
  function drawGlow(ctx, p, radius, color, alpha) {
    if (!p) return;
    const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius);
    g.addColorStop(0, color.replace('__A__', String(alpha)));
    g.addColorStop(0.42, color.replace('__A__', String(alpha * 0.36)));
    g.addColorStop(1, color.replace('__A__', '0'));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  function drawPulseRing(ctx, p, radius, color, phase, label) {
    if (!p) return;
    const pulse = 0.5 + (0.5 * Math.sin(phase));
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 16 + pulse * 12;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.globalAlpha = 0.62 + pulse * 0.22;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius + pulse * 5, 0, Math.PI * 2);
    ctx.stroke();
    if (label) {
      ctx.shadowBlur = 8;
      ctx.fillStyle = color;
      ctx.font = '700 11px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, p.x, p.y - radius - 10);
    }
    ctx.restore();
  }
  function drawFlowLine(ctx, a, b, color, strength, phase, dotCount=1) {
    if (!a || !b) return;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.hypot(dx, dy);
    if (!Number.isFinite(dist) || dist < 2) return;
    ctx.save();
    ctx.lineCap = 'round';
    ctx.strokeStyle = color.replace('__A__', String(0.08 + strength * 0.28));
    ctx.lineWidth = 1.2 + strength * 2.2;
    ctx.shadowColor = color.replace('__A__', '0.55');
    ctx.shadowBlur = 10 + strength * 12;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    for (let j = 0; j < dotCount; j += 1) {
      const t = (phase + j / Math.max(1, dotCount)) % 1;
      const ease = t * t * (3 - 2 * t);
      const x = a.x + dx * ease;
      const y = a.y + dy * ease;
      ctx.fillStyle = color.replace('__A__', String(0.45 + strength * 0.42));
      ctx.shadowBlur = 18 + strength * 10;
      ctx.beginPath();
      ctx.arc(x, y, 2.2 + strength * 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
  function drawBackgroundField(ctx, width, height, time) {
    ctx.save();
    const t = time / 1000;
    const cx = width * 0.52;
    const cy = height * 0.46;
    ctx.globalCompositeOperation = 'screen';
    ctx.strokeStyle = 'rgba(45, 212, 191, 0.045)';
    ctx.lineWidth = 1;
    for (let y = 34; y < height; y += 42) {
      const drift = Math.sin(t * 0.32 + y * 0.015) * 16;
      ctx.beginPath();
      for (let x = -40; x <= width + 40; x += 24) {
        const yy = y + Math.sin((x * 0.012) + t + y * 0.01) * 5;
        if (x === -40) ctx.moveTo(x + drift, yy);
        else ctx.lineTo(x + drift, yy);
      }
      ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.026)';
    for (let i = 0; i < 18; i += 1) {
      const y = ((i * 67 + (t * 12)) % (height + 120)) - 60;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y + Math.sin(t + i) * 20);
      ctx.stroke();
    }
    const rings = [0.35, 0.53, 0.72, 0.91];
    rings.forEach((r, i) => {
      const wobble = Math.sin(t * 0.21 + i) * 8;
      ctx.strokeStyle = 'rgba(148, 163, 184, ' + (0.028 - i * 0.004).toFixed(3) + ')';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.ellipse(cx, cy, width * r * 0.58 + wobble, height * r * 0.36, -0.12, 0, Math.PI * 2);
      ctx.stroke();
    });
    ctx.restore();
  }
  function strongestConnectionPairs(limit=42) {
    if (strongestPairsCache) return strongestPairsCache.slice(0, limit);
    const pairs = [];
    const seen = new Set();
    Object.keys(simMap || {}).forEach(srcKey => {
      const src = Number(srcKey);
      const group = simMap[srcKey] || {};
      const candidates = Array.isArray(group.candidates) ? group.candidates : [];
      candidates.slice(0, 4).forEach(c => {
        const dst = Number(c.idx);
        if (!Number.isFinite(src) || !Number.isFinite(dst) || src === dst) return;
        const key = src < dst ? src + ':' + dst : dst + ':' + src;
        if (seen.has(key)) return;
        seen.add(key);
        const score = Number(c.maest_score_norm ?? c.maest_similarity ?? c.compatibility_score ?? 0);
        pairs.push({ src, dst, score: Number.isFinite(score) ? score : 0 });
      });
    });
    pairs.sort((a, b) => b.score - a.score);
    strongestPairsCache = pairs;
    return strongestPairsCache.slice(0, limit);
  }
  function drawStrongestConnections(ctx) {
    const pairs = strongestConnectionPairs();
    pairs.forEach((pair, i) => {
      const a = plotPointPx(currentPoint(pair.src));
      const b = plotPointPx(currentPoint(pair.dst));
      if (!a || !b) return;
      const alpha = 0.026 + (1 - i / Math.max(1, pairs.length)) * 0.034;
      ctx.save();
      ctx.strokeStyle = 'rgba(248, 250, 252, ' + alpha.toFixed(3) + ')';
      ctx.lineWidth = 0.55;
      ctx.setLineDash([2, 8]);
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.restore();
    });
  }
  function drawSequenceTransitionEffects(ctx, time) {
    const pts = sequence
      .map(idx => idx === null ? null : plotPointPx(currentPoint(idx)))
      .filter(Boolean);
    for (let i = 0; i < pts.length - 1; i += 1) {
      drawFlowLine(ctx, pts[i], pts[i + 1], 'rgba(45, 212, 191, __A__)', 0.34, (time / 2100 + i * 0.19) % 1, 1);
    }
  }
  function drawMapEffectsFrame(time) {
    mapEffectsFrame = null;
    const canvas = els.mapEffects;
    if (!canvas || !plot) return;
    if (els.mapEffectsEnabled && !els.mapEffectsEnabled.checked) {
      const ctxOff = canvas.getContext('2d');
      if (ctxOff) ctxOff.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 20 || rect.height < 20) {
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    drawBackgroundField(ctx, rect.width, rect.height, time);
    drawStrongestConnections(ctx);
    drawSequenceTransitionEffects(ctx, time);

    mapTrailSegments = mapTrailSegments.filter(seg => (time - seg.t0) < seg.life);
    for (const seg of mapTrailSegments) {
      const a = plotPointPx(seg.from);
      const b = plotPointPx(seg.to);
      if (!a || !b) continue;
      const age = clamp((time - seg.t0) / seg.life, 0, 1);
      const alpha = (1 - age) * 0.22;
      ctx.save();
      ctx.strokeStyle = 'rgba(125, 211, 252, ' + alpha.toFixed(3) + ')';
      ctx.lineWidth = 1.1;
      ctx.shadowColor = 'rgba(45, 212, 191, ' + (alpha * 1.4).toFixed(3) + ')';
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.restore();
    }

    const rows = rankedRecommendations().slice(0, 12);
    const src = selectedIdx === null ? null : plotPointPx(currentPoint(selectedIdx));
    if (src && rows.length) {
      const finals = rows.map(r => Number(r.finalScore)).filter(Number.isFinite);
      const minScore = finals.length ? Math.min(...finals) : 0;
      const maxScore = finals.length ? Math.max(...finals) : 1;
      rows.forEach((row, i) => {
        const dst = plotPointPx(currentPoint(row.idx));
        if (!dst) return;
        const score = Number(row.finalScore);
        const scoreStrength = Number.isFinite(score) && Math.abs(maxScore - minScore) > 1e-9
          ? clamp((score - minScore) / (maxScore - minScore), 0, 1)
          : 1 - (i / Math.max(1, rows.length));
        const rankStrength = 1 - (i / Math.max(1, rows.length));
        const strength = clamp(0.35 * scoreStrength + 0.65 * rankStrength, 0, 1);
        drawFlowLine(ctx, src, dst, 'rgba(251, 191, 36, __A__)', strength * 0.72, (time / 1750 + i * 0.071) % 1, i < 4 ? 2 : 1);
        drawGlow(ctx, dst, 14 + strength * 16, 'rgba(251, 191, 36, __A__)', 0.06 + strength * 0.13);
      });
    }

    const fromP = transitionFromIdx === null ? null : plotPointPx(currentPoint(transitionFromIdx));
    const toP = transitionToIdx === null ? null : plotPointPx(currentPoint(transitionToIdx));
    if (fromP && toP) drawFlowLine(ctx, fromP, toP, 'rgba(45, 212, 191, __A__)', 1, (time / 1300) % 1, 3);
    drawPulseRing(ctx, selectedIdx === null ? null : plotPointPx(currentPoint(selectedIdx)), 13, '#f8fafc', time / 520, '');
    drawPulseRing(ctx, fromP, 16, '#34d399', time / 470, '');
    drawPulseRing(ctx, toP, 16, '#fbbf24', time / 520 + 1.2, '');

    if (activePane === 'explore') {
      ensureMapEffectsLoop();
    }
  }
  function ensureMapEffectsLoop() {
    if (els.mapEffectsEnabled && !els.mapEffectsEnabled.checked) return;
    if (mapEffectsFrame !== null || !window.requestAnimationFrame) return;
    mapEffectsFrame = window.requestAnimationFrame(drawMapEffectsFrame);
  }
  function updatePointCoordinates() {
    if (!window.Plotly || !plot || !baseTraceIndices.length) return;
    const byGenre = new Map();
    for (const r of records) {
      const pt = currentPoint(r.idx);
      if (!pt) continue;
      const g = String(r.genre);
      if (!byGenre.has(g)) byGenre.set(g, { x: [], y: [] });
      byGenre.get(g).x.push(Number(pt[0]));
      byGenre.get(g).y.push(Number(pt[1]));
    }
    for (const traceIdx of baseTraceIndices) {
      const trace = plot.data[traceIdx] || {};
      const vals = byGenre.get(String(trace.name));
      if (vals) Plotly.restyle(plot, { x: [vals.x], y: [vals.y] }, [traceIdx]);
    }
  }
  function updateWeightLabels() {
    const w = weights();
    if (els.weightStyleVal) els.weightStyleVal.textContent = fmt(w.maest, 2);
    if (els.weightMaestVal) els.weightMaestVal.textContent = fmt(w.maest, 2);
    if (els.weightChromaVal) els.weightChromaVal.textContent = fmt(w.chroma, 2);
    if (els.weightTempoVal) els.weightTempoVal.textContent = fmt(w.tempo, 2);
    if (els.weightGrooveVal) els.weightGrooveVal.textContent = fmt(w.groove, 2);
    if (w.mix) updateSimplexHandle(w.mix);
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
  function camelotNumber(value) {
    const m = String(value || '').trim().toUpperCase().match(/^(\d{1,2})[AB]?/);
    if (!m) return null;
    const n = Number(m[1]);
    return Number.isFinite(n) ? n : null;
  }
  function recommendationFilterHtml() {
    return '<div class="recommendation-filterbar">' +
      '<label><span><input data-rec-filter="sameKey" type="checkbox"' + (recFilters.sameKey ? ' checked' : '') + '> Key family</span></label>' +
      '<label>BPM +/-<input data-rec-filter="bpmRange" type="number" min="0" max="60" step="1" placeholder="Any" value="' + esc(recFilters.bpmRange) + '"></label>' +
      '<label>Energy +/-<input data-rec-filter="energyRange" type="number" min="0" max="8" step="0.25" placeholder="Any" value="' + esc(recFilters.energyRange) + '"></label>' +
      '<label><span><input data-rec-filter="excludeUsed" type="checkbox"' + (recFilters.excludeUsed ? ' checked' : '') + '> Exclude used</span></label>' +
      '<label>Genre<select data-rec-filter="genreMode">' +
      optionHtml('any', 'Any', recFilters.genreMode) +
      optionHtml('same', 'Same', recFilters.genreMode) +
      optionHtml('different', 'Different', recFilters.genreMode) +
      '</select></label>' +
      '</div>';
  }
  function updateRecFiltersFromElement(el) {
    const key = el && el.getAttribute && el.getAttribute('data-rec-filter');
    if (!key) return false;
    if (key === 'sameKey' || key === 'excludeUsed') recFilters[key] = !!el.checked;
    else if (key === 'genreMode') recFilters.genreMode = el.value || 'any';
    else recFilters[key] = el.value || '';
    return true;
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
  function setTransitionEndpoint(kind, idx, { toggle=true } = {}) {
    idx = Number(idx);
    if (!Number.isFinite(idx) || !byIdx.has(idx)) return;
    resetTransitionRenderState();
    previewFormState.from_cue = '';
    previewFormState.to_cue = '';
    if (kind === 'out') {
      transitionFromIdx = toggle && transitionFromIdx === idx ? null : idx;
      if (transitionToIdx === idx) transitionToIdx = null;
    } else {
      transitionToIdx = toggle && transitionToIdx === idx ? null : idx;
      if (transitionFromIdx === idx) transitionFromIdx = null;
    }
    renderAll();
  }
  function assignTransitionFromDoubleClick(idx) {
    idx = Number(idx);
    if (!Number.isFinite(idx) || !byIdx.has(idx)) return;
    const now = Date.now();
    if (lastPointDoubleClickIdx === idx && (now - lastPointDoubleClickMs) < 900) return;
    lastPointDoubleClickIdx = idx;
    lastPointDoubleClickMs = now;
    if (transitionFromIdx === null) setTransitionEndpoint('out', idx, { toggle: false });
    else setTransitionEndpoint('in', idx, { toggle: false });
  }
  function handlePlotDoubleClick(fromPoint=false) {
    const now = Date.now();
    if ((now - lastPointDoubleClickMs) < 900) return false;
    if (fromPoint && lastClickedIdx !== null && (now - lastClickMs) < 800) {
      assignTransitionFromDoubleClick(lastClickedIdx);
      return false;
    }
    if (transitionFromIdx !== null || transitionToIdx !== null) clearTransitionPair();
    return false;
  }
  function swapTransitionPair() {
    const oldFrom = transitionFromIdx;
    transitionFromIdx = transitionToIdx;
    transitionToIdx = oldFrom;
    resetTransitionRenderState();
    previewFormState.from_cue = '';
    previewFormState.to_cue = '';
    renderAll();
  }
  function clearTransitionPair() {
    transitionFromIdx = null;
    transitionToIdx = null;
    lastClickedIdx = null;
    lastClickMs = 0;
    lastPointDoubleClickMs = 0;
    lastPointDoubleClickIdx = null;
    resetTransitionRenderState();
    previewFormState.from_cue = '';
    previewFormState.to_cue = '';
    renderAll();
  }
  function setSequenceLength(n) {
    n = clamp(Math.round(Number(n || 10)), 2, 40);
    const old = sequence.slice();
    const oldTargets = targetValues.slice();
    sequenceLength = n;
    sequence = Array.from({ length: n }, (_, i) => i < old.length ? old[i] : null);
    targetValues = Array.from({ length: n }, (_, i) => i < oldTargets.length ? oldTargets[i] : null);
    selectedSlot = selectedSlot !== null && selectedSlot < n ? selectedSlot : null;
    renderAll();
  }
  function setTargetSlotValue(slot, value, shouldRender=true) {
    slot = Math.round(Number(slot));
    value = clamp(Number(value), 1, 9);
    if (!Number.isFinite(slot) || !Number.isFinite(value)) return;
    targetValues[slot] = value;
    selectedSlot = slot;
    if (shouldRender) renderAll();
  }
  function targetAnchors() {
    const anchors = [];
    for (let slot = 0; slot < targetValues.length; slot += 1) {
      const val = Number(targetValues[slot]);
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
    const useNorm = config.control_mode === 'genre-mixability';
    const styleScore = useNorm ? Number(c.maest_score_norm ?? c.maest_similarity ?? 0) : Number(c.maest_similarity || 0);
    const tempoScore = useNorm ? Number(c.tempo_score_norm ?? c.tempo_similarity ?? 0) : Number(c.tempo_similarity || 0);
    const grooveScore = useNorm ? Number(c.groove_score_norm ?? c.groove_similarity ?? 0) : 0;
    const keyScore = useNorm ? Number(c.chroma_score_norm ?? c.chroma_similarity ?? 0) : Number(c.chroma_similarity || 0);
    const baseline = w.maest * styleScore + w.tempo * tempoScore + w.groove * grooveScore + w.chroma * keyScore;
    const record = byIdx.get(Number(c.idx));
    const e = energyOf(record);
    const target = targetCurve()[slot] ?? 5;
    const rawError = Number.isFinite(e) ? e - target : 0;
    const normSq = Math.pow(rawError / 8, 2);
    const penalty = Number(els.penaltyScale.value || 0) * normSq;
    const energyScore = 1 - normSq;
    return { baseline, styleScore, tempoScore, grooveScore, keyScore, energy: e, target, rawError, normSq, penalty, energyScore, finalScore: baseline - penalty };
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
      groove_similarity: NaN,
      maest_score_norm: NaN,
      chroma_score_norm: NaN,
      tempo_score_norm: NaN,
      groove_score_norm: NaN,
      baseline: NaN,
      styleScore: NaN,
      tempoScore: NaN,
      grooveScore: NaN,
      keyScore: NaN,
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
    const srcRecord = byIdx.get(selectedIdx);
    const srcKeyFamily = camelotNumber(srcRecord && srcRecord.key);
    const bpmRange = Number(recFilters.bpmRange);
    const hasBpmRange = String(recFilters.bpmRange || '').trim() !== '' && Number.isFinite(bpmRange) && bpmRange >= 0;
    const energyRange = Number(recFilters.energyRange);
    const hasEnergyRange = String(recFilters.energyRange || '').trim() !== '' && Number.isFinite(energyRange) && energyRange >= 0;
    const targetEnergy = targetCurve()[slot] ?? 5;
    const genreMode = recFilters.genreMode || 'any';
    const srcGenre = String((srcRecord && (srcRecord.raw_genre || srcRecord.genre)) || '').toLowerCase();
    const entry = simMap[String(selectedIdx)] || {};
    const candidates = Array.isArray(entry.candidates) ? entry.candidates : [];
    const rows = [];
    for (const c of candidates) {
      const idx = Number(c.idx);
      if (!Number.isFinite(idx) || idx === selectedIdx) continue;
      if (recFilters.excludeUsed && used.has(idx)) continue;
      const record = byIdx.get(idx);
      if (!record) continue;
      if (recFilters.sameKey && srcKeyFamily !== null && camelotNumber(record.key) !== srcKeyFamily) continue;
      if (hasBpmRange) {
        const srcBpm = Number(srcRecord && srcRecord.est_bpm);
        const candBpm = Number(record.est_bpm);
        if (!Number.isFinite(srcBpm) || !Number.isFinite(candBpm) || Math.abs(candBpm - srcBpm) > bpmRange) continue;
      }
      if (hasEnergyRange && Math.abs(energyOf(record) - targetEnergy) > energyRange) continue;
      if (genreMode !== 'any') {
        const candGenre = String(record.raw_genre || record.genre || '').toLowerCase();
        if (genreMode === 'same' && candGenre !== srcGenre) continue;
        if (genreMode === 'different' && candGenre === srcGenre) continue;
      }
      const s = scoreCandidate(c, slot);
      rows.push({ ...c, ...s, slot });
    }
    rows.sort((a,b) => b.finalScore - a.finalScore);
    return rows;
  }
  function updatePacmapOverlays() {
    const hoverMatchesSelection = hoveredIdx !== null && (
      hoveredIdx === selectedIdx ||
      hoveredIdx === transitionFromIdx ||
      hoveredIdx === transitionToIdx
    );
    const hoveredPt = hoveredIdx === null || hoverMatchesSelection ? null : currentPoint(hoveredIdx);
    restyleTrace('Hovered', {
      x: [hoveredPt ? [hoveredPt[0]] : []],
      y: [hoveredPt ? [hoveredPt[1]] : []],
    });
    const selectedPt = selectedIdx === null ? null : currentPoint(selectedIdx);
    restyleTrace('Selected', {
      x: [selectedPt ? [selectedPt[0]] : []],
      y: [selectedPt ? [selectedPt[1]] : []],
    });
    const fromPt = transitionFromIdx === null ? null : currentPoint(transitionFromIdx);
    const toPt = transitionToIdx === null ? null : currentPoint(transitionToIdx);
    restyleTrace('Track 1', {
      x: [fromPt ? [fromPt[0]] : []],
      y: [fromPt ? [fromPt[1]] : []],
      text: [fromPt ? ['T1'] : []],
    });
    restyleTrace('Track 2', {
      x: [toPt ? [toPt[0]] : []],
      y: [toPt ? [toPt[1]] : []],
      text: [toPt ? ['T2'] : []],
    });
    restyleTrace('Transition pair', {
      x: [fromPt && toPt ? [fromPt[0], toPt[0]] : []],
      y: [fromPt && toPt ? [fromPt[1], toPt[1]] : []],
    });

    const pathX = [];
    const pathY = [];
    const pathText = [];
    const pathCustom = [];
    for (let i = 0; i < sequence.length; i += 1) {
      const idx = sequence[i];
      if (idx === null) continue;
      const pt = currentPoint(idx);
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
    const src = selectedIdx === null ? null : currentPoint(selectedIdx);
    if (src) {
      rows.forEach((row, i) => {
        const dst = currentPoint(row.idx);
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
      hoverinfo: ['skip'],
    });
  }
  function selectTrack(idx) {
    idx = Number(idx);
    if (!byIdx.has(idx)) return;
    selectedIdx = idx;
    const r = byIdx.get(idx);
    showSongPopover(r);
    renderSelectionOnly();
  }
  function setCurrentTrack(idx) {
    idx = Number(idx);
    if (!byIdx.has(idx)) return;
    selectedIdx = idx;
    hideSongPopover({ stopAudio: false });
    renderSelectionOnly();
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
  function compareLibraryRecords(a, b) {
    const key = librarySort;
    if (!key) return 0;
    let av = a[key];
    let bv = b[key];
    if (key === 'human_energy' || key === 'glm_energy' || key === 'est_bpm' || key === 'duration_seconds') {
      av = Number(av);
      bv = Number(bv);
      if (!Number.isFinite(av)) av = -Infinity;
      if (!Number.isFinite(bv)) bv = -Infinity;
      return av === bv ? String(a.title).localeCompare(String(b.title)) : av - bv;
    }
    return String(av || '').localeCompare(String(bv || ''), undefined, { numeric: true, sensitivity: 'base' });
  }
  function renderLibrary() {
    if (!els.libraryTable) return;
    const query = String(libraryQuery || '').trim().toLowerCase();
    let rows = records.filter(r => {
      if (!query) return true;
      const haystack = [r.title, r.artists, r.key, r.genre, r.raw_genre, r.mix_slug, r.filename, r.est_bpm, r.duration_text, r.human_energy, r.glm_energy]
        .map(v => String(v == null ? '' : v).toLowerCase()).join(' ');
      return haystack.includes(query);
    });
    if (librarySort) {
      rows = rows.slice().sort(compareLibraryRecords);
      if (libraryDir === 'desc') rows = rows.reverse();
    }
    const sortTh = (key, label, cls='') => {
      const indicator = librarySort === key ? (libraryDir === 'asc' ? '▲' : '▼') : '';
      return '<th ' + (cls ? 'class="' + cls + '" ' : '') + 'data-library-sort="' + esc(key) + '" data-sort-indicator="' + indicator + '">' + esc(label) + '</th>';
    };
    let html = '<table><thead><tr>' +
      sortTh('title', 'Track') +
      sortTh('artists', 'Artist') +
      sortTh('key', 'Key') +
      sortTh('est_bpm', 'BPM', 'num') +
      sortTh('duration_seconds', 'Duration', 'num') +
      sortTh('raw_genre', 'Genre') +
      '<th>Actions</th></tr></thead><tbody>';
    rows.forEach(r => {
      const rowClasses = [];
      if (Number(r.idx) === selectedIdx) rowClasses.push('selected-slot');
      if (Number(r.idx) === previewAudioIdx) rowClasses.push(previewAudioPlaying ? 'now-playing' : 'now-playing-paused');
      const rowClass = rowClasses.length ? ' class="' + rowClasses.join(' ') + '"' : '';
      html += '<tr data-library-row-idx="' + r.idx + '"' + rowClass + '>' +
        '<td>' + trackSummaryHtml(r, { size: 'compact', showArt: true }) + '</td>' +
        '<td>' + esc(r.artists) + '</td>' +
        '<td>' + keyHtml(r.key) + '</td>' +
        '<td class="num">' + roundedBpm(r) + '</td>' +
        '<td class="num">' + esc(durationText(r)) + '</td>' +
        '<td>' + esc(rawGenre(r)) + '</td>' +
        '<td><div class="library-actions">' +
        rowPlayerHtml(r) +
        '<button data-library-current="' + r.idx + '">Current</button>' +
        '<button data-library-outgoing="' + r.idx + '">Track 1</button>' +
        '<button data-library-incoming="' + r.idx + '">Track 2</button>' +
        '<button class="primary" data-library-place="' + r.idx + '">Place</button>' +
        '</div></td></tr>';
    });
    html += '</tbody></table>';
    els.libraryTable.innerHTML = html;
    updatePlayButtons();
    updateRowScrubbers();
  }
  function renderSequence() {
    const targets = targetCurve();
    let html = '<table><thead><tr><th class="num">Slot</th><th>Track</th><th class="num">Target</th><th class="num">Actual</th><th></th></tr></thead><tbody>';
    for (let i = 0; i < sequence.length; i += 1) {
      const idx = sequence[i];
      const r = idx === null ? null : byIdx.get(idx);
      const cls = selectedSlot === i ? ' class="selected-slot"' : '';
      html += '<tr' + cls + '><td class="num">' + (i + 1) + '</td><td>';
      if (r) html += trackSummaryHtml(r, { size: 'compact', showArt: true });
      else html += '<span class="muted">empty</span>';
      const targetValue = Number(targetValues[i]);
      const targetInput = '<input class="target-edit" data-target-slot="' + i + '" type="number" min="1" max="9" step="0.1" placeholder="' + fmt(targets[i], 2) + '" value="' + (Number.isFinite(targetValue) ? fmt(targetValue, 2) : '') + '">';
      html += '</td><td class="num">' + targetInput + '</td><td class="num">' + (r ? fmt(energyOf(r), 2) : '') + '</td><td>';
      if (r) html += '<button class="play-button" data-play-idx="' + r.idx + '" aria-label="Play">▶</button> ';
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
  function renderCurrentTransitionScore() {
    if (!els.currentTransitionScore) return;
    const from = transitionFromIdx === null ? null : byIdx.get(transitionFromIdx);
    const to = transitionToIdx === null ? null : byIdx.get(transitionToIdx);
    if (!from || !to) {
      els.currentTransitionScore.innerHTML = '<div class="muted">Assign outgoing and incoming tracks to inspect the selected transition score.</div>';
      return;
    }
    const slot = Math.max(0, targetSlot());
    const score = scoreTransition(transitionFromIdx, transitionToIdx, slot);
    els.currentTransitionScore.innerHTML =
      '<div><b>' + esc(from.title) + '</b> -> <b>' + esc(to.title) + '</b> <span class="muted">slot ' + (slot + 1) + '</span></div>' +
      '<table><thead><tr><th class="num">Final</th><th class="num">Transition</th><th class="num">Style</th><th class="num">Tempo</th><th class="num">Groove</th><th class="num">Key</th><th class="num">Energy penalty</th></tr></thead><tbody><tr>' +
      '<td class="num">' + fmt(score.finalScore, 4) + '</td>' +
      '<td class="num">' + fmt(score.baseline, 4) + '</td>' +
      '<td class="num">' + fmt(score.styleScore, 4) + '</td>' +
      '<td class="num">' + fmt(score.tempoScore, 4) + '</td>' +
      '<td class="num">' + fmt(score.grooveScore, 4) + '</td>' +
      '<td class="num">' + fmt(score.keyScore, 4) + '</td>' +
      '<td class="num">' + fmt(score.penalty, 4) + '</td>' +
      '</tr></tbody></table>';
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
      '<th class="num">Final</th><th class="num">Transition</th><th class="num">Target</th><th class="num">Energy</th>' +
      '<th class="num">Err</th><th class="num">Penalty</th><th class="num">Style</th><th class="num">Tempo</th><th class="num">Groove</th><th class="num">Key</th>' +
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
        '<br><button class="play-button" data-play-idx="' + from.idx + '" aria-label="Play">▶</button> ' +
        '<button class="play-button" data-play-idx="' + to.idx + '" aria-label="Play">▶</button>' +
        (score.missing ? '<br><span class="warn">missing similarity row</span>' : '') + '</td>' +
        '<td class="num">' + fmt(score.finalScore, 4) + '</td>' +
        '<td class="num">' + fmt(score.baseline, 4) + '</td>' +
        '<td class="num">' + fmt(score.target, 2) + '</td>' +
        '<td class="num">' + fmt(score.energy, 2) + '</td>' +
        '<td class="num">' + fmt(score.rawError, 2) + '</td>' +
        '<td class="num">' + fmt(score.penalty, 4) + '</td>' +
        '<td class="num">' + fmt(score.styleScore, 4) + '</td>' +
        '<td class="num">' + fmt(score.tempoScore, 4) + '</td>' +
        '<td class="num">' + fmt(score.grooveScore, 4) + '</td>' +
        '<td class="num">' + fmt(score.keyScore, 4) + '</td>' +
        '</tr>';
    }
    html += '</tbody></table>';
    els.transitionDiagnostics.innerHTML = html;
  }
  function renderRecommendations() {
    const filterbar = recommendationFilterHtml();
    if (selectedIdx === null) {
      els.recommendationPanel.innerHTML = filterbar + '<div class="muted" style="padding:10px;">Select a current track to score candidates.</div>';
      return;
    }
    const slot = targetSlot();
    if (slot < 0) {
      els.recommendationPanel.innerHTML = filterbar + '<div class="muted" style="padding:10px;">Sequence is full. Select a slot to replace a track.</div>';
      return;
    }
    const rows = rankedRecommendations().slice(0, Math.min(15, Number(config.top_k_rows || 25)));
    const actionLabel = sequence[slot] === null ? 'Append' : 'Replace';
    let html = filterbar +
      '<div class="muted" style="padding:8px 8px 0;">Scoring candidates for slot <b>' + (slot + 1) + '</b> (' + actionLabel.toLowerCase() + ').</div>' +
      '<table><thead><tr><th class="num">#</th><th>Actions</th><th>Track</th><th class="num">Final</th><th class="num">Mix</th><th class="num">Energy</th><th class="num">Penalty</th><th class="num">Style</th><th class="num">Tempo</th><th class="num">Groove</th><th class="num">Key</th></tr></thead><tbody>';
    rows.forEach((r, i) => {
      html += '<tr><td class="num">' + (i + 1) + '</td>' +
        '<td><div class="library-actions">' +
        '<button class="play-button" data-play-idx="' + r.idx + '" aria-label="Play">▶</button>' +
        '<button class="primary" data-append-idx="' + r.idx + '">' + actionLabel + '</button>' +
        '<button data-library-outgoing="' + r.idx + '">T1</button>' +
        '<button data-library-incoming="' + r.idx + '">T2</button>' +
        '</div></td>' +
        '<td>' + trackSummaryHtml(r, { size: 'compact', showArt: true }) + '</td>' +
        '<td class="num">' + fmt(r.finalScore, 2) + '</td><td class="num">' + fmt(r.baseline, 2) + '</td>' +
        '<td class="num">' + fmt(r.energy, 2) + '</td><td class="num">' + fmt(r.penalty, 2) + '</td>' +
        '<td class="num">' + fmt(r.styleScore, 2) + '</td><td class="num">' + fmt(r.tempoScore, 2) + '</td><td class="num">' + fmt(r.grooveScore, 2) + '</td><td class="num">' + fmt(r.keyScore, 2) + '</td></tr>';
    });
    if (!rows.length) {
      html += '<tr><td colspan="11" class="muted" style="padding:10px;">No recommendations match the current filters.</td></tr>';
    }
    html += '</tbody></table>';
    els.recommendationPanel.innerHTML = html;
  }
  function renderEnergyCurve() {
    const targets = targetCurve();
    const x = Array.from({ length: sequence.length }, (_, i) => i + 1);
    const actual = sequence.map(idx => idx === null ? null : energyOf(byIdx.get(idx)));
    const traces = [
      { x, y: targets, type: 'scatter', mode: 'lines+markers', name: 'Target energy', line: { color: '#e5e7eb', width: 2 }, marker: { size: 10 } },
      { x, y: actual, type: 'scatter', mode: 'lines+markers', name: 'Selected actual energy', line: { color: '#14b8a6', width: 2 }, marker: { size: 9 } }
    ];
    const slot = targetSlot();
    if (slot >= 0) traces.push({ x: [slot + 1], y: [targets[slot]], type: 'scatter', mode: 'markers', name: 'Next slot', marker: { size: 14, color: '#f59e0b', symbol: 'x' } });
    Plotly.react('energy-curve', traces, {
      title: { text: 'Drag target points to edit the energy curve', font: { size: 13, color: '#e5e7eb' } },
      margin: { t: 48, r: 20, b: 46, l: 48 },
      template: 'plotly_dark',
      paper_bgcolor: '#111827',
      plot_bgcolor: '#111827',
      font: { color: '#e5e7eb' },
      xaxis: { title: 'Sequence slot', dtick: 1, range: [0.5, sequence.length + 0.5], gridcolor: '#263244', zerolinecolor: '#263244' },
      yaxis: { title: 'Energy', range: [0.5, 9.5], gridcolor: '#263244', zerolinecolor: '#263244' },
      legend: { orientation: 'h', x: 0, y: -0.24, xanchor: 'left', yanchor: 'top' }
    }, { displayModeBar: false, responsive: true });
  }
  function renderTransitionBadges() {
    if (!els.transitionBadges) return;
    const from = transitionFromIdx === null ? null : byIdx.get(transitionFromIdx);
    const to = transitionToIdx === null ? null : byIdx.get(transitionToIdx);
    let scoreHtml = '<div class="transition-mini-score"><span class="muted">Assign Track 1 and Track 2 to preview the transition score.</span></div>';
    if (from && to) {
      const slot = Math.max(0, targetSlot());
      const score = scoreTransition(transitionFromIdx, transitionToIdx, slot);
      scoreHtml =
        '<div class="transition-mini-score">' +
        '<table><thead><tr><th class="num">Final</th><th class="num">Mix</th><th class="num">Style</th><th class="num">Tempo</th><th class="num">Groove</th><th class="num">Key</th><th class="num">Penalty</th></tr></thead><tbody><tr>' +
        '<td class="num">' + fmt(score.finalScore, 2) + '</td>' +
        '<td class="num">' + fmt(score.baseline, 2) + '</td>' +
        '<td class="num">' + fmt(score.styleScore, 2) + '</td>' +
        '<td class="num">' + fmt(score.tempoScore, 2) + '</td>' +
        '<td class="num">' + fmt(score.grooveScore, 2) + '</td>' +
        '<td class="num">' + fmt(score.keyScore, 2) + '</td>' +
        '<td class="num">' + fmt(score.penalty, 2) + '</td>' +
        '</tr></tbody></table>' +
        '</div>';
    }
    els.transitionBadges.innerHTML =
      '<div class="transition-badge out"><b>Track 1</b>' + shortTrack(from) + '</div>' +
      '<div class="transition-badge in"><b>Track 2</b>' + shortTrack(to) + '</div>' +
      scoreHtml;
  }
  function renderTransitionEditorHtml(from, to) {
    if (!from || !to) {
      return '<div class="transition-editor"><div class="transition-editor-empty">Assign Track 1 and Track 2 to align cue points and preview windows.</div></div>';
    }
    const pitchValues = [-3, -2, -1, 0, 1, 2, 3];
    const laneInfoHtml = deck => {
      const info = transitionWindowInfo(deck);
      if (!info) return '';
      const cfg = info.cfg;
      const cueName = info.cue ? (info.cue.name || 'cue') : 'no cue';
      return '<div class="transition-lane-head">' +
        '<b>' + esc(cfg.label) + '</b>' +
        '<div class="transition-lane-title">' + esc(info.record.title || 'Untitled') + '</div>' +
        '<div class="transition-lane-meta">' +
        '<span>' + esc(cueName) + '</span>' +
        '<span>' + fmt(info.bpm, 2) + ' BPM</span>' +
        '<span><span data-transition-nudge-label="' + esc(deck) + '">' + fmt(info.nudgeBeats, 2) + '</span> beats</span>' +
        '</div>' +
        '</div>';
    };
    const laneToolsHtml = deck => {
      const info = transitionWindowInfo(deck);
      if (!info) return '';
      const cfg = info.cfg;
      const pitchField = deck === 'from' ? 'from_pitch_shift' : 'to_pitch_shift';
      const bounds = transitionNudgeBounds(info);
      return '<div class="transition-lane-tools">' +
        cueSelectHtml(info.record, cfg.role, cfg.cueField, 'Cue') +
        numericInput(cfg.nudgeField, 'Nudge beats', previewFormState[cfg.nudgeField], bounds.min, bounds.max, 0.05) +
        selectInput(pitchField, 'Repitch', pitchValues, previewFormState[pitchField], v => (Number(v) > 0 ? '+' : '') + v + ' st') +
        '</div>';
    };
    const overviewHtml = deck => {
      const info = transitionWindowInfo(deck);
      if (!info) return '';
      return '<div class="transition-overview-card ' + esc(deck) + '" data-transition-card="' + esc(deck) + '-overview">' +
        laneInfoHtml(deck) +
        '<div class="transition-overview-row">' +
        '<button type="button" class="transition-track-play" data-transition-track-play="' + esc(deck) + '" aria-label="Play ' + esc(info.cfg.label) + '">▶</button>' +
        '<canvas class="transition-overview-canvas" data-transition-surface="overview" data-transition-deck="' + esc(deck) + '" height="76"></canvas>' +
        '</div>' +
        '</div>';
    };
    const sectionHtml = deck => {
      const info = transitionWindowInfo(deck);
      if (!info) return '';
      const cfg = info.cfg;
      return '<div class="transition-window-card ' + esc(deck) + '" data-transition-card="' + esc(deck) + '-section">' +
        '<div class="transition-window-main">' +
        '<canvas class="transition-section-canvas" data-transition-surface="section" data-transition-deck="' + esc(deck) + '" height="245"></canvas>' +
        '</div>' +
        '<aside class="transition-track-side">' +
        '<div class="transition-track-meta"><div class="transition-track-label">' + esc(cfg.label) + '</div>' + trackMetaHtml(info.record) + '</div>' +
        laneToolsHtml(deck) +
        '</aside>' +
        '</div>';
    };
    return '<div class="transition-editor">' +
      '<div class="transition-editor-head">' +
      '<h2>Transition Alignment</h2>' +
      '<div class="transition-editor-tools">' +
      '<label><input id="transition-snap-to-beat" type="checkbox"' + (transitionSnapToBeat ? ' checked' : '') + '> Snap to beat</label>' +
      '<span id="transition-editor-dirty" class="transition-editor-hint"></span>' +
      '</div>' +
      '</div>' +
      '<div class="transition-editor-hint">Drag a transition lane or overview window to shift timing. Cue markers can be clicked directly; snap locks movement to beats.</div>' +
      '<div class="transition-native-stage">' +
      overviewHtml('from') +
      '<div class="transition-window-stack">' + sectionHtml('from') + transitionTransportHtml() + sectionHtml('to') + '</div>' +
      overviewHtml('to') +
      '</div>' +
      '</div>';
  }
  function cueMarkerColor(role) {
    const r = String(role || '').toLowerCase();
    if (r === 'in') return '#10b981';
    if (r === 'out') return '#f59e0b';
    if (r === 'drop') return '#ec4899';
    if (r === 'break') return '#38bdf8';
    if (r === 'start') return '#94a3b8';
    return '#cbd5e1';
  }
  function peakAtTime(record, timeSec) {
    const duration = Math.max(1, Number(record.duration_seconds) || audioDuration(record) || 1);
    if (Number(timeSec) < 0 || Number(timeSec) > duration) return 0;
    const peaks = waveformSeries(waveformPayload(record), 'detail');
    if (!peaks.length) return 0.12;
    const pos = clamp((Number(timeSec) / duration) * (peaks.length - 1), 0, peaks.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(peaks.length - 1, lo + 1);
    const frac = pos - lo;
    const a = Number(peaks[lo]) || 0.05;
    const b = Number(peaks[hi]) || a;
    return clamp(a + ((b - a) * frac), 0.035, 1);
  }
  function profileValueAtTime(record, profile, timeSec) {
    const values = Array.isArray(profile) ? profile : [];
    if (!values.length) return peakAtTime(record, timeSec);
    const duration = Math.max(1, Number(record.duration_seconds) || audioDuration(record) || 1);
    if (Number(timeSec) < 0 || Number(timeSec) > duration) return 0;
    const pos = clamp((Number(timeSec) / duration) * (values.length - 1), 0, values.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(values.length - 1, lo + 1);
    const frac = pos - lo;
    const a = Number(values[lo]) || 0.05;
    const b = Number(values[hi]) || a;
    return clamp(a + ((b - a) * frac), 0.035, 1);
  }
  function signedProfileValueAtTime(record, values, timeSec) {
    const profile = Array.isArray(values) ? values : [];
    if (!profile.length) return 0;
    const duration = Math.max(1, Number(record.duration_seconds) || audioDuration(record) || 1);
    if (Number(timeSec) < 0 || Number(timeSec) > duration) return 0;
    const pos = clamp((Number(timeSec) / duration) * (profile.length - 1), 0, profile.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(profile.length - 1, lo + 1);
    const frac = pos - lo;
    const a = Number(profile[lo]) || 0;
    const b = Number(profile[hi]) || a;
    return clamp(a + ((b - a) * frac), -1, 1);
  }
  function transitionLaneRect(width, height, surface) {
    if (surface === 'overview') return { x: 8, y: 7, w: Math.max(1, width - 16), h: Math.max(1, height - 14) };
    return { x: 10, y: 10, w: Math.max(1, width - 20), h: Math.max(1, height - 20) };
  }
  function transitionVisibleRange(info, surface) {
    const trackDuration = Math.max(info.beatSec, info.trackDuration || info.duration);
    if (surface === 'overview') return { start: 0, end: trackDuration, duration: trackDuration };
    const start = info.contextStart;
    const end = info.contextEnd;
    return { start, end, duration: Math.max(info.beatSec, end - start) };
  }
  function xForTransitionTime(time, view, lane) {
    return lane.x + (clamp((Number(time) - view.start) / Math.max(1e-9, view.duration), 0, 1) * lane.w);
  }
  function transitionWaveSamples(record, view, count) {
    const n = Math.max(8, Math.round(count || 160));
    const samples = [];
    for (let i = 0; i < n; i += 1) {
      const t = view.start + ((i + 0.5) / n) * view.duration;
      let peak = peakAtTime(record, t);
      if (!Number.isFinite(peak)) peak = 0.08 + 0.05 * Math.sin(i * 0.43);
      samples.push(peak <= 0 ? 0 : clamp(peak, 0.035, 1));
    }
    return samples;
  }
  function transitionProfileSamples(record, profile, view, count) {
    const n = Math.max(8, Math.round(count || 160));
    const values = normalizeWaveformValues(profile);
    const samples = [];
    for (let i = 0; i < n; i += 1) {
      const t = view.start + ((i + 0.5) / n) * view.duration;
      samples.push(profileValueAtTime(record, values, t));
    }
    return samples;
  }
  function transitionBandSamples(record, view, count) {
    const payload = waveformPayload(record);
    if (!payload || !payload.bands) return null;
    const low = payload.bands.low && payload.bands.low.length ? transitionProfileSamples(record, payload.bands.low, view, count) : null;
    const mid = payload.bands.mid && payload.bands.mid.length ? transitionProfileSamples(record, payload.bands.mid, view, count) : null;
    const high = payload.bands.high && payload.bands.high.length ? transitionProfileSamples(record, payload.bands.high, view, count) : null;
    return (low || mid || high) ? { low, mid, high } : null;
  }
  function transitionEnvelopeSamples(record, view, count) {
    const payload = waveformPayload(record);
    const envelope = waveformEnvelopeSeries(payload, 'detail');
    if (!envelope || !envelope.min || !envelope.max || !envelope.min.length || !envelope.max.length) return null;
    const minValues = normalizeSignedWaveformValues(envelope.min);
    const maxValues = normalizeSignedWaveformValues(envelope.max);
    const n = Math.max(8, Math.round(count || 160));
    const low = [];
    const high = [];
    for (let i = 0; i < n; i += 1) {
      const t = view.start + ((i + 0.5) / n) * view.duration;
      low.push(signedProfileValueAtTime(record, minValues, t));
      high.push(signedProfileValueAtTime(record, maxValues, t));
    }
    return { min: low, max: high };
  }
  function smoothTransitionSample(samples, i, pow=1) {
    const a = samples[Math.max(0, i - 2)] || 0;
    const b = samples[Math.max(0, i - 1)] || 0;
    const c = samples[i] || 0;
    const d = samples[Math.min(samples.length - 1, i + 1)] || 0;
    const e = samples[Math.min(samples.length - 1, i + 2)] || 0;
    return Math.pow(clamp((a + 2 * b + 3 * c + 2 * d + e) / 9, 0, 1), pow);
  }
  function drawTransitionEnvelope(ctx, samples, lane, color, scale, pow=1) {
    if (!samples.length) return;
    const mid = lane.y + lane.h * 0.5;
    const amp = lane.h * 0.46 * scale;
    const denom = Math.max(1, samples.length - 1);
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    samples.forEach((_, i) => {
      const x = lane.x + (i / denom) * lane.w;
      const y = mid - smoothTransitionSample(samples, i, pow) * amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    for (let i = samples.length - 1; i >= 0; i -= 1) {
      const x = lane.x + (i / denom) * lane.w;
      const y = mid + smoothTransitionSample(samples, i, pow) * amp;
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
  function drawTransitionSignedEnvelope(ctx, envelope, lane, color, strokeColor=null) {
    if (!envelope || !envelope.min || !envelope.max || !envelope.min.length || !envelope.max.length) return;
    const n = Math.min(envelope.min.length, envelope.max.length);
    if (n < 2) return;
    const mid = lane.y + lane.h * 0.5;
    const amp = lane.h * 0.48;
    const denom = Math.max(1, n - 1);
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    for (let i = 0; i < n; i += 1) {
      const x = lane.x + (i / denom) * lane.w;
      const y = mid - (clamp(Number(envelope.max[i]) || 0, -1, 1) * amp);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    for (let i = n - 1; i >= 0; i -= 1) {
      const x = lane.x + (i / denom) * lane.w;
      const y = mid - (clamp(Number(envelope.min[i]) || 0, -1, 1) * amp);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    if (strokeColor) {
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.restore();
  }
  function transitionAutomationValue(info, family, timeSec) {
    const modes = resolvedPreviewModes();
    const mode = family + ':' + (modes[family] || 'none');
    const deck = info.cfg.deck === 'from' ? 'a' : 'b';
    const duration = Math.max(1e-9, info.transitionEnd - info.transitionStart);
    const x = clamp((Number(timeSec) - info.transitionStart) / duration, 0, 1);
    if (timeSec < info.transitionStart) {
      if (family === 'volume') return deck === 'a' ? 1 : 0;
      if (family === 'eq') return 1;
      return null;
    }
    if (timeSec > info.transitionEnd) {
      if (family === 'volume') return deck === 'a' ? 0 : 1;
      if (family === 'eq') return 1;
      return null;
    }
    return previewValueFor(mode, deck, x);
  }
  function drawTransitionAutomationCurve(ctx, info, view, lane, family, color) {
    const samples = Math.max(90, Math.floor(lane.w / 5));
    let started = false;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = family === 'filter' ? 2.0 : 2.4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i <= samples; i += 1) {
      const pct = i / samples;
      const timeSec = view.start + (pct * view.duration);
      const value = transitionAutomationValue(info, family, timeSec);
      if (value == null || !Number.isFinite(Number(value))) {
        started = false;
        continue;
      }
      const x = lane.x + (pct * lane.w);
      const y = lane.y + ((1 - clamp(Number(value), 0, 1)) * lane.h);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.restore();
  }
  function drawTransitionAutomationCurves(ctx, info, view, lane) {
    if (!info || !view || !lane) return;
    drawTransitionAutomationCurve(ctx, info, view, lane, 'volume', 'rgba(72,200,242,.94)');
    drawTransitionAutomationCurve(ctx, info, view, lane, 'eq', 'rgba(255,209,61,.90)');
    drawTransitionAutomationCurve(ctx, info, view, lane, 'filter', 'rgba(229,140,255,.88)');
    ctx.save();
    ctx.fillStyle = 'rgba(242,242,242,.70)';
    ctx.font = '10px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.fillText('volume / EQ / filter', lane.x + 8, lane.y + 14);
    ctx.restore();
  }
  function drawTransitionWaveform(ctx, info, view, lane, surface) {
    const sampleCount = surface === 'overview' ? Math.max(260, Math.floor(lane.w * 1.25)) : Math.max(900, Math.floor(lane.w * 2.4));
    const samples = transitionWaveSamples(info.record, view, sampleCount);
    const signedEnvelope = transitionEnvelopeSamples(info.record, view, sampleCount);
    const bands = transitionBandSamples(info.record, view, sampleCount);
    if (signedEnvelope) {
      drawTransitionSignedEnvelope(ctx, signedEnvelope, lane, 'rgba(238,242,255,.22)', 'rgba(255,255,255,.12)');
    }
    drawTransitionEnvelope(ctx, bands && bands.low ? bands.low : samples, lane, 'rgba(24,75,216,.86)', 1.0, 1.08);
    drawTransitionEnvelope(ctx, bands && bands.mid ? bands.mid : samples, lane, 'rgba(211,128,39,.78)', 0.70, 0.90);
    drawTransitionEnvelope(ctx, bands && bands.high ? bands.high : samples, lane, 'rgba(248,242,226,.90)', 0.34, 0.55);
    const mid = lane.y + lane.h * 0.5;
    const amp = lane.h * 0.34;
    const denom = Math.max(1, samples.length - 1);
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,.13)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    samples.forEach((_, i) => {
      const x = lane.x + (i / denom) * lane.w;
      const y = mid - smoothTransitionSample(samples, i, 1.1) * amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  }
  function drawTransitionGrid(ctx, info, view, lane, surface) {
    const firstBeat = Math.floor((view.start - info.transitionStart) / info.beatSec) - 1;
    const lastBeat = Math.ceil((view.end - info.transitionStart) / info.beatSec) + 1;
    const beatSpan = Math.max(1, lastBeat - firstBeat);
    const step = surface === 'overview'
      ? Math.max(info.beatsPerBar * 4, Math.ceil(beatSpan / 80 / info.beatsPerBar) * info.beatsPerBar)
      : 1;
    for (let b = firstBeat; b <= lastBeat; b += step) {
      const t = info.transitionStart + b * info.beatSec;
      if (t < view.start || t > view.end) continue;
      const x = xForTransitionTime(t, view, lane);
      const isBar = b % info.beatsPerBar === 0;
      const isPhrase = b % (info.beatsPerBar * 4) === 0;
      ctx.strokeStyle = isPhrase ? '#555' : (isBar ? '#3e3e3e' : '#2d2d2d');
      ctx.lineWidth = isPhrase ? 1.25 : 1;
      ctx.beginPath();
      ctx.moveTo(x, lane.y);
      ctx.lineTo(x, lane.y + lane.h);
      ctx.stroke();
      if (surface !== 'overview' && isBar && lane.w / Math.max(1, beatSpan / info.beatsPerBar) > 24) {
        ctx.fillStyle = isPhrase ? '#9f9f9f' : '#666';
        ctx.font = '10px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.fillText(String(Math.round(b / info.beatsPerBar)), x + 4, lane.y + 13);
      }
    }
  }
  function drawTransitionSelection(ctx, info, view, lane) {
    const selectedStart = Math.max(view.start, info.transitionStart);
    const selectedEnd = Math.min(view.end, info.transitionEnd);
    if (selectedEnd <= selectedStart) return;
    const x0 = xForTransitionTime(selectedStart, view, lane);
    const x1 = xForTransitionTime(selectedEnd, view, lane);
    ctx.save();
    ctx.fillStyle = info.cfg.deck === 'from' ? 'rgba(16,185,129,.16)' : 'rgba(245,158,11,.16)';
    ctx.fillRect(x0, lane.y, Math.max(2, x1 - x0), lane.h);
    ctx.strokeStyle = info.cfg.color;
    ctx.lineWidth = 2;
    [x0, x1].forEach(x => {
      ctx.beginPath();
      ctx.moveTo(x, lane.y);
      ctx.lineTo(x, lane.y + lane.h);
      ctx.stroke();
    });
    ctx.restore();
  }
  function drawTransitionCueMarkers(ctx, info, view, lane, surface) {
    cuesForRecord(info.record).forEach(cue => {
      const start = Number(cue.start_seconds);
      if (!Number.isFinite(start) || start < view.start || start > view.end) return;
      const x = xForTransitionTime(start, view, lane);
      ctx.strokeStyle = cueMarkerColor(cue.role);
      ctx.lineWidth = surface === 'overview' ? 1 : 1.4;
      ctx.beginPath();
      ctx.moveTo(x, lane.y + 4);
      ctx.lineTo(x, lane.y + lane.h - 4);
      ctx.stroke();
      const label = String(cue.name || cue.role || '').slice(0, surface === 'overview' ? 9 : 16);
      if (label && (surface !== 'overview' || x > lane.x + 8 && x < lane.x + lane.w - 42)) {
        ctx.fillStyle = cueMarkerColor(cue.role);
        ctx.font = (surface === 'overview' ? '10px' : '11px') + ' ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.fillText(label, clamp(x + 4, lane.x + 4, lane.x + lane.w - 84), surface === 'overview' ? lane.y + 12 : lane.y + lane.h - 8);
      }
    });
  }
  function transitionPlaybackProgress() {
    if (!transitionRender || !transitionRender.ok) return null;
    const audio = transitionAudioElement();
    const scrub = document.getElementById('transition-scrub');
    const duration = transitionDuration(audio);
    if (!Number.isFinite(duration) || duration <= 0) return null;
    let current = audio && Number.isFinite(Number(audio.currentTime)) ? Number(audio.currentTime) : NaN;
    if (!Number.isFinite(current) && scrub) current = Number(scrub.value);
    if (!Number.isFinite(current)) return null;
    return clamp(current / duration, 0, 1);
  }
  function drawTransitionPlaybackHead(ctx, info, view, lane, surface) {
    const progress = transitionPlaybackProgress();
    if (progress == null) return;
    const trackTime = info.contextStart + (progress * info.duration);
    if (trackTime < view.start || trackTime > view.end) return;
    const x = xForTransitionTime(trackTime, view, lane);
    ctx.save();
    ctx.strokeStyle = surface === 'overview' ? 'rgba(255,255,255,.72)' : 'rgba(255,255,255,.94)';
    ctx.lineWidth = surface === 'overview' ? 1.4 : 2.1;
    ctx.shadowColor = 'rgba(103,232,249,.62)';
    ctx.shadowBlur = surface === 'overview' ? 4 : 8;
    ctx.beginPath();
    ctx.moveTo(x, lane.y + 2);
    ctx.lineTo(x, lane.y + lane.h - 2);
    ctx.stroke();
    if (surface === 'section') {
      ctx.fillStyle = 'rgba(255,255,255,.95)';
      ctx.beginPath();
      ctx.moveTo(x, lane.y + 2);
      ctx.lineTo(x - 5, lane.y + 11);
      ctx.lineTo(x + 5, lane.y + 11);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }
  function drawTransitionSurface(canvas, info, surface) {
    const rect = canvas.getBoundingClientRect();
    const widthCss = Math.max(1, rect.width || canvas.clientWidth || (surface === 'overview' ? 320 : 420));
    const heightCss = Math.max(1, rect.height || (surface === 'overview' ? 54 : 148));
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(widthCss * dpr);
    canvas.height = Math.floor(heightCss * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, widthCss, heightCss);
    ctx.fillStyle = '#121212';
    ctx.fillRect(0, 0, widthCss, heightCss);
    const lane = transitionLaneRect(widthCss, heightCss, surface);
    ctx.fillStyle = '#1d1d1d';
    if (ctx.roundRect) {
      ctx.beginPath();
      ctx.roundRect(lane.x, lane.y, lane.w, lane.h, surface === 'overview' ? 5 : 7);
      ctx.fill();
    } else {
      ctx.fillRect(lane.x, lane.y, lane.w, lane.h);
    }
    const view = transitionVisibleRange(info, surface);
    drawTransitionGrid(ctx, info, view, lane, surface);
    drawTransitionWaveform(ctx, info, view, lane, surface);
    drawTransitionSelection(ctx, info, view, lane);
    if (surface === 'section') drawTransitionAutomationCurves(ctx, info, view, lane);
    drawTransitionCueMarkers(ctx, info, view, lane, surface);
    drawTransitionPlaybackHead(ctx, info, view, lane, surface);
    if (surface === 'section') {
      ctx.fillStyle = 'rgba(242,242,242,.84)';
      ctx.font = '11px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText('drag timing / click cue', lane.x + 8, lane.y + lane.h - 10);
    }
  }
  function drawTransitionOverview(canvas, info) {
    drawTransitionSurface(canvas, info, 'overview');
  }
  function drawTransitionSection(canvas, info) {
    drawTransitionSurface(canvas, info, 'section');
  }
  function transitionCueAtPoint(canvas, info, ev) {
    const rect = canvas.getBoundingClientRect();
    const surface = canvas.getAttribute('data-transition-surface') || 'section';
    const lane = transitionLaneRect(Math.max(1, rect.width), Math.max(1, rect.height), surface);
    const x = Number(ev.clientX) - rect.left;
    const y = Number(ev.clientY) - rect.top;
    if (x < lane.x || x > lane.x + lane.w || y < lane.y - 6 || y > lane.y + lane.h + 6) return null;
    const view = transitionVisibleRange(info, surface);
    const threshold = surface === 'overview' ? 8 : 10;
    let best = null;
    let bestDx = Infinity;
    cuesForRecord(info.record).forEach(cue => {
      const start = Number(cue.start_seconds);
      if (!Number.isFinite(start) || start < view.start || start > view.end) return;
      const cueX = xForTransitionTime(start, view, lane);
      const dx = Math.abs(cueX - x);
      if (dx < bestDx && dx <= threshold) {
        best = cue;
        bestDx = dx;
      }
    });
    return best;
  }
  function drawTransitionEditor() {
    safeUi('transition editor draw', () => {
      ['from', 'to'].forEach(deck => {
        const info = transitionWindowInfo(deck);
        if (!info) return;
        ensureDetailedWaveform(info.record, { redrawTransition: true });
        const overview = document.querySelector('[data-transition-surface="overview"][data-transition-deck="' + deck + '"]');
        const section = document.querySelector('[data-transition-surface="section"][data-transition-deck="' + deck + '"]');
        if (overview) drawTransitionOverview(overview, info);
        if (section) drawTransitionSection(section, info);
        const label = document.querySelector('[data-transition-nudge-label="' + deck + '"]');
        if (label) label.textContent = fmt(info.nudgeBeats, 2);
      });
      updateTransitionTrackPlayButtons();
      updateTransitionDirtyUi();
    });
  }
  function renderTransitionPreview() {
    if (!els.transitionPreview) return;
    const from = transitionFromIdx === null ? null : byIdx.get(transitionFromIdx);
    const to = transitionToIdx === null ? null : byIdx.get(transitionToIdx);
    let controlsHtml = '';
    if (config.app_mode) {
      if (!appLoadStarted) loadAppRuntime();
      if (appOptions && appOptions.error) {
        controlsHtml = '<div class="warn" style="margin-top:14px;">' + esc(appOptions.error) + '</div>';
      } else if (!appLoadDone) {
        controlsHtml = '<div class="muted" style="margin-top:14px;">Loading render controls...</div>';
      } else if (!from || !to) {
        controlsHtml = '<div class="' + (transitionRenderWarn ? 'warn' : 'muted') + '" style="margin-top:14px;">' + esc(transitionRenderMessage || 'Assign Track 1 and Track 2 to render a transition preview.') + '</div>';
      } else {
        const presets = (appOptions.presets || ['auto']);
        const volumeModes = appOptions.volume_modes || [];
        const eqModes = appOptions.eq_modes || [];
        const filterModes = appOptions.filter_modes || [];
        const renderStatus = transitionRenderStatusInfo();
        controlsHtml =
          '<div class="transition-settings-panel">' +
          '<section class="preview-render-section"><h3>Render Controls</h3><div class="preview-render-grid preview-timing-grid">' +
          numericInput('overlap_bars', 'Overlap', previewFormState.overlap_bars, 1, 64, 1) +
          numericInput('front_padding_bars', 'Front', previewFormState.front_padding_bars, 0, 16, 1) +
          numericInput('back_padding_bars', 'Back', previewFormState.back_padding_bars, 0, 16, 1) +
          '</div></section>' +
          '<section class="preview-render-section"><h3>Transition FX</h3><div class="preview-mode-controls">' +
          selectInput('preset', 'Preset', presets, previewFormState.preset) +
          selectInput('volume_mode', 'Volume', volumeModes, previewFormState.volume_mode) +
          selectInput('eq_mode', 'EQ', eqModes, previewFormState.eq_mode) +
          selectInput('filter_mode', 'Filter', filterModes, previewFormState.filter_mode) +
          '</div></section>' +
          '<div class="preview-render-actions">' +
          automationSummaryHtml() +
          '<div class="preview-actions">' +
          '<button id="transition-render-button" type="button" class="primary" data-preview-action="render" onclick="return window.__djRenderTransition ? window.__djRenderTransition(event) : false;"' + (transitionRenderPending ? ' disabled' : '') + '>' +
          (transitionRenderPending ? 'Rendering...' : (transitionRender && previewSettingsDirty() ? 'Render again' : 'Render transition')) +
          '</button>' +
          '<button type="button" data-preview-action="swap">Swap</button>' +
          '<button type="button" data-preview-action="clear">Clear transition</button>' +
          '</div>' +
          transitionLoadingHtml(transitionRenderPending) +
          '<div id="transition-render-status" class="preview-status' + (renderStatus.warn ? ' warn' : '') + '">' + esc(renderStatus.text) + '</div>' +
          '</div>' +
          '</div>';
      }
    } else {
      controlsHtml =
        '<div class="preview-actions">' +
        '<button type="button" data-preview-action="swap">Swap</button>' +
        '<button type="button" data-preview-action="clear">Clear transition</button>' +
        '</div>';
    }
    const editorHtml = safeUi(
      'transition editor render',
      () => renderTransitionEditorHtml(from, to),
      '<div class="transition-editor"><div class="transition-editor-empty warn">Transition editor failed to render. Other app controls remain available.</div></div>'
    );
    const t = transitionRender && transitionRender.transition ? transitionRender.transition : null;
    const metaText = t ? (String(t.from_track_number || '') + ' ' + (t.from_title || '') + ' -> ' + String(t.to_track_number || '') + ' ' + (t.to_title || '') + (t.duration_seconds ? ' / ' + Number(t.duration_seconds).toFixed(2) + 's' : '')) : 'No render yet';
    const linksHtml = transitionRender && transitionRender.urls ?
      '<a href="' + esc(transitionRender.urls.preview) + '" target="_blank">WAV</a>' : '';
    els.transitionPreview.innerHTML =
      '<div class="transition-workbench">' +
      '<section class="transition-main">' +
      '<section class="transition-controls">' + controlsHtml + '</section>' +
      editorHtml +
      '<div class="transition-topbar"><div class="transition-meta">' + esc(metaText) + '</div><div class="transition-links">' + linksHtml + '</div></div>' +
      '</section>' +
      '</div>';
    safeUi('transition preview bind', bindTransitionPreviewOverlay);
  }
  function transitionTransportHtml() {
    const result = renderResultHtml();
    const body = result || '<div class="transition-scrubbar transition-scrubbar-empty">' +
      (transitionRenderPending ? transitionLoadingHtml(true, '') + '<div>Rendering transition audio...</div>' : '<div>Render transition to enable playback.</div>') +
      '</div>';
    const status = transitionRenderStatusInfo();
    return '<div class="transition-transport-row">' +
      '<div class="transition-transport-main">' + body + '</div>' +
      '<div class="transition-transport-side' + (status.warn ? ' warn' : '') + '">' + esc(status.text) + '</div>' +
      '</div>';
  }
  function renderResultHtml() {
    if (!transitionRender || !transitionRender.ok || !transitionRender.urls) return '';
    const t = transitionRender.transition || {};
    const duration = Math.max(1, Number(t.duration_seconds) || 1);
    const stamp = esc(transitionRender.rendered_at || '');
    return '<div class="transition-scrubbar">' +
      '<div class="transition-transport-controls">' +
      '<button id="transition-play-toggle" class="play-button" type="button" aria-label="Play">▶</button>' +
      '<div class="transition-transport-title">Rendered transition</div>' +
      '<span id="transition-scrub-time" class="scrub-time">0:00 / ' + esc(timeText(duration)) + '</span>' +
      '</div>' +
      '<input id="transition-scrub" class="scrub-range" type="range" min="0" max="' + esc(duration) + '" step="0.01" value="0">' +
      '<audio id="transition-preview-audio" preload="metadata" src="' + esc(transitionRender.urls.preview) + '?t=' + stamp + '"></audio>' +
      '</div>';
  }
  function transitionAudioElement() {
    return document.getElementById('transition-preview-audio');
  }
  function transitionDuration(audio) {
    const audioDuration = Number(audio && audio.duration);
    if (Number.isFinite(audioDuration) && audioDuration > 0) return audioDuration;
    const renderDuration = Number(transitionRender && transitionRender.transition && transitionRender.transition.duration_seconds);
    return Number.isFinite(renderDuration) && renderDuration > 0 ? renderDuration : 0;
  }
  function scheduleTransitionScrubber() {
    if (transitionScrubFrame !== null || !window.requestAnimationFrame) return;
    transitionScrubFrame = window.requestAnimationFrame(() => {
      transitionScrubFrame = null;
      updateTransitionScrubber();
    });
  }
  function requestTransitionEditorDraw() {
    if (activePane !== 'preview') return;
    if (!window.requestAnimationFrame) {
      drawTransitionEditor();
      return;
    }
    if (transitionEditorFrame !== null) return;
    transitionEditorFrame = window.requestAnimationFrame(() => {
      transitionEditorFrame = null;
      drawTransitionEditor();
    });
  }
  function updateTransitionScrubber() {
    const audio = transitionAudioElement();
    const button = document.getElementById('transition-play-toggle');
    const scrub = document.getElementById('transition-scrub');
    const time = document.getElementById('transition-scrub-time');
    if (!button || !scrub || !time) return;
    const duration = transitionDuration(audio);
    const current = audio && Number.isFinite(Number(audio.currentTime)) ? Number(audio.currentTime) : 0;
    button.textContent = audio && !audio.paused && !audio.ended ? '⏸' : '▶';
    button.setAttribute('aria-label', audio && !audio.paused && !audio.ended ? 'Pause' : 'Play');
    if (Number.isFinite(duration) && duration > 0) scrub.max = String(duration);
    if (!transitionScrubDragging) scrub.value = String(clamp(current, 0, Number(scrub.max) || duration || 1));
    setRangeProgress(scrub, scrub.value, scrub.max);
    time.textContent = timeText(current) + ' / ' + timeText(duration);
    requestTransitionEditorDraw();
    if (audio && !audio.paused && !audio.ended) scheduleTransitionScrubber();
  }
  function bindTransitionScrubber() {
    const audio = transitionAudioElement();
    const button = document.getElementById('transition-play-toggle');
    const scrub = document.getElementById('transition-scrub');
    if (!button || !scrub) return;
    applyMasterVolume();
    if (!button.__seqBound) {
      button.__seqBound = true;
      button.addEventListener('click', () => {
        const activeAudio = transitionAudioElement();
        if (!activeAudio) return;
        if (activeAudio.paused || activeAudio.ended) {
          const promise = activeAudio.play();
          if (promise && promise.catch) promise.catch(() => {});
        } else {
          activeAudio.pause();
        }
        updateTransitionScrubber();
      });
    }
    if (!scrub.__seqBound) {
      scrub.__seqBound = true;
      scrub.addEventListener('input', () => {
        transitionScrubDragging = true;
        const activeAudio = transitionAudioElement();
        const value = Number(scrub.value || 0);
        if (activeAudio && Number.isFinite(value)) {
          try { activeAudio.currentTime = value; } catch (err) {}
        }
        setRangeProgress(scrub, value, scrub.max);
        updateTransitionScrubber();
      });
      scrub.addEventListener('change', () => {
        transitionScrubDragging = false;
        const activeAudio = transitionAudioElement();
        const value = Number(scrub.value || 0);
        if (activeAudio && Number.isFinite(value)) {
          try { activeAudio.currentTime = value; } catch (err) {}
        }
        updateTransitionScrubber();
      });
      scrub.addEventListener('pointerup', () => { transitionScrubDragging = false; updateTransitionScrubber(); });
      scrub.addEventListener('pointercancel', () => { transitionScrubDragging = false; updateTransitionScrubber(); });
    }
    if (audio && !audio.__seqParentBound) {
      audio.__seqParentBound = true;
      ['play', 'pause', 'ended', 'timeupdate', 'loadedmetadata', 'seeked'].forEach(name => {
        audio.addEventListener(name, updateTransitionScrubber);
      });
    }
    updateTransitionScrubber();
  }
  function applyTransitionEditorDrag(ev) {
    if (!transitionEditorDrag) return;
    const info = transitionWindowInfo(transitionEditorDrag.deck);
    if (!info) return;
    const dx = Number(ev.clientX) - transitionEditorDrag.startX;
    const secondsPerPixel = transitionEditorDrag.secondsPerPixel;
    const deltaSeconds = dx * secondsPerPixel;
    let nudge = transitionEditorDrag.startNudge + (deltaSeconds / info.beatSec);
    nudge = transitionSnapToBeat ? Math.round(nudge) : Math.round(nudge * 100) / 100;
    nudge = clampTransitionNudge(transitionEditorDrag.deck, nudge);
    previewFormState[info.cfg.nudgeField] = nudge;
    syncPreviewFieldElements(info.cfg.nudgeField, nudge);
    drawTransitionEditor();
    updatePreviewEffectDisplay({ dirty: true });
  }
  function bindTransitionEditor() {
    const snap = document.getElementById('transition-snap-to-beat');
    if (snap && !snap.__seqBound) {
      snap.__seqBound = true;
      snap.addEventListener('change', () => {
        transitionSnapToBeat = !!snap.checked;
        drawTransitionEditor();
      });
    }
    document.querySelectorAll('[data-transition-surface][data-transition-deck]').forEach(canvas => {
      if (canvas.__seqBound) return;
      canvas.__seqBound = true;
      canvas.addEventListener('pointerdown', ev => {
        if (ev.button !== 0) return;
        const deck = canvas.getAttribute('data-transition-deck');
        const info = transitionWindowInfo(deck);
        if (!info) return;
        const cue = transitionCueAtPoint(canvas, info, ev);
        if (cue) {
          previewFormState[info.cfg.cueField] = cue.name || '';
          syncPreviewFieldElements(info.cfg.cueField, previewFormState[info.cfg.cueField]);
          drawTransitionEditor();
          updatePreviewEffectDisplay({ dirty: true });
          ev.preventDefault();
          ev.stopPropagation();
          return;
        }
        const rect = canvas.getBoundingClientRect();
        const surface = canvas.getAttribute('data-transition-surface');
        const view = transitionVisibleRange(info, surface);
        const seconds = view.duration;
        transitionEditorDrag = {
          deck,
          startX: Number(ev.clientX),
          startNudge: Number(previewFormState[info.cfg.nudgeField]) || 0,
          secondsPerPixel: seconds / Math.max(1, rect.width),
        };
        try { canvas.setPointerCapture(ev.pointerId); } catch (err) {}
        ev.preventDefault();
        ev.stopPropagation();
      });
      canvas.addEventListener('pointermove', ev => {
        if (!transitionEditorDrag || transitionEditorDrag.deck !== canvas.getAttribute('data-transition-deck')) return;
        applyTransitionEditorDrag(ev);
        ev.preventDefault();
      });
      const endDrag = ev => {
        if (!transitionEditorDrag) return;
        applyTransitionEditorDrag(ev);
        transitionEditorDrag = null;
        try { canvas.releasePointerCapture(ev.pointerId); } catch (err) {}
        updateTransitionDirtyUi();
      };
      canvas.addEventListener('pointerup', endDrag);
      canvas.addEventListener('pointercancel', () => { transitionEditorDrag = null; });
    });
    drawTransitionEditor();
  }
  function bindTransitionPreviewOverlay() {
    const renderButton = document.getElementById('transition-render-button');
    if (renderButton) {
      renderButton.onclick = window.__djRenderTransition;
    }
    setTimeout(() => safeUi('transition editor bind', bindTransitionEditor), 0);
    setTimeout(() => safeUi('transition scrubber bind', bindTransitionScrubber), 0);
    setTimeout(() => safeUi('transition editor draw', drawTransitionEditor), 0);
    setTimeout(() => safeUi('transition dirty ui update', updateTransitionDirtyUi), 0);
  }
  function updatePreviewFormStateFromElement(el) {
    const field = el && el.getAttribute && el.getAttribute('data-preview-field');
    if (!field) return;
    if (field === 'preset') {
      applyPresetToPreviewState(el.value);
      updateTransitionDirtyUi();
      return;
    }
    if (el.type === 'number') previewFormState[field] = Number(el.value);
    else previewFormState[field] = el.value;
    if (field === 'from_nudge_beats') previewFormState[field] = clampTransitionNudge('from', previewFormState[field]);
    if (field === 'to_nudge_beats') previewFormState[field] = clampTransitionNudge('to', previewFormState[field]);
    if ((field === 'from_nudge_beats' || field === 'to_nudge_beats') && el) el.value = String(previewFormState[field]);
    syncPreviewFieldElements(field, previewFormState[field], el);
    if (['volume_mode', 'eq_mode', 'filter_mode'].includes(field) && previewFormState.preset !== 'custom') {
      previewFormState.preset = 'custom';
      const presetEl = document.getElementById('preset');
      if (presetEl) presetEl.value = 'custom';
    }
    drawTransitionEditor();
  }
  function syncPreviewFormStateFromDom() {
    document.querySelectorAll('[data-preview-field]').forEach(updatePreviewFormStateFromElement);
  }
  async function renderBackendTransition() {
    if (transitionRenderPending) return;
    const from = transitionFromIdx === null ? null : byIdx.get(transitionFromIdx);
    const to = transitionToIdx === null ? null : byIdx.get(transitionToIdx);
    if (!from || !to) {
      transitionRenderMessage = 'Assign Track 1 and Track 2 before rendering.';
      transitionRenderWarn = true;
      renderTransitionPreview();
      return;
    }
    syncPreviewFormStateFromDom();
    const body = Object.assign({}, previewFormState, {
      from_track: trackId(from),
      to_track: trackId(to),
      overwrite: true,
    });
    const requestId = transitionRenderRequestId + 1;
    transitionRenderRequestId = requestId;
    transitionRenderPending = true;
    transitionRenderMessage = 'Rendering transition...';
    transitionRenderWarn = false;
    showTransitionPendingUi();
    try {
      const res = await fetch('/api/render-transition', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (requestId !== transitionRenderRequestId) return;
      if (!data.ok) throw new Error(data.error || 'Render failed');
      data.rendered_at = Date.now();
      transitionRender = data;
      transitionRenderedState = Object.assign({}, body);
      transitionRenderMessage = 'Rendered preview is current.';
      transitionRenderWarn = false;
    } catch (err) {
      if (requestId !== transitionRenderRequestId) return;
      transitionRenderMessage = err.message || String(err);
      transitionRenderWarn = true;
    } finally {
      if (requestId === transitionRenderRequestId) {
        transitionRenderPending = false;
        renderTransitionPreview();
      }
    }
  }
  window.__djRenderTransition = function(ev) {
    if (ev) {
      ev.preventDefault();
      ev.stopPropagation();
    }
    renderBackendTransition();
    return false;
  };
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
  function updateAppendControls() {
    const slot = targetSlot();
    const actionLabel = slot >= 0 && sequence[slot] !== null ? 'Replace' : 'Append';
    els.appendSelected.textContent = actionLabel;
    els.appendSelected.disabled = selectedIdx === null || !canPlaceTrack(selectedIdx, slot);
  }
  function renderSelectionOnly() {
    safeUi('map overlays', updatePacmapOverlays);
    safeUi('append controls', updateAppendControls);
    if (activePane === 'explore') {
      safeUi('sequence render', renderSequence);
      safeUi('recommendations render', renderRecommendations);
    }
    if (activePane === 'library') safeUi('library render', renderLibrary);
    safeUi('transition badges render', renderTransitionBadges);
    safeUi('audio ui sync', syncAudioUi);
  }
  function renderAll() {
    safeUi('weight labels', updateWeightLabels);
    safeUi('map coordinates', () => {
      const previousPoints = currentPoints;
      currentPoints = layoutInterpolatedPoints(weights());
      recordMapTrails(previousPoints, currentPoints);
      updatePointCoordinates();
      applyPointColorMode();
      updatePacmapOverlays();
      ensureMapEffectsLoop();
    });
    safeUi('append controls', updateAppendControls);
    if (activePane === 'explore') {
      safeUi('sequence render', renderSequence);
      safeUi('recommendations render', renderRecommendations);
      safeUi('energy curve render', renderEnergyCurve);
    }
    if (activePane === 'library') safeUi('library render', renderLibrary);
    if (activePane === 'diagnostics') {
      safeUi('transition score render', renderCurrentTransitionScore);
      safeUi('transition diagnostics render', renderTransitionDiagnostics);
    }
    if (activePane === 'preview') safeUi('transition preview render', renderTransitionPreview);
    safeUi('transition badges render', renderTransitionBadges);
    safeUi('audio ui sync', syncAudioUi);
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
  if (config.control_mode === 'genre-mixability') {
    if (els.weightStyle) els.weightStyle.value = String((config.weights && config.weights.maest) || 0.45);
    setMixSliders({
      tempo: (config.weights && config.weights.mix_tempo) || 0.34,
      groove: (config.weights && config.weights.mix_groove) || 0.33,
      chroma: (config.weights && config.weights.mix_chroma) || 0.33,
    });
  } else {
    if (els.weightMaest) els.weightMaest.value = String((config.weights && config.weights.maest) || 0.6);
    if (els.weightChroma) els.weightChroma.value = String((config.weights && config.weights.chroma) || 0.25);
    if (els.weightTempo) els.weightTempo.value = String((config.weights && config.weights.tempo) || 0.15);
  }
  els.tabButtons.forEach(btn => btn.addEventListener('click', () => setActivePane(btn.getAttribute('data-pane-tab') || 'explore')));
  els.sequenceLength.addEventListener('change', () => setSequenceLength(els.sequenceLength.value));
  els.energySource.addEventListener('change', renderAll);
  els.colorMode.addEventListener('change', renderAll);
  if (els.mapEffectsEnabled) els.mapEffectsEnabled.addEventListener('change', () => {
    if (!els.mapEffectsEnabled.checked && els.mapEffects) {
      const ctx = els.mapEffects.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, els.mapEffects.width, els.mapEffects.height);
      mapTrailSegments = [];
    }
    renderAll();
  });
  els.penaltyScale.addEventListener('input', renderAll);
  if (els.weightStyle) els.weightStyle.addEventListener('input', renderAll);
  if (els.weightMaest) els.weightMaest.addEventListener('input', renderAll);
  [els.weightChroma, els.weightTempo, els.weightGroove].forEach(el => {
    if (!el) return;
    el.addEventListener('input', () => {
      if (config.control_mode === 'genre-mixability') setMixSliders(mixWeightsRaw());
      renderAll();
    });
  });
  if (els.simplex) {
    let draggingSimplex = false;
    els.simplex.addEventListener('pointerdown', ev => { draggingSimplex = true; els.simplex.setPointerCapture(ev.pointerId); setWeightsFromSimplexEvent(ev); });
    els.simplex.addEventListener('pointermove', ev => { if (draggingSimplex) setWeightsFromSimplexEvent(ev); });
    els.simplex.addEventListener('pointerup', ev => { draggingSimplex = false; try { els.simplex.releasePointerCapture(ev.pointerId); } catch (err) {} });
    els.simplex.addEventListener('pointercancel', () => { draggingSimplex = false; });
  }
  if (els.setOutgoing) els.setOutgoing.addEventListener('click', () => { if (selectedIdx !== null) setTransitionEndpoint('out', selectedIdx); });
  if (els.setIncoming) els.setIncoming.addEventListener('click', () => { if (selectedIdx !== null) setTransitionEndpoint('in', selectedIdx); });
  els.appendSelected.addEventListener('click', () => { if (selectedIdx !== null) appendTrack(selectedIdx); });
  if (els.clearTransition) els.clearTransition.addEventListener('click', clearTransitionPair);
  els.clearLast.addEventListener('click', () => {
    for (let i = sequence.length - 1; i >= 0; i -= 1) { if (sequence[i] !== null) { sequence[i] = null; break; } }
    renderAll();
  });
  els.resetSequence.addEventListener('click', () => { sequence = Array.from({ length: sequenceLength }, () => null); selectedSlot = null; renderAll(); });
  els.downloadSequence.addEventListener('click', downloadCsv);
  if (els.librarySearch) els.librarySearch.addEventListener('input', () => {
    libraryQuery = els.librarySearch.value || '';
    renderLibrary();
  });
  if (els.audio) {
    applyMasterVolume();
    els.audio.addEventListener('play', () => {
      previewAudioPlaying = true;
      syncAudioUi();
      requestAudioProgressFrame();
    });
    els.audio.addEventListener('pause', () => {
      previewAudioPlaying = false;
      syncAudioUi();
    });
    els.audio.addEventListener('ended', () => {
      previewAudioPlaying = false;
      syncAudioUi();
    });
    els.audio.addEventListener('timeupdate', () => {
      if (previewAudioIdx !== null) rowScrubPositions.set(Number(previewAudioIdx), Number(els.audio.currentTime || 0));
      syncAudioUi();
    });
    els.audio.addEventListener('loadedmetadata', syncAudioUi);
    els.audio.addEventListener('seeked', syncAudioUi);
  }
  if (els.songPlayerToggle) {
    els.songPlayerToggle.addEventListener('click', () => {
      if (selectedIdx === null || !byIdx.has(selectedIdx)) return;
      if (previewAudioIdx === selectedIdx && previewAudioContext === 'main' && previewAudioPlaying) pauseSharedAudio();
      else {
        const record = byIdx.get(selectedIdx);
        const start = previewAudioIdx === selectedIdx && els.audio ? Number(els.audio.currentTime || 0) : previewStart(record);
        playRecordAt(record, start, 'main');
      }
    });
  }
  if (els.songWaveform) {
    els.songWaveform.addEventListener('pointerdown', ev => {
      waveformPointerActive = true;
      try { els.songWaveform.setPointerCapture(ev.pointerId); } catch (err) {}
      seekMainWaveformFromEvent(ev, true);
    });
    els.songWaveform.addEventListener('pointermove', ev => {
      if (waveformPointerActive) seekMainWaveformFromEvent(ev, true);
    });
    els.songWaveform.addEventListener('pointerup', ev => {
      waveformPointerActive = false;
      try { els.songWaveform.releasePointerCapture(ev.pointerId); } catch (err) {}
      seekMainWaveformFromEvent(ev, true);
    });
    els.songWaveform.addEventListener('pointercancel', () => {
      waveformPointerActive = false;
    });
  }
  if (els.globalToggle) {
    els.globalToggle.addEventListener('click', () => {
      const record = currentAudioRecord();
      if (!record) return;
      if (previewAudioIdx === Number(record.idx) && previewAudioPlaying) {
        pauseSharedAudio();
        return;
      }
      const start = previewAudioIdx === Number(record.idx) && els.audio
        ? Number(els.audio.currentTime || 0)
        : (rowScrubPositions.has(Number(record.idx)) ? rowScrubValue(record) : previewStart(record));
      playRecordAt(record, start, 'global');
    });
  }
  if (els.globalScrub) {
    const handleGlobalScrub = () => {
      const record = currentAudioRecord();
      if (!record) return;
      const idx = Number(record.idx);
      const value = Number(els.globalScrub.value || 0);
      if (!Number.isFinite(idx) || !Number.isFinite(value)) return;
      rowScrubPositions.set(idx, value);
      if (previewAudioIdx === idx && els.audio) seekSharedAudio(value);
      updateRowScrubbers();
      updateGlobalPlayer();
      drawMainWaveform();
    };
    els.globalScrub.addEventListener('input', handleGlobalScrub);
    els.globalScrub.addEventListener('change', handleGlobalScrub);
  }
  if (els.globalVolume) {
    masterVolume = clamp(Number(els.globalVolume.value || masterVolume), 0, 1);
    applyMasterVolume();
    els.globalVolume.addEventListener('input', () => {
      masterVolume = clamp(Number(els.globalVolume.value), 0, 1);
      applyMasterVolume();
    });
    els.globalVolume.addEventListener('change', () => {
      masterVolume = clamp(Number(els.globalVolume.value), 0, 1);
      applyMasterVolume();
    });
  }
  if (els.globalTrack1) els.globalTrack1.addEventListener('click', () => {
    const record = currentAudioRecord();
    if (record) setTransitionEndpoint('out', Number(record.idx), { toggle: false });
  });
  if (els.globalTrack2) els.globalTrack2.addEventListener('click', () => {
    const record = currentAudioRecord();
    if (record) setTransitionEndpoint('in', Number(record.idx), { toggle: false });
  });
  document.body.addEventListener('pointerdown', ev => {
    const wave = closestEl(ev.target, '[data-library-waveform-idx]');
    if (!wave) return;
    rowWaveformPointerIdx = Number(wave.getAttribute('data-library-waveform-idx'));
    try { wave.setPointerCapture(ev.pointerId); } catch (err) {}
    seekLibraryWaveformFromEvent(ev, rowWaveformPointerIdx, { play: true });
    ev.preventDefault();
    ev.stopPropagation();
  });
  document.body.addEventListener('pointermove', ev => {
    if (rowWaveformPointerIdx === null) return;
    seekLibraryWaveformFromEvent(ev, rowWaveformPointerIdx, { play: false });
    ev.preventDefault();
  });
  document.body.addEventListener('pointerup', ev => {
    if (rowWaveformPointerIdx === null) return;
    seekLibraryWaveformFromEvent(ev, rowWaveformPointerIdx, { play: false });
    rowWaveformPointerIdx = null;
    ev.preventDefault();
  });
  document.body.addEventListener('pointercancel', () => {
    rowWaveformPointerIdx = null;
  });
  window.addEventListener('resize', () => setTimeout(() => safeUi('transition editor draw', drawTransitionEditor), 30));
  window.addEventListener('resize', () => setTimeout(() => safeUi('main waveform draw', drawMainWaveform), 30));
  window.addEventListener('resize', () => setTimeout(() => safeUi('library waveform draw', drawLibraryWaveforms), 30));
  window.addEventListener('resize', () => setTimeout(() => safeUi('transition editor draw', drawTransitionEditor), 30));

  document.body.addEventListener('click', ev => {
    const appendIdx = closestAttr(ev.target, 'data-append-idx');
    const removeSlotValue = closestAttr(ev.target, 'data-remove-slot');
    const seqSlotValue = closestAttr(ev.target, 'data-seq-slot');
    const placeSlotValue = closestAttr(ev.target, 'data-place-slot');
    const playButton = closestEl(ev.target, '[data-play-idx]');
    const playIdx = playButton ? playButton.getAttribute('data-play-idx') : null;
    const playMode = playButton ? (playButton.getAttribute('data-play-mode') || 'point') : 'point';
    const libraryCurrent = closestAttr(ev.target, 'data-library-current');
    const libraryOutgoing = closestAttr(ev.target, 'data-library-outgoing');
    const libraryIncoming = closestAttr(ev.target, 'data-library-incoming');
    const libraryPlace = closestAttr(ev.target, 'data-library-place');
    const librarySortKey = closestAttr(ev.target, 'data-library-sort');
    const previewAction = closestAttr(ev.target, 'data-preview-action');
    const transitionTrackPlay = closestAttr(ev.target, 'data-transition-track-play');
    if (librarySortKey !== null) {
      if (librarySort !== librarySortKey) {
        librarySort = librarySortKey;
        libraryDir = 'asc';
      } else if (libraryDir === 'asc') {
        libraryDir = 'desc';
      } else {
        librarySort = null;
        libraryDir = null;
      }
      renderLibrary();
    }
    if (appendIdx !== null) appendTrack(Number(appendIdx));
    if (removeSlotValue !== null) removeSlot(Number(removeSlotValue));
    if (seqSlotValue !== null) { selectedSlot = Number(seqSlotValue); renderAll(); }
    if (placeSlotValue !== null && selectedIdx !== null) { selectedSlot = Number(placeSlotValue); appendTrack(selectedIdx); }
    if (playIdx !== null) toggleTrackPreview(Number(playIdx), { mode: playMode });
    if (libraryCurrent !== null) setCurrentTrack(Number(libraryCurrent));
    if (libraryOutgoing !== null) setTransitionEndpoint('out', Number(libraryOutgoing), { toggle: false });
    if (libraryIncoming !== null) setTransitionEndpoint('in', Number(libraryIncoming), { toggle: false });
    if (libraryPlace !== null) appendTrack(Number(libraryPlace));
    if (previewAction === 'swap') swapTransitionPair();
    if (previewAction === 'clear') clearTransitionPair();
    if (previewAction === 'render') renderBackendTransition();
    if (transitionTrackPlay !== null) toggleTransitionTrackPreview(transitionTrackPlay);
  });
  document.body.addEventListener('change', ev => {
    if (updateRecFiltersFromElement(ev.target)) {
      renderRecommendations();
      updatePacmapOverlays();
      ensureMapEffectsLoop();
      return;
    }
    if (ev.target && ev.target.getAttribute && ev.target.getAttribute('data-preview-field')) {
      updatePreviewFormStateFromElement(ev.target);
      updatePreviewEffectDisplay({ dirty: true });
      return;
    }
    const slotValue = ev.target && ev.target.getAttribute && ev.target.getAttribute('data-target-slot');
    if (slotValue === null) return;
    const slot = Number(slotValue);
    const val = Number(ev.target.value);
    if (Number.isFinite(val)) targetValues[slot] = clamp(val, 1, 9);
    else targetValues[slot] = null;
    selectedSlot = Number.isFinite(slot) ? slot : selectedSlot;
    renderAll();
  });
  document.body.addEventListener('input', ev => {
    if (updateRecFiltersFromElement(ev.target)) {
      renderRecommendations();
      updatePacmapOverlays();
      ensureMapEffectsLoop();
      return;
    }
    if (ev.target && ev.target.getAttribute && ev.target.getAttribute('data-preview-field')) {
      updatePreviewFormStateFromElement(ev.target);
      updatePreviewEffectDisplay({ dirty: true });
      return;
    }
    const scrubIdx = ev.target && ev.target.getAttribute && ev.target.getAttribute('data-library-scrub-idx');
    if (scrubIdx !== null) {
      const idx = Number(scrubIdx);
      const value = Number(ev.target.value || 0);
      if (Number.isFinite(idx) && Number.isFinite(value)) {
        rowScrubPositions.set(idx, value);
        if (previewAudioIdx === idx && els.audio) seekSharedAudio(value);
        updateRowScrubbers();
      }
    }
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
      if (Number.isFinite(idx)) {
        const now = Date.now();
        lastPlotPointClickMs = now;
        const nativeDetail = Number(ev.event && ev.event.detail) || 0;
        const isDouble = nativeDetail >= 2 || (lastClickedIdx === idx && (now - lastClickMs) < 520);
        lastClickedIdx = idx;
        lastClickMs = now;
        selectTrack(idx);
        if (isDouble) assignTransitionFromDoubleClick(idx);
      }
    });
    plot.on('plotly_hover', ev => {
      if (!ev || !ev.points || !ev.points.length) return;
      const c = ev.points[0].customdata || [];
      const idx = Number(c[0]);
      if (Number.isFinite(idx) && byIdx.has(idx)) showSongHover(byIdx.get(idx), ev.event);
    });
    plot.on('plotly_unhover', hideSongHover);
    plot.on('plotly_doubleclick', () => {
      if ((Date.now() - lastPointDoubleClickMs) < 900) return false;
      if (lastClickedIdx !== null && (Date.now() - lastClickMs) < 720) {
        assignTransitionFromDoubleClick(lastClickedIdx);
        return false;
      }
      if (transitionFromIdx !== null || transitionToIdx !== null) clearTransitionPair();
      return false;
    });
    plot.addEventListener('dblclick', ev => {
      const onPoint = isPlotPointTarget(ev.target);
      handlePlotDoubleClick(onPoint);
      ev.preventDefault();
      ev.stopPropagation();
    });
    plot.addEventListener('click', ev => {
      if (Number(ev.detail || 0) > 1) return;
      const onPoint = isPlotPointTarget(ev.target);
      if (onPoint) return;
      const nearTrackPoint = plotClickNearTrackPoint(ev);
      if (nearTrackPoint) return;
      const clearRequestTime = Date.now();
      const selectedAtClick = selectedIdx;
      window.setTimeout(() => {
        if (lastPlotPointClickMs >= clearRequestTime - 40) return;
        if (selectedIdx !== selectedAtClick) return;
        if (selectedIdx === null) return;
        clearCurrentTrackSelection();
      }, 650);
    });
  }
  setActivePane('explore', { render: false });
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
            f'<header class="app-header"><h1>{title}</h1>{tabs_html}</header>',
            controls_html,
            data_scripts,
            script,
            "</body>",
            "</html>",
        ]
    )


def export_dj_sequence(
    *,
    project_root: Path = PROJECT_ROOT,
    output_file: Path | None = None,
    mix_slugs: list[str] | None = None,
    energy_npz_path: Path | None = None,
    default_length: int = 10,
    control_mode: str = "genre-mixability",
    step: float = 0.1,
    static_layout: bool = False,
    open_browser: bool = False,
    app_mode: bool = False,
) -> Path:
    project_root = project_root.expanduser().resolve()
    if control_mode not in CONTROL_MODE_CHOICES:
        raise ValueError(f"control_mode must be one of {CONTROL_MODE_CHOICES}, got {control_mode!r}.")
    mix_slugs = mix_slugs or ["aries-mix", "ara-mix"]
    output_file = output_file or (project_root / "data" / "exports" / "dj_sequence_builder.html")
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
    default_weights = (
        {"maest": 0.45, "mix_tempo": 0.34, "mix_groove": 0.33, "mix_chroma": 0.33}
        if control_mode == "genre-mixability"
        else {"maest": 0.60, "chroma": 0.25, "tempo": 0.15}
    )
    groove_embeddings = _load_combined_groove_embeddings(
        project_root=project_root,
        mix_slugs=mix_slugs,
        groove_dir=project_root / "data" / "groove_embeddings",
    )
    matrices = _component_matrices_4way(
        features,
        groove_embeddings=groove_embeddings,
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
    similarity_payload = _build_similarity_payload(
        records=records,
        temperature=0.08,
        maest_similarity=matrices["maest_similarity"],
        tempo_similarity=matrices["tempo_similarity"],
        groove_similarity=matrices["groove_similarity"],
        chroma_similarity=matrices["chroma_similarity"],
        maest_distance=matrices["maest_distance"],
        tempo_distance=matrices["tempo_distance"],
        groove_distance=matrices["groove_distance"],
        chroma_distance=matrices["chroma_distance"],
    )
    layouts = None
    coords: np.ndarray
    if control_mode == "genre-mixability" and static_layout:
        static_weights = _default_genre_mixability_weights(default_weights)
        static_layouts = _compute_pacmap_knn_4way_layouts(
            X_reference=features.maest,
            D_maest=matrices["maest_distance"],
            D_tempo=matrices["tempo_distance"],
            D_groove=matrices["groove_distance"],
            D_chroma=matrices["chroma_distance"],
            grid=[static_weights],
            n_neighbors=10,
            mn_ratio=0.5,
            fp_ratio=1.5,
            distance="angular",
            random_state=7777,
            align=True,
            pair_source="neighbors-only",
            distance_combine="l2",
            layout_init="neighbor",
        )
        coords = np.asarray(next(iter(static_layouts.values())), dtype=np.float32)
    elif control_mode == "genre-mixability":
        layouts = _compute_pacmap_knn_4way_layouts(
            X_reference=features.maest,
            D_maest=matrices["maest_distance"],
            D_tempo=matrices["tempo_distance"],
            D_groove=matrices["groove_distance"],
            D_chroma=matrices["chroma_distance"],
            grid=_simplex_grid_4way(step),
            n_neighbors=10,
            mn_ratio=0.5,
            fp_ratio=1.5,
            distance="angular",
            random_state=7777,
            align=True,
            pair_source="neighbors-only",
            distance_combine="l2",
            layout_init="neighbor",
        )
        coords = np.asarray(next(iter(layouts.values())), dtype=np.float32)
    else:
        coords = _compute_pacmap_coords(features, random_state=7777, n_neighbors=10)

    title = "DJ Sequence Builder"
    plot_div_id = "dj_sequence_builder_pacmap"
    plot_html = _build_plot(records, coords, plot_div_id=plot_div_id, title="")
    html = _build_html(
        plot_html=plot_html,
        records=records,
        coords=coords,
        layouts=layouts,
        similarity_payload=similarity_payload,
        plot_div_id=plot_div_id,
        title=title,
        default_length=default_length,
        default_weights=default_weights,
        control_mode=control_mode,
        static_layout=static_layout,
        temperature=0.08,
        top_k_rows=25,
        app_mode=bool(app_mode),
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
    parser.add_argument("--control-mode", choices=CONTROL_MODE_CHOICES, default="genre-mixability")
    parser.add_argument("--step", type=float, default=0.1, help="Simplex grid step for dynamic PaCMAP layouts.")
    parser.add_argument(
        "--static",
        action="store_true",
        dest="static_layout",
        help="Keep the PaCMAP layout fixed at the default weights while controls still update transition scoring.",
    )
    parser.add_argument("--open", action="store_true", help="Open the exported HTML in a browser.")
    args = parser.parse_args(argv)

    output_file = export_dj_sequence(
        output_file=args.output_file,
        energy_npz_path=args.energy_npz,
        default_length=args.sequence_length,
        control_mode=args.control_mode,
        step=args.step,
        static_layout=bool(args.static_layout),
        mix_slugs=args.mix_slugs,
        open_browser=bool(args.open),
    )
    print(f"Saved DJ sequence builder HTML: {output_file}")
    return 0


export_energy_sequence_builder = export_dj_sequence


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
