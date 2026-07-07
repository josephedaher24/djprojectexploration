"""Export an interactive PaCMAP energy sequence builder as standalone HTML."""

from __future__ import annotations

import argparse
import hashlib
import html
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
from djprojectexploration.frontend_assets import frontend_asset_text, render_standalone_document
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
from djprojectexploration.tracklists import read_csv_rows
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

        csv_rows = read_csv_rows(tracklist_csv)
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
        weight_controls_html = frontend_asset_text("templates/energy_sequence_builder_legacy_weights.html")
    else:
        weight_controls_html = frontend_asset_text("templates/energy_sequence_builder_mixability_weights.html")

    body_html = (
        frontend_asset_text("templates/energy_sequence_builder_body.html")
        .replace("{{TITLE}}", html.escape(title))
        .replace("{{PLOT_HTML}}", plot_html)
        .replace("{{WEIGHT_CONTROLS_HTML}}", weight_controls_html)
    )

    return render_standalone_document(
        title=title,
        css_asset="static/energy_sequence_builder.css",
        body_html=body_html,
        data_scripts=data_scripts,
        script_asset="static/energy_sequence_builder.js",
        script_replacements={"__PLOT_ID__": plot_div_id},
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
