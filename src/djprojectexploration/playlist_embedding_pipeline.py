"""Playlist embedding pipelines for MAEST, chroma, tempo, and DEAM.

This module moves notebook-style embedding generation into reusable, scriptable
functions that export a single compressed NPZ collection per playlist.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAEST_OUTPUT_DIR = PROJECT_ROOT / "data" / "maest_embeddings"
DEFAULT_CHROMA_OUTPUT_DIR = PROJECT_ROOT / "data" / "chroma_embeddings"
DEFAULT_TEMPO_OUTPUT_DIR = PROJECT_ROOT / "data" / "tempo_embeddings"
DEFAULT_DEAM_OUTPUT_DIR = PROJECT_ROOT / "data" / "deam_embeddings"
DEFAULT_MODEL_FILENAME = "discogs-maest-30s-pw-519l-2.pb"
DEFAULT_OUTPUT_NODE = "PartitionedCall/Identity_7"
DEFAULT_MODEL_FILE = PROJECT_ROOT / "models" / DEFAULT_MODEL_FILENAME


@dataclass(frozen=True)
class PlaylistTrack:
    """Normalized track metadata row from a playlist CSV."""

    track_number: int
    title: str
    artists: str
    mp3_name: str
    audio_path: Path
    genre: str
    key: str
    bpm: float | None
    onset_time: float | None
    key_shift: float | None


def _to_project_relpath(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _string_array(values: list[str]) -> np.ndarray:
    # Ensure explicit unicode dtype (not object/pickle-dependent).
    return np.asarray(values, dtype=np.str_)


def _metadata_arrays(tracks: list[PlaylistTrack]) -> dict[str, np.ndarray]:
    return {
        "track_numbers": np.asarray([t.track_number for t in tracks], dtype=np.int32),
        "titles": _string_array([t.title for t in tracks]),
        "artists": _string_array([t.artists for t in tracks]),
        "filenames": _string_array([t.mp3_name for t in tracks]),
        "audio_paths": _string_array([_to_project_relpath(t.audio_path) for t in tracks]),
        "genres": _string_array([t.genre for t in tracks]),
        "keys": _string_array([t.key for t in tracks]),
        "bpm": np.asarray(
            [np.nan if t.bpm is None else float(t.bpm) for t in tracks],
            dtype=np.float32,
        ),
        "onset_time": np.asarray(
            [np.nan if t.onset_time is None else float(t.onset_time) for t in tracks],
            dtype=np.float32,
        ),
        "key_shift": np.asarray(
            [np.nan if t.key_shift is None else float(t.key_shift) for t in tracks],
            dtype=np.float32,
        ),
    }


def _default_npz_name(tracklist_csv: Path) -> str:
    stem = tracklist_csv.stem
    if stem.endswith("_tracks"):
        return f"{stem}.npz"
    return f"{stem}_tracks.npz"


def _resolve_audio_path(
    *,
    csv_dir: Path,
    music_dir: Path | None,
    mp3_name: str,
    filepath_raw: str,
) -> Path | None:
    if filepath_raw:
        return Path(filepath_raw).expanduser()

    if not mp3_name:
        return None

    filename_path = Path(mp3_name).expanduser()
    if filename_path.is_absolute():
        return filename_path

    base_dir = music_dir if music_dir is not None else csv_dir
    return base_dir / filename_path


def load_playlist_tracks(
    tracklist_csv: str | Path,
    *,
    music_dir: str | Path | None = None,
    skip_missing_audio: bool = False,
) -> list[PlaylistTrack]:
    """Load and normalize playlist tracks from CSV.

    Supported CSV schemas:
    - Notebook style: track_number,title,artists,mp3_name,key,bpm,onset-time,genre,key shift
    - Apple Music export style: name,artist,album,genre,bpm,filepath

    Track order is sorted by ``track_number`` when present; otherwise row order is
    used with synthetic 1-based numbering.
    """
    csv_path = Path(tracklist_csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Tracklist CSV not found: {csv_path}")

    resolved_music_dir = None
    if music_dir is not None:
        resolved_music_dir = Path(music_dir).expanduser().resolve()

    parsed_rows: list[tuple[int, int, PlaylistTrack]] = []
    skipped_missing: list[tuple[int, str]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=1):
            track_number_raw = (row.get("track_number") or row.get("#") or "").strip()
            if track_number_raw:
                try:
                    track_number = int(track_number_raw)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid track_number '{track_number_raw}' at row {row_index} in {csv_path}"
                    ) from exc
            else:
                track_number = row_index

            title = (row.get("title") or row.get("name") or "").strip()
            artists = (row.get("artists") or row.get("artist") or "").strip()
            genre = (row.get("genre") or "").strip()
            key = (row.get("key") or "").strip()

            bpm = _optional_float(row.get("bpm"))
            onset_time = _optional_float(row.get("onset-time") or row.get("onset_time") or row.get("onset"))
            key_shift = _optional_float(row.get("key shift") or row.get("key_shift"))

            mp3_name = (row.get("mp3_name") or "").strip()
            filepath_raw = (row.get("filepath") or row.get("location") or "").strip()
            audio_path = _resolve_audio_path(
                csv_dir=csv_path.parent,
                music_dir=resolved_music_dir,
                mp3_name=mp3_name,
                filepath_raw=filepath_raw,
            )

            if audio_path is None:
                raise ValueError(
                    f"Could not resolve audio path at row {row_index} in {csv_path}. "
                    "Need either filepath/location or mp3_name (+ music_dir/csv-dir)."
                )

            resolved_audio = audio_path.expanduser().resolve()

            if not mp3_name:
                mp3_name = resolved_audio.name
            if not title:
                title = Path(mp3_name).stem

            if not resolved_audio.exists():
                if skip_missing_audio:
                    skipped_missing.append((row_index, str(resolved_audio)))
                    continue
                raise FileNotFoundError(
                    f"Audio file not found for row {row_index}: {resolved_audio}. "
                    "Use --skip-missing-audio to continue without this track."
                )

            track = PlaylistTrack(
                track_number=track_number,
                title=title,
                artists=artists,
                mp3_name=mp3_name,
                audio_path=resolved_audio,
                genre=genre,
                key=key,
                bpm=bpm,
                onset_time=onset_time,
                key_shift=key_shift,
            )
            parsed_rows.append((track_number, row_index, track))

    if not parsed_rows:
        raise RuntimeError(
            f"No usable tracks found in {csv_path}."
            + (" All rows were missing audio files." if skipped_missing else "")
        )

    parsed_rows.sort(key=lambda item: (item[0], item[1]))
    tracks = [item[2] for item in parsed_rows]

    return tracks


def _save_collection_npz(
    *,
    output_file: Path,
    embedding_type: str,
    playlist_csv: Path,
    embeddings: np.ndarray,
    metadata: dict[str, np.ndarray],
    extra: dict[str, np.ndarray],
) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    created_utc = datetime.now(tz=timezone.utc).isoformat()

    payload: dict[str, Any] = {
        "embedding_type": np.array(embedding_type, dtype=np.str_),
        "playlist_csv": np.array(_to_project_relpath(playlist_csv), dtype=np.str_),
        "created_utc": np.array(created_utc, dtype=np.str_),
        "num_tracks": np.array(int(embeddings.shape[0]), dtype=np.int32),
        "embedding_dimension": np.array(int(embeddings.shape[1]), dtype=np.int32),
        "embeddings": embeddings.astype(np.float32),
    }
    payload.update(metadata)
    payload.update(extra)

    np.savez_compressed(output_file, **payload)
    return output_file


def create_maest_playlist_embeddings_npz(
    tracklist_csv: str | Path,
    *,
    music_dir: str | Path | None = None,
    model_file: str | Path = DEFAULT_MODEL_FILE,
    output_node: str = DEFAULT_OUTPUT_NODE,
    output_file: str | Path | None = None,
    output_dir: str | Path = DEFAULT_MAEST_OUTPUT_DIR,
    skip_missing_audio: bool = False,
) -> Path:
    """Build MAEST embeddings for playlist tracks and save one NPZ collection."""
    from djprojectexploration.maest_embedding_extractor import extract_embedding

    csv_path = Path(tracklist_csv).expanduser().resolve()
    tracks = load_playlist_tracks(
        csv_path,
        music_dir=music_dir,
        skip_missing_audio=skip_missing_audio,
    )

    resolved_model_file = Path(model_file).expanduser().resolve()
    if not resolved_model_file.exists():
        raise FileNotFoundError(
            f"MAEST model file not found: {resolved_model_file}. "
            "Download from https://essentia.upf.edu/models.html#MAEST or pass --model-file."
        )

    vectors: list[np.ndarray] = []
    reductions: list[str] = []
    raw_shapes: list[str] = []

    for track in tracks:
        vector, raw_shape, reduction = extract_embedding(track.audio_path, resolved_model_file, output_node)
        vectors.append(np.asarray(vector, dtype=np.float32).reshape(-1))
        reductions.append(reduction)
        raw_shapes.append("x".join(str(dim) for dim in raw_shape))

    embeddings = np.vstack(vectors).astype(np.float32)

    if output_file is None:
        resolved_output_dir = Path(output_dir).expanduser().resolve()
        resolved_output_file = resolved_output_dir / _default_npz_name(csv_path)
    else:
        resolved_output_file = Path(output_file).expanduser().resolve()

    metadata = _metadata_arrays(tracks)
    extra = {
        "maest_model_file": np.array(_to_project_relpath(resolved_model_file), dtype=np.str_),
        "maest_output_node": np.array(output_node, dtype=np.str_),
        "maest_reductions": _string_array(reductions),
        "maest_raw_prediction_shapes": _string_array(raw_shapes),
    }

    saved_path = _save_collection_npz(
        output_file=resolved_output_file,
        embedding_type="maest",
        playlist_csv=csv_path,
        embeddings=embeddings,
        metadata=metadata,
        extra=extra,
    )

    print(f"Saved MAEST playlist collection: {_to_project_relpath(saved_path)}")
    print(f"Tracks: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Model: {_to_project_relpath(resolved_model_file)}")
    return saved_path


def create_chroma_playlist_embeddings_npz(
    tracklist_csv: str | Path,
    *,
    music_dir: str | Path | None = None,
    output_file: str | Path | None = None,
    output_dir: str | Path = DEFAULT_CHROMA_OUTPUT_DIR,
    skip_missing_audio: bool = False,
    sample_rate: int = 44100,
    frame_size: int = 4096,
    hop_size: int = 1024,
    chroma_bins: int = 12,
    include_key_features: bool = True,
    center_baseline: float | None = 1.0 / 12.0,
) -> Path:
    """Build chroma embeddings for playlist tracks and save one NPZ collection."""
    from djprojectexploration.chroma_embedding import generate_chroma_embedding

    csv_path = Path(tracklist_csv).expanduser().resolve()
    tracks = load_playlist_tracks(
        csv_path,
        music_dir=music_dir,
        skip_missing_audio=skip_missing_audio,
    )

    vectors: list[np.ndarray] = []
    embedding_subtypes: list[str] = []
    base_dims: list[int] = []
    key_feature_dims: list[int] = []
    chroma_bins_values: list[int] = []
    detected_keys: list[str] = []
    detected_scales: list[str] = []
    detected_strengths: list[float] = []
    pitch_class_means: list[np.ndarray] = []
    pitch_class_stds: list[np.ndarray] = []
    beat_bpms: list[float] = []
    beat_counts: list[int] = []
    beat_sources: list[str] = []
    beat_phase_anchors: list[float] = []

    for track in tracks:
        payload = generate_chroma_embedding(
            audio_file=track.audio_path,
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
            chroma_bins=chroma_bins,
            include_key_features=include_key_features,
            center_baseline=center_baseline,
        )

        vectors.append(np.asarray(payload["embedding"], dtype=np.float32).reshape(-1))
        embedding_subtypes.append(str(payload.get("embedding_type", "unknown")))
        base_dims.append(int(payload.get("base_embedding_dimension", 0)))
        key_feature_dims.append(int(payload.get("key_feature_dimension", 0)))
        chroma_bins_values.append(int(payload.get("chroma_bins", chroma_bins)))

        key_estimate = payload.get("key_estimate", {})
        detected_keys.append(str(key_estimate.get("key", "")))
        detected_scales.append(str(key_estimate.get("scale", "")))
        detected_strengths.append(float(key_estimate.get("strength", 0.0)))
        pitch_class_means.append(np.asarray(payload.get("pitch_class_mean", []), dtype=np.float32).reshape(-1))
        pitch_class_stds.append(np.asarray(payload.get("pitch_class_std", []), dtype=np.float32).reshape(-1))

        beat_pooling = payload.get("beat_pooling", {})
        beat_bpms.append(float(beat_pooling.get("bpm", 0.0)))
        beat_counts.append(int(beat_pooling.get("beat_count", 0)))
        beat_sources.append(str(beat_pooling.get("beat_source", "")))
        phase_anchor = beat_pooling.get("phase_anchor_seconds")
        if phase_anchor is None:
            beat_phase_anchors.append(np.nan)
        else:
            beat_phase_anchors.append(float(phase_anchor))

    embeddings = np.vstack(vectors).astype(np.float32)
    pitch_mean_matrix = np.vstack(pitch_class_means).astype(np.float32)
    pitch_std_matrix = np.vstack(pitch_class_stds).astype(np.float32)

    if output_file is None:
        resolved_output_dir = Path(output_dir).expanduser().resolve()
        resolved_output_file = resolved_output_dir / _default_npz_name(csv_path)
    else:
        resolved_output_file = Path(output_file).expanduser().resolve()

    metadata = _metadata_arrays(tracks)
    extra = {
        "chroma_embedding_subtype": _string_array(embedding_subtypes),
        "chroma_base_embedding_dimension": np.asarray(base_dims, dtype=np.int32),
        "chroma_key_feature_dimension": np.asarray(key_feature_dims, dtype=np.int32),
        "chroma_bins": np.asarray(chroma_bins_values, dtype=np.int32),
        "chroma_pitch_class_mean": pitch_mean_matrix,
        "chroma_pitch_class_std": pitch_std_matrix,
        "detected_key": _string_array(detected_keys),
        "detected_scale": _string_array(detected_scales),
        "detected_key_strength": np.asarray(detected_strengths, dtype=np.float32),
        "beat_bpm": np.asarray(beat_bpms, dtype=np.float32),
        "beat_count": np.asarray(beat_counts, dtype=np.int32),
        "beat_source": _string_array(beat_sources),
        "beat_phase_anchor_seconds": np.asarray(beat_phase_anchors, dtype=np.float32),
        "config_sample_rate": np.array(sample_rate, dtype=np.int32),
        "config_frame_size": np.array(frame_size, dtype=np.int32),
        "config_hop_size": np.array(hop_size, dtype=np.int32),
        "config_include_key_features": np.array(bool(include_key_features), dtype=np.bool_),
        "config_center_baseline": np.array(
            np.nan if center_baseline is None else float(center_baseline),
            dtype=np.float32,
        ),
    }

    saved_path = _save_collection_npz(
        output_file=resolved_output_file,
        embedding_type="chroma",
        playlist_csv=csv_path,
        embeddings=embeddings,
        metadata=metadata,
        extra=extra,
    )

    print(f"Saved chroma playlist collection: {_to_project_relpath(saved_path)}")
    print(f"Tracks: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Include key features: {include_key_features}")
    print(f"Center baseline: {center_baseline}")
    return saved_path


def create_tempo_playlist_embeddings_npz(
    tracklist_csv: str | Path,
    *,
    music_dir: str | Path | None = None,
    model_file: str | Path | None = None,
    auto_download_model: bool = False,
    output_file: str | Path | None = None,
    output_dir: str | Path = DEFAULT_TEMPO_OUTPUT_DIR,
    skip_missing_audio: bool = False,
    sample_rate: int = 11025,
    resample_quality: int = 4,
    snippet_length_sec: float | None = None,
    window_sec: float = 12.0,
    hop_sec: float = 6.0,
    rms_percentile: float = 20.0,
) -> Path:
    """Build TempoCNN embeddings for playlist tracks and save one NPZ collection."""
    from djprojectexploration.tempo_embedding import (
        DEFAULT_TEMPOCNN_MODEL_URL,
        generate_tempo_embedding,
        resolve_tempocnn_model_file,
    )

    csv_path = Path(tracklist_csv).expanduser().resolve()
    tracks = load_playlist_tracks(
        csv_path,
        music_dir=music_dir,
        skip_missing_audio=skip_missing_audio,
    )

    resolved_model_file = resolve_tempocnn_model_file(
        model_file=model_file,
        auto_download=bool(auto_download_model),
    )

    vectors: list[np.ndarray] = []
    tempo_bpms: list[float] = []
    confidences: list[float] = []
    active_agreements: list[float] = []
    active_probabilities: list[float] = []
    active_fractions: list[float] = []
    active_windows: list[int] = []
    total_windows: list[int] = []

    local_start_index: list[int] = []
    local_counts: list[int] = []
    local_bpm_chunks: list[np.ndarray] = []
    local_prob_chunks: list[np.ndarray] = []
    local_time_chunks: list[np.ndarray] = []
    local_active_chunks: list[np.ndarray] = []

    cursor = 0
    for track in tracks:
        payload = generate_tempo_embedding(
            audio_file=track.audio_path,
            model_file=resolved_model_file,
            auto_download_model=False,
            sample_rate=int(sample_rate),
            resample_quality=int(resample_quality),
            snippet_length_sec=snippet_length_sec,
            window_sec=float(window_sec),
            hop_sec=float(hop_sec),
            rms_percentile=float(rms_percentile),
        )

        vector = np.asarray(payload["embedding"], dtype=np.float32).reshape(-1)
        vectors.append(vector)
        tempo_bpms.append(float(payload.get("tempo_bpm", np.nan)))
        confidences.append(float(payload.get("confidence", 0.0)))
        active_agreements.append(float(payload.get("confidence_active_agreement", 0.0)))
        active_probabilities.append(float(payload.get("mean_prob_active", 0.0)))
        active_fractions.append(float(payload.get("active_fraction", 0.0)))
        active_windows.append(int(payload.get("active_windows", 0)))
        total_windows.append(int(payload.get("total_windows", 0)))

        local_bpm = np.asarray(payload.get("local_bpm", []), dtype=np.float32).reshape(-1)
        local_prob = np.asarray(payload.get("local_probability", []), dtype=np.float32).reshape(-1)
        local_time = np.asarray(payload.get("local_times_sec", []), dtype=np.float32).reshape(-1)
        local_active = np.asarray(payload.get("local_active_mask", []), dtype=np.int8).reshape(-1)

        n_local = int(min(local_bpm.size, local_prob.size, local_time.size, local_active.size))
        local_bpm = local_bpm[:n_local]
        local_prob = local_prob[:n_local]
        local_time = local_time[:n_local]
        local_active = local_active[:n_local]

        local_start_index.append(cursor)
        local_counts.append(n_local)
        cursor += n_local

        local_bpm_chunks.append(local_bpm)
        local_prob_chunks.append(local_prob)
        local_time_chunks.append(local_time)
        local_active_chunks.append(local_active)

    embeddings = np.vstack(vectors).astype(np.float32)

    if output_file is None:
        resolved_output_dir = Path(output_dir).expanduser().resolve()
        resolved_output_file = resolved_output_dir / _default_npz_name(csv_path)
    else:
        resolved_output_file = Path(output_file).expanduser().resolve()

    metadata = _metadata_arrays(tracks)

    if cursor > 0:
        local_bpm_flat = np.concatenate(local_bpm_chunks).astype(np.float32)
        local_prob_flat = np.concatenate(local_prob_chunks).astype(np.float32)
        local_time_flat = np.concatenate(local_time_chunks).astype(np.float32)
        local_active_flat = np.concatenate(local_active_chunks).astype(np.int8)
    else:
        local_bpm_flat = np.array([], dtype=np.float32)
        local_prob_flat = np.array([], dtype=np.float32)
        local_time_flat = np.array([], dtype=np.float32)
        local_active_flat = np.array([], dtype=np.int8)

    extra = {
        "tempo_model_file": np.array(_to_project_relpath(resolved_model_file), dtype=np.str_),
        "tempo_model_url": np.array(DEFAULT_TEMPOCNN_MODEL_URL, dtype=np.str_),
        "tempo_bpm": np.asarray(tempo_bpms, dtype=np.float32),
        "tempo_confidence": np.asarray(confidences, dtype=np.float32),
        "tempo_confidence_active_agreement": np.asarray(active_agreements, dtype=np.float32),
        "tempo_mean_active_probability": np.asarray(active_probabilities, dtype=np.float32),
        "tempo_active_fraction": np.asarray(active_fractions, dtype=np.float32),
        "tempo_active_windows": np.asarray(active_windows, dtype=np.int32),
        "tempo_total_windows": np.asarray(total_windows, dtype=np.int32),
        "tempo_local_start_index": np.asarray(local_start_index, dtype=np.int64),
        "tempo_local_count": np.asarray(local_counts, dtype=np.int32),
        "tempo_local_bpm_flat": local_bpm_flat,
        "tempo_local_probability_flat": local_prob_flat,
        "tempo_local_times_sec_flat": local_time_flat,
        "tempo_local_active_mask_flat": local_active_flat,
        "config_sample_rate": np.array(sample_rate, dtype=np.int32),
        "config_resample_quality": np.array(resample_quality, dtype=np.int32),
        "config_snippet_length_sec": np.array(
            np.nan if snippet_length_sec is None else float(snippet_length_sec),
            dtype=np.float32,
        ),
        "config_window_sec": np.array(window_sec, dtype=np.float32),
        "config_hop_sec": np.array(hop_sec, dtype=np.float32),
        "config_rms_percentile": np.array(rms_percentile, dtype=np.float32),
    }

    saved_path = _save_collection_npz(
        output_file=resolved_output_file,
        embedding_type="tempo",
        playlist_csv=csv_path,
        embeddings=embeddings,
        metadata=metadata,
        extra=extra,
    )

    print(f"Saved tempo playlist collection: {_to_project_relpath(saved_path)}")
    print(f"Tracks: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Model: {_to_project_relpath(resolved_model_file)}")
    return saved_path


def create_deam_playlist_embeddings_npz(
    tracklist_csv: str | Path,
    *,
    music_dir: str | Path | None = None,
    embedding_backend: str = "musicnn",
    embedding_model_file: str | Path = PROJECT_ROOT / "models" / "msd-musicnn-1.pb",
    regression_model_file: str | Path = PROJECT_ROOT / "models" / "deam-msd-musicnn-2.pb",
    embedding_output: str | None = None,
    regression_output: str = "model/Identity",
    sample_rate: int = 16000,
    output_file: str | Path | None = None,
    output_dir: str | Path = DEFAULT_DEAM_OUTPUT_DIR,
    skip_missing_audio: bool = False,
) -> Path:
    """Build DEAM valence/arousal embeddings for playlist tracks and save one NPZ collection."""
    from djprojectexploration.deam_valence_arousal import DEAM_VALUE_RANGE, generate_deam_embedding

    csv_path = Path(tracklist_csv).expanduser().resolve()
    tracks = load_playlist_tracks(
        csv_path,
        music_dir=music_dir,
        skip_missing_audio=skip_missing_audio,
    )

    resolved_embedding_model = Path(embedding_model_file).expanduser().resolve()
    resolved_regression_model = Path(regression_model_file).expanduser().resolve()
    if not resolved_embedding_model.exists():
        raise FileNotFoundError(f"DEAM embedding model file not found: {resolved_embedding_model}")
    if not resolved_regression_model.exists():
        raise FileNotFoundError(f"DEAM regression model file not found: {resolved_regression_model}")

    vectors: list[np.ndarray] = []
    valence_means: list[float] = []
    valence_mins: list[float] = []
    valence_maxs: list[float] = []
    valence_stds: list[float] = []
    arousal_means: list[float] = []
    arousal_mins: list[float] = []
    arousal_maxs: list[float] = []
    arousal_stds: list[float] = []
    segment_counts: list[int] = []

    series_start_index: list[int] = []
    series_counts: list[int] = []
    valence_series_chunks: list[np.ndarray] = []
    arousal_series_chunks: list[np.ndarray] = []
    cursor = 0

    embedding_subtypes: list[str] = []
    embedding_labels: list[str] | None = None
    backend_from_payload: str | None = None

    for track in tracks:
        payload = generate_deam_embedding(
            audio_file=track.audio_path,
            embedding_model_file=resolved_embedding_model,
            regression_model_file=resolved_regression_model,
            embedding_backend=embedding_backend,
            embedding_output=embedding_output,
            regression_output=regression_output,
            sample_rate=int(sample_rate),
        )

        vector = np.asarray(payload.get("embedding", []), dtype=np.float32).reshape(-1)
        vectors.append(vector)
        embedding_subtypes.append(str(payload.get("embedding_subtype", "unknown")))
        if embedding_labels is None:
            embedding_labels = [str(v) for v in payload.get("embedding_labels", [])]
        if backend_from_payload is None:
            backend_from_payload = str(payload.get("embedding_backend", embedding_backend))

        track_prediction = payload.get("track_prediction", {})
        valence_means.append(float(track_prediction.get("valence", np.nan)))
        valence_mins.append(float(track_prediction.get("valence_min", np.nan)))
        valence_maxs.append(float(track_prediction.get("valence_max", np.nan)))
        valence_stds.append(float(track_prediction.get("valence_std", np.nan)))
        arousal_means.append(float(track_prediction.get("arousal", np.nan)))
        arousal_mins.append(float(track_prediction.get("arousal_min", np.nan)))
        arousal_maxs.append(float(track_prediction.get("arousal_max", np.nan)))
        arousal_stds.append(float(track_prediction.get("arousal_std", np.nan)))

        valence_series = np.asarray(payload.get("valence_series", []), dtype=np.float32).reshape(-1)
        arousal_series = np.asarray(payload.get("arousal_series", []), dtype=np.float32).reshape(-1)
        n_series = int(min(valence_series.size, arousal_series.size))
        valence_series = valence_series[:n_series]
        arousal_series = arousal_series[:n_series]

        segment_counts.append(n_series)
        series_start_index.append(cursor)
        series_counts.append(n_series)
        cursor += n_series

        valence_series_chunks.append(valence_series)
        arousal_series_chunks.append(arousal_series)

    embeddings = np.vstack(vectors).astype(np.float32)

    if output_file is None:
        resolved_output_dir = Path(output_dir).expanduser().resolve()
        resolved_output_file = resolved_output_dir / _default_npz_name(csv_path)
    else:
        resolved_output_file = Path(output_file).expanduser().resolve()

    metadata = _metadata_arrays(tracks)

    if cursor > 0:
        valence_series_flat = np.concatenate(valence_series_chunks).astype(np.float32)
        arousal_series_flat = np.concatenate(arousal_series_chunks).astype(np.float32)
    else:
        valence_series_flat = np.array([], dtype=np.float32)
        arousal_series_flat = np.array([], dtype=np.float32)

    extra = {
        "deam_backend": np.array(backend_from_payload or embedding_backend, dtype=np.str_),
        "deam_embedding_subtype": _string_array(embedding_subtypes),
        "deam_embedding_labels": _string_array(embedding_labels or []),
        "deam_value_range": np.asarray([DEAM_VALUE_RANGE[0], DEAM_VALUE_RANGE[1]], dtype=np.float32),
        "deam_num_segments": np.asarray(segment_counts, dtype=np.int32),
        "deam_valence_mean": np.asarray(valence_means, dtype=np.float32),
        "deam_valence_min": np.asarray(valence_mins, dtype=np.float32),
        "deam_valence_max": np.asarray(valence_maxs, dtype=np.float32),
        "deam_valence_std": np.asarray(valence_stds, dtype=np.float32),
        "deam_arousal_mean": np.asarray(arousal_means, dtype=np.float32),
        "deam_arousal_min": np.asarray(arousal_mins, dtype=np.float32),
        "deam_arousal_max": np.asarray(arousal_maxs, dtype=np.float32),
        "deam_arousal_std": np.asarray(arousal_stds, dtype=np.float32),
        "deam_series_start_index": np.asarray(series_start_index, dtype=np.int64),
        "deam_series_count": np.asarray(series_counts, dtype=np.int32),
        "deam_valence_series_flat": valence_series_flat,
        "deam_arousal_series_flat": arousal_series_flat,
        "deam_embedding_model_file": np.array(_to_project_relpath(resolved_embedding_model), dtype=np.str_),
        "deam_regression_model_file": np.array(_to_project_relpath(resolved_regression_model), dtype=np.str_),
        "deam_embedding_output": np.array("" if embedding_output is None else str(embedding_output), dtype=np.str_),
        "deam_regression_output": np.array(str(regression_output), dtype=np.str_),
        "config_sample_rate": np.array(sample_rate, dtype=np.int32),
    }

    saved_path = _save_collection_npz(
        output_file=resolved_output_file,
        embedding_type="deam_valence_arousal",
        playlist_csv=csv_path,
        embeddings=embeddings,
        metadata=metadata,
        extra=extra,
    )

    print(f"Saved DEAM playlist collection: {_to_project_relpath(saved_path)}")
    print(f"Tracks: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Embedding backend: {backend_from_payload or embedding_backend}")
    return saved_path


def _maest_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create MAEST playlist embeddings and export one NPZ file.",
    )
    parser.add_argument(
        "tracklist_csv",
        type=Path,
        help="Playlist CSV path (e.g. music/ara-mix/ara_mix_tracks.csv or Apple export CSV).",
    )
    parser.add_argument(
        "--music-dir",
        type=Path,
        default=None,
        help="Directory to resolve mp3_name when CSV has no filepath column.",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        help="MAEST TensorFlow graph (.pb).",
    )
    parser.add_argument(
        "--output-node",
        default=DEFAULT_OUTPUT_NODE,
        help="TensorFlow output node for MAEST embeddings.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output NPZ file path. Defaults to data/maest_embeddings/<playlist>_tracks.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MAEST_OUTPUT_DIR,
        help="Output directory used when --output-file is omitted.",
    )
    parser.add_argument(
        "--skip-missing-audio",
        action="store_true",
        help="Skip rows whose audio files do not exist instead of failing.",
    )
    return parser


def _chroma_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create chroma playlist embeddings and export one NPZ file.",
    )
    parser.add_argument(
        "tracklist_csv",
        type=Path,
        help="Playlist CSV path (e.g. music/ara-mix/ara_mix_tracks.csv or Apple export CSV).",
    )
    parser.add_argument(
        "--music-dir",
        type=Path,
        default=None,
        help="Directory to resolve mp3_name when CSV has no filepath column.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output NPZ file path. Defaults to data/chroma_embeddings/<playlist>_tracks.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CHROMA_OUTPUT_DIR,
        help="Output directory used when --output-file is omitted.",
    )
    parser.add_argument(
        "--skip-missing-audio",
        action="store_true",
        help="Skip rows whose audio files do not exist instead of failing.",
    )
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--frame-size", type=int, default=4096)
    parser.add_argument("--hop-size", type=int, default=1024)
    parser.add_argument("--chroma-bins", type=int, default=12)
    parser.add_argument(
        "--center-baseline",
        type=float,
        default=1.0 / 12.0,
        help="Baseline subtracted from each unit-sum pitch-class bin before pooling (default: 1/12).",
    )
    parser.add_argument(
        "--exclude-key-features",
        action="store_true",
        help="Disable appended key features (default is enabled).",
    )
    return parser


def _tempo_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create TempoCNN playlist embeddings and export one NPZ file.",
    )
    parser.add_argument(
        "tracklist_csv",
        type=Path,
        help="Playlist CSV path (e.g. music/ara-mix/ara_mix_tracks.csv or Apple export CSV).",
    )
    parser.add_argument(
        "--music-dir",
        type=Path,
        default=None,
        help="Directory to resolve mp3_name when CSV has no filepath column.",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="TempoCNN graph (.pb). Defaults to models/deeptemp-k16-3.pb if available.",
    )
    parser.add_argument(
        "--auto-download-model",
        action="store_true",
        help="Download default TempoCNN model if --model-file/default model is missing.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output NPZ file path. Defaults to data/tempo_embeddings/<playlist>_tracks.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TEMPO_OUTPUT_DIR,
        help="Output directory used when --output-file is omitted.",
    )
    parser.add_argument(
        "--skip-missing-audio",
        action="store_true",
        help="Skip rows whose audio files do not exist instead of failing.",
    )
    parser.add_argument("--sample-rate", type=int, default=11025)
    parser.add_argument("--resample-quality", type=int, default=4)
    parser.add_argument(
        "--snippet-length-sec",
        type=float,
        default=None,
        help="Optional max duration per track to analyze (seconds).",
    )
    parser.add_argument("--window-sec", type=float, default=12.0)
    parser.add_argument("--hop-sec", type=float, default=6.0)
    parser.add_argument("--rms-percentile", type=float, default=20.0)
    return parser


def _deam_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create DEAM valence/arousal playlist embeddings and export one NPZ file.",
    )
    parser.add_argument(
        "tracklist_csv",
        type=Path,
        help="Playlist CSV path (e.g. music/ara-mix/ara_mix_tracks.csv or Apple export CSV).",
    )
    parser.add_argument(
        "--music-dir",
        type=Path,
        default=None,
        help="Directory to resolve mp3_name when CSV has no filepath column.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["musicnn", "vggish"],
        default="musicnn",
        help="Embedding frontend used before DEAM regression.",
    )
    parser.add_argument(
        "--embedding-model-file",
        type=Path,
        default=PROJECT_ROOT / "models" / "msd-musicnn-1.pb",
        help="Embedding TensorFlow graph (.pb).",
    )
    parser.add_argument(
        "--regression-model-file",
        type=Path,
        default=PROJECT_ROOT / "models" / "deam-msd-musicnn-2.pb",
        help="DEAM regression TensorFlow graph (.pb).",
    )
    parser.add_argument(
        "--embedding-output",
        default=None,
        help="Optional embedding output node (defaults depend on backend).",
    )
    parser.add_argument(
        "--regression-output",
        default="model/Identity",
        help="DEAM regression model output node.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output NPZ file path. Defaults to data/deam_embeddings/<playlist>_tracks.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DEAM_OUTPUT_DIR,
        help="Output directory used when --output-file is omitted.",
    )
    parser.add_argument(
        "--skip-missing-audio",
        action="store_true",
        help="Skip rows whose audio files do not exist instead of failing.",
    )
    return parser


def maest_main() -> None:
    args = _maest_cli_parser().parse_args()
    create_maest_playlist_embeddings_npz(
        args.tracklist_csv,
        music_dir=args.music_dir,
        model_file=args.model_file,
        output_node=args.output_node,
        output_file=args.output_file,
        output_dir=args.output_dir,
        skip_missing_audio=bool(args.skip_missing_audio),
    )


def chroma_main() -> None:
    args = _chroma_cli_parser().parse_args()
    create_chroma_playlist_embeddings_npz(
        args.tracklist_csv,
        music_dir=args.music_dir,
        output_file=args.output_file,
        output_dir=args.output_dir,
        skip_missing_audio=bool(args.skip_missing_audio),
        sample_rate=int(args.sample_rate),
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        chroma_bins=int(args.chroma_bins),
        center_baseline=float(args.center_baseline),
        include_key_features=not bool(args.exclude_key_features),
    )


def tempo_main() -> None:
    args = _tempo_cli_parser().parse_args()
    create_tempo_playlist_embeddings_npz(
        args.tracklist_csv,
        music_dir=args.music_dir,
        model_file=args.model_file,
        auto_download_model=bool(args.auto_download_model),
        output_file=args.output_file,
        output_dir=args.output_dir,
        skip_missing_audio=bool(args.skip_missing_audio),
        sample_rate=int(args.sample_rate),
        resample_quality=int(args.resample_quality),
        snippet_length_sec=args.snippet_length_sec,
        window_sec=float(args.window_sec),
        hop_sec=float(args.hop_sec),
        rms_percentile=float(args.rms_percentile),
    )


def deam_main() -> None:
    args = _deam_cli_parser().parse_args()
    create_deam_playlist_embeddings_npz(
        args.tracklist_csv,
        music_dir=args.music_dir,
        embedding_backend=str(args.embedding_backend),
        embedding_model_file=args.embedding_model_file,
        regression_model_file=args.regression_model_file,
        embedding_output=args.embedding_output,
        regression_output=str(args.regression_output),
        sample_rate=int(args.sample_rate),
        output_file=args.output_file,
        output_dir=args.output_dir,
        skip_missing_audio=bool(args.skip_missing_audio),
    )
