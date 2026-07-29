"""Playlist embedding pipelines for MAEST, chroma, tempo, and DEAM.

This module moves notebook-style embedding generation into reusable, scriptable
functions that export a single compressed NPZ collection per playlist.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from djprojectexploration.native_warnings import suppress_native_stderr
from djprojectexploration.tracklists import (
    PROJECT_ROOT,
    PlaylistTrack,
    load_playlist_tracks,
    optional_float as _optional_float,
    resolve_audio_path as _resolve_audio_path,
    to_project_relpath as _to_project_relpath,
)

DEFAULT_MAEST_OUTPUT_DIR = PROJECT_ROOT / "data" / "maest_embeddings"
DEFAULT_CHROMA_OUTPUT_DIR = PROJECT_ROOT / "data" / "chroma_embeddings"
DEFAULT_TEMPO_OUTPUT_DIR = PROJECT_ROOT / "data" / "tempo_embeddings"
DEFAULT_DEAM_OUTPUT_DIR = PROJECT_ROOT / "data" / "deam_embeddings"
DEFAULT_GROOVE_OUTPUT_DIR = PROJECT_ROOT / "data" / "groove_embeddings"
DEFAULT_ANNOTATION_OUTPUT_DIR = PROJECT_ROOT / "data" / "annotations"
DEFAULT_MODEL_FILENAME = "discogs-maest-30s-pw-519l-2.pb"
DEFAULT_OUTPUT_NODE = "PartitionedCall/Identity_7"
DEFAULT_MODEL_FILE = PROJECT_ROOT / "models" / DEFAULT_MODEL_FILENAME
DEFAULT_GENRE_MODEL_FILE = PROJECT_ROOT / "models" / "genre_discogs519-discogs-maest-30s-pw-519l-1.pb"
GENRE_EMBEDDING_OUTPUT_NODE = "PartitionedCall/Identity_12"
KEY_TONIC_INDEX = {name: index for index, name in enumerate(("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"))}


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
        "track_uid": _string_array([_track_uid(t) for t in tracks]),
    }


def _track_uid(track: PlaylistTrack) -> str:
    """Stable join key for local collections; paths are resolved by the loader."""
    identity = f"{track.audio_path.as_posix()}\0{track.mp3_name}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _annotation_output_file(playlist_csv: Path, annotation_type: str, model_slug: str) -> Path:
    return DEFAULT_ANNOTATION_OUTPUT_DIR / f"{_default_npz_name(playlist_csv)[:-4]}__{annotation_type}_{model_slug}.npz"


def _annotation_common(
    *, tracks: list[PlaylistTrack], playlist_csv: Path, annotation_type: str,
    input_artifact_type: str, model_id: str, metadata: dict[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "annotation_schema_version": np.array(1, dtype=np.int32),
        "annotation_type": np.array(annotation_type, dtype=np.str_),
        "created_utc": np.array(datetime.now(tz=timezone.utc).isoformat(), dtype=np.str_),
        "playlist_csv": np.array(_to_project_relpath(playlist_csv), dtype=np.str_),
        "input_artifact_type": np.array(input_artifact_type, dtype=np.str_),
        "model_id": np.array(model_id, dtype=np.str_),
        "num_tracks": np.array(len(tracks), dtype=np.int32),
        "track_uid": metadata["track_uid"],
        "track_numbers": metadata["track_numbers"],
        "filenames": metadata["filenames"],
    }


def _genre_label_names(metadata_file: Path, expected_count: int) -> list[str]:
    """Read Essentia model metadata without coupling to one metadata revision."""
    if not metadata_file.exists():
        return [f"class_{i}" for i in range(expected_count)]
    raw = json.loads(metadata_file.read_text(encoding="utf-8"))
    for key in ("classes", "labels", "class_names"):
        values = raw.get(key)
        if isinstance(values, list) and len(values) == expected_count:
            return [str(value) for value in values]
    return [f"class_{i}" for i in range(expected_count)]


def _tonic_index(key_label: str) -> int:
    normalized = str(key_label).strip().upper().replace("♯", "#").replace("♭", "B")
    enharmonic = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#", "CB": "B", "FB": "E"}
    return KEY_TONIC_INDEX.get(enharmonic.get(normalized, normalized), -1)


def _default_npz_name(tracklist_csv: Path, *, variant: str | None = None) -> str:
    stem = tracklist_csv.stem
    if stem.endswith("_tracks"):
        base = stem
    else:
        base = f"{stem}_tracks"
    if variant:
        return f"{base}_{variant}.npz"
    return f"{base}.npz"


def _tempo_bpm_lookup_from_npz(path: str | Path | None) -> dict[int, float]:
    if path is None:
        return {}
    npz_path = Path(path).expanduser().resolve()
    if not npz_path.exists():
        return {}
    with np.load(npz_path, allow_pickle=False) as data:
        if "track_numbers" not in data.files or "tempo_bpm" not in data.files:
            return {}
        track_numbers = np.asarray(data["track_numbers"], dtype=np.int64).reshape(-1)
        tempo_bpm = np.asarray(data["tempo_bpm"], dtype=np.float64).reshape(-1)
        out: dict[int, float] = {}
        for index, track_number in enumerate(track_numbers):
            bpm = float(tempo_bpm[index]) if index < tempo_bpm.size else float("nan")
            if np.isfinite(bpm) and bpm > 0:
                out[int(track_number)] = bpm
        return out


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
    suppress_essentia_warnings: bool = True,
    section: str = "full",
    peak_window_sec: float = 30.0,
    peak_hop_sec: float = 1.0,
    enrich_genre: bool = False,
    genre_model_file: str | Path = DEFAULT_GENRE_MODEL_FILE,
    genre_annotation_file: str | Path | None = None,
    genre_top_k: int = 5,
) -> Path:
    """Build MAEST embeddings for playlist tracks and save one NPZ collection."""
    from essentia.standard import MonoLoader, TensorflowPredict, TensorflowPredictMAEST

    from djprojectexploration.maest_embedding_extractor import (
        extract_embedding_from_audio, extract_peak_rms_embedding_from_audio,
        predict_discogs519_genres_from_audio,
    )

    section = str(section).lower().strip()
    if section not in {"full", "peak30"}:
        raise ValueError(f"Unsupported MAEST section: {section!r}. Expected 'full' or 'peak30'.")

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
    resolved_genre_model_file = Path(genre_model_file).expanduser().resolve()
    if enrich_genre and not resolved_genre_model_file.exists():
        raise FileNotFoundError(
            f"Genre head model file not found: {resolved_genre_model_file}. "
            "Download genre_discogs519-discogs-maest-30s-pw-519l-1.pb from Essentia "
            "or pass enrich_genre=False."
        )
    if enrich_genre and section != "full":
        raise ValueError("Genre enrichment is only supported for MAEST section='full'.")

    vectors: list[np.ndarray] = []
    reductions: list[str] = []
    raw_shapes: list[str] = []
    peak_start_sec: list[float] = []
    peak_end_sec: list[float] = []
    peak_window_rms_db: list[float] = []
    genre_scores: list[np.ndarray] = []
    with suppress_native_stderr(suppress_essentia_warnings):
        model = TensorflowPredictMAEST(graphFilename=str(resolved_model_file), output=output_node)
        genre_embedding_model = (
            TensorflowPredictMAEST(graphFilename=str(resolved_model_file), output=GENRE_EMBEDDING_OUTPUT_NODE)
            if enrich_genre else None
        )
        genre_model = (
            TensorflowPredict(
                graphFilename=str(resolved_genre_model_file), inputs=["embeddings"],
                outputs=["PartitionedCall/Identity_1"],
            ) if enrich_genre else None
        )
        total = len(tracks)
        for index, track in enumerate(tracks, start=1):
            print(f"[{index}/{total}] Extracting MAEST {section} embedding: {track.title}", flush=True)
            audio = MonoLoader(filename=str(track.audio_path), sampleRate=16000, resampleQuality=4)()
            if section == "peak30":
                vector, raw_shape, reduction, start_sec, end_sec, rms_db = extract_peak_rms_embedding_from_audio(
                    audio,
                    resolved_model_file,
                    output_node,
                    model=model,
                    window_sec=float(peak_window_sec),
                    hop_sec=float(peak_hop_sec),
                )
                peak_start_sec.append(float(start_sec))
                peak_end_sec.append(float(end_sec))
                peak_window_rms_db.append(float(rms_db))
            else:
                vector, raw_shape, reduction = extract_embedding_from_audio(
                    audio,
                    resolved_model_file,
                    output_node,
                    model=model,
                )
                peak_start_sec.append(np.nan)
                peak_end_sec.append(np.nan)
                peak_window_rms_db.append(np.nan)
            if enrich_genre:
                assert genre_embedding_model is not None and genre_model is not None
                genre_scores.append(predict_discogs519_genres_from_audio(
                    audio, embedding_model=genre_embedding_model, genre_model=genre_model,
                ))
            vectors.append(np.asarray(vector, dtype=np.float32).reshape(-1))
            reductions.append(reduction)
            raw_shapes.append("x".join(str(dim) for dim in raw_shape))
        del model
        del genre_embedding_model
        del genre_model

    embeddings = np.vstack(vectors).astype(np.float32)

    if output_file is None:
        resolved_output_dir = Path(output_dir).expanduser().resolve()
        resolved_output_file = resolved_output_dir / _default_npz_name(
            csv_path,
            variant=None if section == "full" else section,
        )
    else:
        resolved_output_file = Path(output_file).expanduser().resolve()

    metadata = _metadata_arrays(tracks)
    extra = {
        "maest_section": np.array(section, dtype=np.str_),
        "maest_model_file": np.array(_to_project_relpath(resolved_model_file), dtype=np.str_),
        "maest_output_node": np.array(output_node, dtype=np.str_),
        "maest_reductions": _string_array(reductions),
        "maest_raw_prediction_shapes": _string_array(raw_shapes),
        "maest_peak_start_sec": np.asarray(peak_start_sec, dtype=np.float32),
        "maest_peak_end_sec": np.asarray(peak_end_sec, dtype=np.float32),
        "maest_peak_window_rms_db": np.asarray(peak_window_rms_db, dtype=np.float32),
        "config_peak_window_sec": np.array(float(peak_window_sec), dtype=np.float32),
        "config_peak_hop_sec": np.array(float(peak_hop_sec), dtype=np.float32),
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
    print(f"Section: {section}")
    print(f"Model: {_to_project_relpath(resolved_model_file)}")
    if enrich_genre:
        scores = np.vstack(genre_scores).astype(np.float32)
        label_names = _genre_label_names(resolved_genre_model_file.with_suffix(".json"), scores.shape[1])
        top_k = min(max(1, int(genre_top_k)), scores.shape[1])
        top_indices = np.argsort(scores, axis=1)[:, -top_k:][:, ::-1].astype(np.int16)
        annotation_path = Path(genre_annotation_file).expanduser().resolve() if genre_annotation_file else _annotation_output_file(
            csv_path, "genre", "discogs519_maest30pw519l_v1",
        )
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _annotation_common(
            tracks=tracks, playlist_csv=csv_path, annotation_type="genre", input_artifact_type="maest",
            model_id="genre_discogs519-discogs-maest-30s-pw-519l-1", metadata=metadata,
        )
        payload.update({
            "model_file": np.array(_to_project_relpath(resolved_genre_model_file), dtype=np.str_),
            "embedding_output_node": np.array(GENRE_EMBEDDING_OUTPUT_NODE, dtype=np.str_),
            "label_names": _string_array(label_names), "scores": scores,
            "top_label_indices": top_indices,
            "top_label_scores": np.take_along_axis(scores, top_indices, axis=1).astype(np.float32),
        })
        np.savez_compressed(annotation_path, **payload)
        print(f"Saved genre annotations: {_to_project_relpath(annotation_path)}")
    return saved_path


def create_maest_genre_annotations_npz(
    tracklist_csv: str | Path,
    *,
    music_dir: str | Path | None = None,
    model_file: str | Path = DEFAULT_MODEL_FILE,
    genre_model_file: str | Path = DEFAULT_GENRE_MODEL_FILE,
    output_file: str | Path | None = None,
    skip_missing_audio: bool = False,
    suppress_essentia_warnings: bool = True,
    genre_top_k: int = 5,
) -> Path:
    """Create only the Discogs-519 annotation sidecar from audio.

    This is used when a generic MAEST collection already exists, avoiding an
    unnecessary second ``Identity_7`` extraction merely to fill annotations.
    """
    from essentia.standard import MonoLoader, TensorflowPredict, TensorflowPredictMAEST
    from djprojectexploration.maest_embedding_extractor import predict_discogs519_genres_from_audio

    csv_path = Path(tracklist_csv).expanduser().resolve()
    tracks = load_playlist_tracks(csv_path, music_dir=music_dir, skip_missing_audio=skip_missing_audio)
    resolved_model_file = Path(model_file).expanduser().resolve()
    resolved_genre_model_file = Path(genre_model_file).expanduser().resolve()
    for path, description in ((resolved_model_file, "MAEST model"), (resolved_genre_model_file, "Discogs-519 genre head")):
        if not path.exists():
            raise FileNotFoundError(f"{description} file not found: {path}")

    scores_rows: list[np.ndarray] = []
    with suppress_native_stderr(suppress_essentia_warnings):
        embedding_model = TensorflowPredictMAEST(
            graphFilename=str(resolved_model_file), output=GENRE_EMBEDDING_OUTPUT_NODE,
        )
        genre_model = TensorflowPredict(
            graphFilename=str(resolved_genre_model_file), inputs=["embeddings"],
            outputs=["PartitionedCall/Identity_1"],
        )
        for index, track in enumerate(tracks, start=1):
            print(f"[{index}/{len(tracks)}] Extracting MAEST genre annotations: {track.title}", flush=True)
            audio = MonoLoader(filename=str(track.audio_path), sampleRate=16000, resampleQuality=4)()
            scores_rows.append(predict_discogs519_genres_from_audio(
                audio, embedding_model=embedding_model, genre_model=genre_model,
            ))

    scores = np.vstack(scores_rows).astype(np.float32)
    label_names = _genre_label_names(resolved_genre_model_file.with_suffix(".json"), scores.shape[1])
    top_k = min(max(1, int(genre_top_k)), scores.shape[1])
    top_indices = np.argsort(scores, axis=1)[:, -top_k:][:, ::-1].astype(np.int16)
    metadata = _metadata_arrays(tracks)
    annotation_path = Path(output_file).expanduser().resolve() if output_file else _annotation_output_file(
        csv_path, "genre", "discogs519_maest30pw519l_v1",
    )
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _annotation_common(
        tracks=tracks, playlist_csv=csv_path, annotation_type="genre", input_artifact_type="maest",
        model_id="genre_discogs519-discogs-maest-30s-pw-519l-1", metadata=metadata,
    )
    payload.update({
        "model_file": np.array(_to_project_relpath(resolved_genre_model_file), dtype=np.str_),
        "embedding_output_node": np.array(GENRE_EMBEDDING_OUTPUT_NODE, dtype=np.str_),
        "label_names": _string_array(label_names), "scores": scores,
        "top_label_indices": top_indices,
        "top_label_scores": np.take_along_axis(scores, top_indices, axis=1).astype(np.float32),
    })
    np.savez_compressed(annotation_path, **payload)
    print(f"Saved genre annotations: {_to_project_relpath(annotation_path)}")
    return annotation_path


def create_maest_segment_series_npz(
    tracklist_csv: str | Path,
    *,
    music_dir: str | Path | None = None,
    model_file: str | Path = DEFAULT_MODEL_FILE,
    output_node: str = DEFAULT_OUTPUT_NODE,
    output_file: str | Path | None = None,
    output_dir: str | Path = DEFAULT_MAEST_OUTPUT_DIR,
    window_sec: float = 30.0,
    hop_sec: float = 15.0,
    include_partial: bool = False,
    skip_missing_audio: bool = False,
    suppress_essentia_warnings: bool = True,
) -> Path:
    """Export time-stamped MAEST CLS segment embeddings for every playlist track.

    Segment rows are stored contiguously in ``segment_embeddings``. Use
    ``segment_track_index`` to join a row back to the track-level metadata.
    """
    from essentia.standard import TensorflowPredictMAEST

    from djprojectexploration.maest_embedding_extractor import (
        MAEST_MEL_HOP_SAMPLES,
        MAEST_SAMPLE_RATE,
        extract_segment_embeddings,
    )

    csv_path = Path(tracklist_csv).expanduser().resolve()
    tracks = load_playlist_tracks(csv_path, music_dir=music_dir, skip_missing_audio=skip_missing_audio)
    resolved_model_file = Path(model_file).expanduser().resolve()
    if not resolved_model_file.exists():
        raise FileNotFoundError(f"MAEST model file not found: {resolved_model_file}")

    all_embeddings: list[np.ndarray] = []
    track_indices: list[np.ndarray] = []
    start_sec: list[np.ndarray] = []
    end_sec: list[np.ndarray] = []
    center_sec: list[np.ndarray] = []
    raw_shapes: list[np.ndarray] = []
    segment_counts: list[int] = []

    patch_hop_frames = max(1, int(round(float(hop_sec) * MAEST_SAMPLE_RATE / MAEST_MEL_HOP_SAMPLES)))
    effective_hop_sec = patch_hop_frames * MAEST_MEL_HOP_SAMPLES / MAEST_SAMPLE_RATE
    with suppress_native_stderr(suppress_essentia_warnings):
        model = TensorflowPredictMAEST(
            graphFilename=str(resolved_model_file),
            output=output_node,
            patchHopSize=patch_hop_frames,
            lastPatchMode="repeat" if include_partial else "discard",
        )
        for index, track in enumerate(tracks):
            print(f"[{index + 1}/{len(tracks)}] Extracting MAEST segment series: {track.title}", flush=True)
            series = extract_segment_embeddings(
                track.audio_path, resolved_model_file, output_node, model=model,
                window_sec=window_sec, hop_sec=hop_sec, include_partial=include_partial,
            )
            vectors = np.asarray(series["embeddings"], dtype=np.float32)
            count = int(vectors.shape[0])
            segment_counts.append(count)
            if count:
                all_embeddings.append(vectors)
                track_indices.append(np.full(count, index, dtype=np.int32))
                start_sec.append(np.asarray(series["start_sec"], dtype=np.float32))
                end_sec.append(np.asarray(series["end_sec"], dtype=np.float32))
                center_sec.append(np.asarray(series["center_sec"], dtype=np.float32))
                raw_shapes.append(np.asarray(series["raw_prediction_shape"], dtype=np.int32))

    if not all_embeddings:
        raise ValueError("No MAEST segment embeddings were produced for the playlist.")

    if output_file is None:
        output_file = Path(output_dir).expanduser().resolve() / _default_npz_name(
            csv_path, variant=f"maest_segments_w{window_sec:g}_h{hop_sec:g}"
        )
    saved_path = Path(output_file).expanduser().resolve()
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        saved_path,
        embedding_type=np.array("maest_segment_series", dtype=np.str_),
        created_utc=np.array(datetime.now(tz=timezone.utc).isoformat(), dtype=np.str_),
        playlist_csv=np.array(_to_project_relpath(csv_path), dtype=np.str_),
        maest_model_file=np.array(_to_project_relpath(resolved_model_file), dtype=np.str_),
        maest_output_node=np.array(output_node, dtype=np.str_),
        sample_rate=np.array(16000, dtype=np.int32),
        window_sec=np.array(window_sec, dtype=np.float32),
        hop_sec=np.array(hop_sec, dtype=np.float32),
        effective_hop_sec=np.array(effective_hop_sec, dtype=np.float32),
        last_window_mode=np.array("repeat" if include_partial else "discard", dtype=np.str_),
        num_tracks=np.array(len(tracks), dtype=np.int32),
        num_segments=np.array(sum(segment_counts), dtype=np.int32),
        embedding_dimension=np.array(all_embeddings[0].shape[1], dtype=np.int32),
        segment_embeddings=np.vstack(all_embeddings).astype(np.float32),
        segment_track_index=np.concatenate(track_indices),
        segment_start_sec=np.concatenate(start_sec),
        segment_end_sec=np.concatenate(end_sec),
        segment_center_sec=np.concatenate(center_sec),
        segment_raw_prediction_shape=np.vstack(raw_shapes),
        track_segment_counts=np.asarray(segment_counts, dtype=np.int32),
        **_metadata_arrays(tracks),
    )
    print(f"Saved MAEST segment series: {_to_project_relpath(saved_path)}")
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
    suppress_essentia_warnings: bool = True,
    enrich_key: bool = True,
    key_annotation_file: str | Path | None = None,
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
        with suppress_native_stderr(suppress_essentia_warnings):
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
    if suppress_essentia_warnings:
        print("Suppressed repeated native Essentia warnings during chroma extraction.")
    if enrich_key:
        annotation_path = Path(key_annotation_file).expanduser().resolve() if key_annotation_file else _annotation_output_file(
            csv_path, "key", "chroma_hpcp_v1",
        )
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        tonic_indices = np.asarray([_tonic_index(key) for key in detected_keys], dtype=np.int8)
        annotation_payload = _annotation_common(
            tracks=tracks, playlist_csv=csv_path, annotation_type="key", input_artifact_type="chroma",
            model_id="essentia_hpcp_key_estimator_v1", metadata=metadata,
        )
        annotation_payload.update({
            "key_label": _string_array(detected_keys), "tonic_index": tonic_indices,
            "mode": _string_array(detected_scales),
            "confidence": np.asarray(detected_strengths, dtype=np.float32),
            "detector_config": np.array(json.dumps({
                "sample_rate": sample_rate, "frame_size": frame_size, "hop_size": hop_size,
                "chroma_bins": chroma_bins,
            }, sort_keys=True), dtype=np.str_),
        })
        np.savez_compressed(annotation_path, **annotation_payload)
        print(f"Saved key annotations: {_to_project_relpath(annotation_path)}")
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
    suppress_essentia_warnings: bool = True,
) -> Path:
    """Build TempoCNN embeddings for playlist tracks and save one NPZ collection."""
    from essentia.standard import TempoCNN

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
    with suppress_native_stderr(suppress_essentia_warnings):
        model = TempoCNN(graphFilename=str(resolved_model_file))
        for track in tracks:
            payload = generate_tempo_embedding(
                audio_file=track.audio_path,
                model_file=resolved_model_file,
                auto_download_model=False,
                model=model,
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


def create_groove_playlist_embeddings_npz(
    tracklist_csv: str | Path,
    *,
    music_dir: str | Path | None = None,
    model_file: str | Path | None = None,
    auto_download_model: bool = False,
    output_file: str | Path | None = None,
    output_dir: str | Path = DEFAULT_GROOVE_OUTPUT_DIR,
    skip_missing_audio: bool = False,
    tempo_embeddings: str | Path | None = None,
    sample_rate: int = 44100,
    tempocnn_sample_rate: int = 11025,
    snippet_length_sec: float | None = None,
    hop_length: int = 256,
    n_fft: int = 2048,
    use_csv_bpm: bool = True,
    auto_phase_align: bool = True,
    phase_align_mode: str = "adaptive",
    phase_align_max_shift_sec: float = 0.18,
    phase_align_step_sec: float = 0.002,
    auto_prepend_start_beats: bool = True,
    subdivisions_per_beat: int = 4,
    beats_per_bar: int = 4,
    phrase_bars: int = 1,
    profile_mode: str = "phrase",
    pooling_mode: str = "mean",
    pooling_topk: int = 3,
    normalize_per_beat: bool = True,
    suppress_essentia_warnings: bool = True,
) -> Path:
    """Build groove embeddings for playlist tracks and save one NPZ collection."""
    from djprojectexploration.groove_embedding import generate_groove_embedding
    from djprojectexploration.tempo_embedding import DEFAULT_TEMPOCNN_MODEL_URL

    csv_path = Path(tracklist_csv).expanduser().resolve()
    tracks = load_playlist_tracks(
        csv_path,
        music_dir=music_dir,
        skip_missing_audio=skip_missing_audio,
    )
    tempo_bpm_lookup = _tempo_bpm_lookup_from_npz(tempo_embeddings)

    vectors: list[np.ndarray] = []
    embedding_subtypes: list[str] = []
    beat_profiles: list[np.ndarray] = []
    phrase_profiles: list[np.ndarray] = []
    band_names: list[str] | None = None

    beat_bpms: list[float] = []
    beat_bpm_seeds: list[float] = []
    tempocnn_bpms: list[float] = []
    beat_counts: list[int] = []
    beat_sources: list[str] = []
    phase_anchors: list[float] = []
    phase_shifts: list[float] = []
    phase_modes: list[str] = []
    phase_search_modes: list[str] = []
    phase_search_max_shifts: list[float] = []
    prepended_counts: list[int] = []
    complete_phrases: list[int] = []
    model_files: list[str] = []
    manual_bpm_used: list[float] = []

    local_start_index: list[int] = []
    local_counts: list[int] = []
    local_bpm_chunks: list[np.ndarray] = []
    local_prob_chunks: list[np.ndarray] = []
    cursor = 0

    with suppress_native_stderr(suppress_essentia_warnings):
        for track in tracks:
            manual_bpm = None
            if use_csv_bpm and track.bpm is not None and np.isfinite(float(track.bpm)) and float(track.bpm) > 0:
                manual_bpm = float(track.bpm)
            elif tempo_bpm_lookup:
                manual_bpm = tempo_bpm_lookup.get(int(track.track_number))
            onset_time_sec = track.onset_time
            payload = generate_groove_embedding(
                audio_file=track.audio_path,
                sample_rate=int(sample_rate),
                tempocnn_sample_rate=int(tempocnn_sample_rate),
                model_file=model_file,
                auto_download_model=bool(auto_download_model),
                snippet_length_sec=snippet_length_sec,
                hop_length=int(hop_length),
                n_fft=int(n_fft),
                manual_bpm=manual_bpm,
                onset_time_sec=onset_time_sec,
                auto_phase_align=bool(auto_phase_align),
                phase_align_mode=phase_align_mode,
                phase_align_max_shift_sec=float(phase_align_max_shift_sec),
                phase_align_step_sec=float(phase_align_step_sec),
                auto_prepend_start_beats=bool(auto_prepend_start_beats),
                subdivisions_per_beat=int(subdivisions_per_beat),
                beats_per_bar=int(beats_per_bar),
                phrase_bars=int(phrase_bars),
                pooling_mode=pooling_mode,
                pooling_topk=int(pooling_topk),
                profile_mode=profile_mode,
                normalize_per_beat=bool(normalize_per_beat),
            )

            vectors.append(np.asarray(payload["embedding"], dtype=np.float32).reshape(-1))
            embedding_subtypes.append(str(payload.get("embedding_subtype", "unknown")))
            beat_profiles.append(np.asarray(payload.get("beat_profile", []), dtype=np.float32))
            phrase_profiles.append(np.asarray(payload.get("phrase_profile", []), dtype=np.float32))
            if band_names is None:
                band_names = [str(name) for name in payload.get("band_names", [])]

            beat_pooling = payload.get("beat_pooling", {})
            beat_bpms.append(float(beat_pooling.get("bpm", np.nan)))
            beat_bpm_seeds.append(float(beat_pooling.get("bpm_seed", np.nan)))
            tempocnn_value = beat_pooling.get("tempocnn_bpm")
            tempocnn_bpms.append(np.nan if tempocnn_value is None else float(tempocnn_value))
            beat_counts.append(int(beat_pooling.get("beat_count", 0)))
            beat_sources.append(str(beat_pooling.get("beat_source", "")))
            phase_anchors.append(float(beat_pooling.get("phase_anchor_seconds", np.nan)))
            phase_shifts.append(float(beat_pooling.get("phase_shift_seconds", 0.0)))
            phase_modes.append(str(beat_pooling.get("phase_align_mode", "")))
            phase_search_modes.append(str(beat_pooling.get("phase_align_search_mode", "")))
            phase_search_max_shifts.append(float(beat_pooling.get("phase_align_search_max_shift_seconds", np.nan)))
            prepended_counts.append(int(beat_pooling.get("prepended_start_beats", 0)))
            complete_phrases.append(int(beat_pooling.get("complete_phrases", 0)))
            manual_bpm_used.append(np.nan if manual_bpm is None else float(manual_bpm))

            config = payload.get("config", {})
            model_files.append(str(config.get("tempocnn_model_file") or ""))

            local_bpm = np.asarray(payload.get("local_bpm", []), dtype=np.float32).reshape(-1)
            local_prob = np.asarray(payload.get("local_probability", []), dtype=np.float32).reshape(-1)
            n_local = int(min(local_bpm.size, local_prob.size))
            local_bpm = local_bpm[:n_local]
            local_prob = local_prob[:n_local]
            local_start_index.append(cursor)
            local_counts.append(n_local)
            cursor += n_local
            local_bpm_chunks.append(local_bpm)
            local_prob_chunks.append(local_prob)

    embeddings = np.vstack(vectors).astype(np.float32)
    beat_profile_tensor = np.stack(beat_profiles).astype(np.float32)
    phrase_profile_tensor = np.stack(phrase_profiles).astype(np.float32)

    if output_file is None:
        resolved_output_dir = Path(output_dir).expanduser().resolve()
        resolved_output_file = resolved_output_dir / _default_npz_name(csv_path)
    else:
        resolved_output_file = Path(output_file).expanduser().resolve()

    metadata = _metadata_arrays(tracks)

    if cursor > 0:
        local_bpm_flat = np.concatenate(local_bpm_chunks).astype(np.float32)
        local_prob_flat = np.concatenate(local_prob_chunks).astype(np.float32)
    else:
        local_bpm_flat = np.array([], dtype=np.float32)
        local_prob_flat = np.array([], dtype=np.float32)

    extra = {
        "groove_embedding_subtype": _string_array(embedding_subtypes),
        "groove_band_names": _string_array(band_names or []),
        "groove_beat_profile": beat_profile_tensor,
        "groove_phrase_profile": phrase_profile_tensor,
        "groove_bpm": np.asarray(beat_bpms, dtype=np.float32),
        "groove_bpm_seed": np.asarray(beat_bpm_seeds, dtype=np.float32),
        "groove_tempocnn_bpm": np.asarray(tempocnn_bpms, dtype=np.float32),
        "groove_manual_bpm_used": np.asarray(manual_bpm_used, dtype=np.float32),
        "groove_beat_count": np.asarray(beat_counts, dtype=np.int32),
        "groove_beat_source": _string_array(beat_sources),
        "groove_phase_anchor_seconds": np.asarray(phase_anchors, dtype=np.float32),
        "groove_phase_shift_seconds": np.asarray(phase_shifts, dtype=np.float32),
        "groove_phase_align_mode": _string_array(phase_modes),
        "groove_phase_align_search_mode": _string_array(phase_search_modes),
        "groove_phase_align_search_max_shift_seconds": np.asarray(phase_search_max_shifts, dtype=np.float32),
        "groove_prepended_start_beats": np.asarray(prepended_counts, dtype=np.int32),
        "groove_complete_phrases": np.asarray(complete_phrases, dtype=np.int32),
        "groove_tempocnn_model_file": _string_array(model_files),
        "groove_local_start_index": np.asarray(local_start_index, dtype=np.int64),
        "groove_local_count": np.asarray(local_counts, dtype=np.int32),
        "groove_local_bpm_flat": local_bpm_flat,
        "groove_local_probability_flat": local_prob_flat,
        "groove_model_url": np.array(DEFAULT_TEMPOCNN_MODEL_URL, dtype=np.str_),
        "config_sample_rate": np.array(sample_rate, dtype=np.int32),
        "config_tempocnn_sample_rate": np.array(tempocnn_sample_rate, dtype=np.int32),
        "config_snippet_length_sec": np.array(
            np.nan if snippet_length_sec is None else float(snippet_length_sec),
            dtype=np.float32,
        ),
        "config_hop_length": np.array(hop_length, dtype=np.int32),
        "config_n_fft": np.array(n_fft, dtype=np.int32),
        "config_use_csv_bpm": np.array(bool(use_csv_bpm), dtype=np.bool_),
        "config_auto_phase_align": np.array(bool(auto_phase_align), dtype=np.bool_),
        "config_phase_align_mode": np.array(str(phase_align_mode), dtype=np.str_),
        "config_phase_align_max_shift_sec": np.array(phase_align_max_shift_sec, dtype=np.float32),
        "config_phase_align_step_sec": np.array(phase_align_step_sec, dtype=np.float32),
        "config_auto_prepend_start_beats": np.array(bool(auto_prepend_start_beats), dtype=np.bool_),
        "config_subdivisions_per_beat": np.array(subdivisions_per_beat, dtype=np.int32),
        "config_beats_per_bar": np.array(beats_per_bar, dtype=np.int32),
        "config_phrase_bars": np.array(phrase_bars, dtype=np.int32),
        "config_profile_mode": np.array(str(profile_mode), dtype=np.str_),
        "config_pooling_mode": np.array(str(pooling_mode), dtype=np.str_),
        "config_pooling_topk": np.array(pooling_topk, dtype=np.int32),
        "config_normalize_per_beat": np.array(bool(normalize_per_beat), dtype=np.bool_),
    }

    saved_path = _save_collection_npz(
        output_file=resolved_output_file,
        embedding_type="groove",
        playlist_csv=csv_path,
        embeddings=embeddings,
        metadata=metadata,
        extra=extra,
    )

    print(f"Saved groove playlist collection: {_to_project_relpath(saved_path)}")
    print(f"Tracks: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Profile mode: {profile_mode}")
    print(f"Use CSV BPM: {use_csv_bpm}")
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
        "--section",
        choices=["full", "peak30"],
        default="full",
        help="Audio section to embed. 'full' preserves the existing full-track segment average; 'peak30' embeds the highest-RMS sliding window.",
    )
    parser.add_argument(
        "--peak-window-sec",
        type=float,
        default=30.0,
        help="Peak-RMS window length in seconds when --section peak30 is used.",
    )
    parser.add_argument(
        "--peak-hop-sec",
        type=float,
        default=1.0,
        help="Sliding RMS hop length in seconds when --section peak30 is used.",
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
    parser.add_argument(
        "--show-essentia-warnings",
        action="store_true",
        help="Show native Essentia/TensorFlow stderr warnings. Hidden by default.",
    )
    parser.add_argument(
        "--enrich-genre",
        action="store_true",
        help="Write a Discogs-519 genre annotation sidecar (requires the genre-head model).",
    )
    parser.add_argument("--genre-model-file", type=Path, default=DEFAULT_GENRE_MODEL_FILE)
    parser.add_argument("--genre-annotation-file", type=Path, default=None)
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
    parser.add_argument(
        "--show-essentia-warnings",
        action="store_true",
        help="Show repeated native Essentia warnings during chroma extraction.",
    )
    parser.add_argument(
        "--skip-key-annotations",
        action="store_true",
        help="Do not write the key-detection annotation sidecar; legacy chroma fields are unchanged.",
    )
    parser.add_argument("--key-annotation-file", type=Path, default=None)
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
    parser.add_argument(
        "--show-essentia-warnings",
        action="store_true",
        help="Show native Essentia/TensorFlow stderr warnings. Hidden by default.",
    )
    return parser


def _groove_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create groove playlist embeddings and export one NPZ file.",
    )
    parser.add_argument(
        "tracklist_csv",
        type=Path,
        help="Playlist CSV path (e.g. music/aries-mix/aries_mix_tracks.csv or Apple export CSV).",
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
        help="TempoCNN graph (.pb). Defaults to models/deeptemp-k16-3.pb if needed.",
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
        help="Output NPZ file path. Defaults to data/groove_embeddings/<playlist>_tracks.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GROOVE_OUTPUT_DIR,
        help="Output directory used when --output-file is omitted.",
    )
    parser.add_argument(
        "--skip-missing-audio",
        action="store_true",
        help="Skip rows whose audio files do not exist instead of failing.",
    )
    parser.add_argument(
        "--tempo-embeddings",
        type=Path,
        default=None,
        help="Optional TempoCNN playlist NPZ to reuse for BPM seeds before falling back to per-track TempoCNN.",
    )
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--tempocnn-sample-rate", type=int, default=11025)
    parser.add_argument(
        "--snippet-length-sec",
        type=float,
        default=None,
        help="Optional max duration per track to analyze (seconds).",
    )
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument(
        "--ignore-csv-bpm",
        action="store_true",
        help="Ignore CSV bpm/onset-time and use TempoCNN for every track.",
    )
    parser.add_argument(
        "--disable-phase-align",
        action="store_true",
        help="Disable onset-envelope phase alignment.",
    )
    parser.add_argument(
        "--phase-align-mode",
        choices=["full", "low_mid", "adaptive"],
        default="adaptive",
    )
    parser.add_argument("--phase-align-max-shift-sec", type=float, default=0.18)
    parser.add_argument("--phase-align-step-sec", type=float, default=0.002)
    parser.add_argument(
        "--disable-prepend-start-beats",
        action="store_true",
        help="Disable near-zero missing-start-beat repair.",
    )
    parser.add_argument("--subdivisions-per-beat", type=int, default=4)
    parser.add_argument("--beats-per-bar", type=int, default=4)
    parser.add_argument("--phrase-bars", type=int, default=1)
    parser.add_argument(
        "--profile-mode",
        choices=["beat", "phrase"],
        default="phrase",
        help="Which profile to flatten as the exported embedding.",
    )
    parser.add_argument(
        "--pooling-mode",
        choices=["center", "mean", "max", "topk_mean"],
        default="mean",
    )
    parser.add_argument("--pooling-topk", type=int, default=3)
    parser.add_argument(
        "--disable-normalize-per-beat",
        action="store_true",
        help="Disable per-beat activity normalization before averaging.",
    )
    parser.add_argument(
        "--show-essentia-warnings",
        action="store_true",
        help="Show native Essentia/TensorFlow stderr warnings. Hidden by default.",
    )
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
        suppress_essentia_warnings=not bool(args.show_essentia_warnings),
        section=str(args.section),
        peak_window_sec=float(args.peak_window_sec),
        peak_hop_sec=float(args.peak_hop_sec),
        enrich_genre=bool(args.enrich_genre),
        genre_model_file=args.genre_model_file,
        genre_annotation_file=args.genre_annotation_file,
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
        suppress_essentia_warnings=not bool(args.show_essentia_warnings),
        enrich_key=not bool(args.skip_key_annotations),
        key_annotation_file=args.key_annotation_file,
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
        suppress_essentia_warnings=not bool(args.show_essentia_warnings),
    )


def groove_main() -> None:
    args = _groove_cli_parser().parse_args()
    create_groove_playlist_embeddings_npz(
        args.tracklist_csv,
        music_dir=args.music_dir,
        model_file=args.model_file,
        auto_download_model=bool(args.auto_download_model),
        output_file=args.output_file,
        output_dir=args.output_dir,
        skip_missing_audio=bool(args.skip_missing_audio),
        tempo_embeddings=args.tempo_embeddings,
        sample_rate=int(args.sample_rate),
        tempocnn_sample_rate=int(args.tempocnn_sample_rate),
        snippet_length_sec=args.snippet_length_sec,
        hop_length=int(args.hop_length),
        n_fft=int(args.n_fft),
        use_csv_bpm=not bool(args.ignore_csv_bpm),
        auto_phase_align=not bool(args.disable_phase_align),
        phase_align_mode=str(args.phase_align_mode),
        phase_align_max_shift_sec=float(args.phase_align_max_shift_sec),
        phase_align_step_sec=float(args.phase_align_step_sec),
        auto_prepend_start_beats=not bool(args.disable_prepend_start_beats),
        subdivisions_per_beat=int(args.subdivisions_per_beat),
        beats_per_bar=int(args.beats_per_bar),
        phrase_bars=int(args.phrase_bars),
        profile_mode=str(args.profile_mode),
        pooling_mode=str(args.pooling_mode),
        pooling_topk=int(args.pooling_topk),
        normalize_per_beat=not bool(args.disable_normalize_per_beat),
        suppress_essentia_warnings=not bool(args.show_essentia_warnings),
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
