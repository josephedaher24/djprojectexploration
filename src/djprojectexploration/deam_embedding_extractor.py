"""Single-track DEAM valence/arousal embedding extractor (.npz)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from djprojectexploration.deam_valence_arousal import (
    DEFAULT_DEAM_MODEL_FILE,
    DEFAULT_DEAM_OUTPUT,
    DEFAULT_MUSICNN_MODEL_FILE,
    DEAM_VALUE_RANGE,
    generate_deam_embedding,
)


DEFAULT_OUTPUT_FILENAME = "deam_embedding_musicnn.npz"


def _to_project_relpath(path: Path, project_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


def _first_mp3(music_dir: Path) -> Path | None:
    mp3_files = sorted(music_dir.glob("*.mp3"))
    return mp3_files[0] if mp3_files else None


def _string_array(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=np.str_)


def _build_single_track_npz_payload(
    *,
    prediction_payload: dict[str, Any],
    created_utc: str,
    sample_rate: int,
) -> dict[str, np.ndarray]:
    embedding = np.asarray(prediction_payload.get("embedding", []), dtype=np.float32).reshape(1, -1)
    valence_series = np.asarray(prediction_payload.get("valence_series", []), dtype=np.float32).reshape(-1)
    arousal_series = np.asarray(prediction_payload.get("arousal_series", []), dtype=np.float32).reshape(-1)
    n_segments = int(min(valence_series.size, arousal_series.size))
    valence_series = valence_series[:n_segments]
    arousal_series = arousal_series[:n_segments]

    track = prediction_payload.get("track_prediction", {})
    models = prediction_payload.get("models", {})
    filename = str(prediction_payload.get("filename", ""))
    audio_file = str(prediction_payload.get("audio_file", ""))

    return {
        "embedding_type": np.array(str(prediction_payload.get("embedding_type", "deam_valence_arousal")), dtype=np.str_),
        "embedding_subtype": np.array(str(prediction_payload.get("embedding_subtype", "unknown")), dtype=np.str_),
        "playlist_csv": np.array("", dtype=np.str_),
        "created_utc": np.array(created_utc, dtype=np.str_),
        "num_tracks": np.array(1, dtype=np.int32),
        "embedding_dimension": np.array(int(embedding.shape[1]), dtype=np.int32),
        "embeddings": embedding.astype(np.float32),
        "track_numbers": np.array([1], dtype=np.int32),
        "titles": _string_array([Path(filename).stem]),
        "artists": _string_array([""]),
        "filenames": _string_array([filename]),
        "audio_paths": _string_array([audio_file]),
        "genres": _string_array([""]),
        "keys": _string_array([""]),
        "bpm": np.array([np.nan], dtype=np.float32),
        "onset_time": np.array([np.nan], dtype=np.float32),
        "key_shift": np.array([np.nan], dtype=np.float32),
        "deam_backend": np.array(str(prediction_payload.get("embedding_backend", "musicnn")), dtype=np.str_),
        "deam_value_range": np.asarray(
            prediction_payload.get("value_range", [DEAM_VALUE_RANGE[0], DEAM_VALUE_RANGE[1]]),
            dtype=np.float32,
        ).reshape(-1),
        "deam_num_segments": np.array([n_segments], dtype=np.int32),
        "deam_valence_mean": np.array([float(track.get("valence", np.nan))], dtype=np.float32),
        "deam_valence_min": np.array([float(track.get("valence_min", np.nan))], dtype=np.float32),
        "deam_valence_max": np.array([float(track.get("valence_max", np.nan))], dtype=np.float32),
        "deam_valence_std": np.array([float(track.get("valence_std", np.nan))], dtype=np.float32),
        "deam_arousal_mean": np.array([float(track.get("arousal", np.nan))], dtype=np.float32),
        "deam_arousal_min": np.array([float(track.get("arousal_min", np.nan))], dtype=np.float32),
        "deam_arousal_max": np.array([float(track.get("arousal_max", np.nan))], dtype=np.float32),
        "deam_arousal_std": np.array([float(track.get("arousal_std", np.nan))], dtype=np.float32),
        "deam_series_start_index": np.array([0], dtype=np.int64),
        "deam_series_count": np.array([n_segments], dtype=np.int32),
        "deam_valence_series_flat": valence_series.astype(np.float32),
        "deam_arousal_series_flat": arousal_series.astype(np.float32),
        "deam_embedding_model_file": np.array(str(models.get("embedding_model_file", "")), dtype=np.str_),
        "deam_embedding_output": np.array(str(models.get("embedding_output", "")), dtype=np.str_),
        "deam_regression_model_file": np.array(str(models.get("regression_model_file", "")), dtype=np.str_),
        "deam_regression_output": np.array(str(models.get("regression_output", DEFAULT_DEAM_OUTPUT)), dtype=np.str_),
        "config_sample_rate": np.array(int(sample_rate), dtype=np.int32),
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_music_dir = project_root / "music"
    default_output_file = default_music_dir / DEFAULT_OUTPUT_FILENAME

    parser = argparse.ArgumentParser(description="Extract single-track DEAM valence/arousal embedding to NPZ.")
    parser.add_argument(
        "--audio-file",
        type=Path,
        default=None,
        help="Path to an input audio file. If omitted, the first .mp3 in --music-dir is used.",
    )
    parser.add_argument(
        "--music-dir",
        type=Path,
        default=default_music_dir,
        help="Directory searched for .mp3 files when --audio-file is not provided.",
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
        default=DEFAULT_MUSICNN_MODEL_FILE,
        help="Embedding TensorFlow graph (.pb).",
    )
    parser.add_argument(
        "--regression-model-file",
        type=Path,
        default=DEFAULT_DEAM_MODEL_FILE,
        help="DEAM regression TensorFlow graph (.pb).",
    )
    parser.add_argument(
        "--embedding-output",
        default=None,
        help="Optional embedding model output node. Defaults depend on --embedding-backend.",
    )
    parser.add_argument(
        "--regression-output",
        default=DEFAULT_DEAM_OUTPUT,
        help="DEAM regression model output node.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=default_output_file,
        help="Output NPZ file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    if args.audio_file is None:
        audio_file = _first_mp3(args.music_dir)
        if audio_file is None:
            raise SystemExit(f"No .mp3 files found in: {args.music_dir}")
    else:
        audio_file = args.audio_file

    resolved_audio_file = audio_file.expanduser().resolve()
    if not resolved_audio_file.exists():
        raise SystemExit(f"Audio file not found: {resolved_audio_file}")

    prediction_payload = generate_deam_embedding(
        audio_file=resolved_audio_file,
        embedding_model_file=args.embedding_model_file,
        regression_model_file=args.regression_model_file,
        embedding_backend=str(args.embedding_backend),
        embedding_output=args.embedding_output,
        regression_output=str(args.regression_output),
        sample_rate=int(args.sample_rate),
    )

    created_utc = datetime.now(tz=timezone.utc).isoformat()
    payload = _build_single_track_npz_payload(
        prediction_payload=prediction_payload,
        created_utc=created_utc,
        sample_rate=int(args.sample_rate),
    )

    output_file = args.output_file.expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **payload)

    embedding_dim = int(np.asarray(payload["embedding_dimension"], dtype=np.int32).item())
    num_segments = int(np.asarray(payload["deam_num_segments"], dtype=np.int32).reshape(-1)[0])

    print(f"Audio file: {_to_project_relpath(resolved_audio_file, project_root)}")
    print(f"Embedding backend: {args.embedding_backend}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Segments: {num_segments}")
    print(f"Saved DEAM embedding NPZ: {_to_project_relpath(output_file.resolve(), project_root)}")


if __name__ == "__main__":
    main()
