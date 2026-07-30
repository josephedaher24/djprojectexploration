"""MAEST Embedding Extractor (Essentia).

Extracts a track-level MAEST embedding from an audio file and writes a
clearly labeled JSON payload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from essentia.standard import MonoLoader, TensorflowPredictMAEST
except ImportError as exc:
    raise SystemExit(
        "Essentia MAEST extractor is unavailable. Install a TensorFlow-enabled "
        "Essentia build (for example: `uv add essentia-tensorflow`)."
    ) from exc


DEFAULT_MODEL_FILENAME = "discogs-maest-30s-pw-519l-2.pb"
DEFAULT_OUTPUT_NODE = "PartitionedCall/Identity_7"
DEFAULT_OUTPUT_FILENAME = "maest_embedding_discogs-maest-30s-pw.json"
EPS = 1e-12


def _to_project_relpath(path: Path, project_root: Path) -> str:
    """Return path relative to project_root when possible, else absolute."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


def _first_mp3(music_dir: Path) -> Path | None:
    mp3_files = sorted(music_dir.glob("*.mp3"))
    return mp3_files[0] if mp3_files else None


def _reduce_to_track_embedding(raw_predictions: np.ndarray) -> tuple[np.ndarray, str]:
    """Reduce model output into one fixed-length vector per track."""
    if raw_predictions.size == 0:
        raise ValueError("MAEST returned an empty prediction tensor.")

    if raw_predictions.ndim == 1:
        return raw_predictions.astype(np.float32), "none_already_vector"

    if raw_predictions.ndim == 2:
        # Typical shape: [segments, embedding_dim].
        return raw_predictions.mean(axis=0).astype(np.float32), "mean_over_segments"

    if raw_predictions.ndim == 4 and raw_predictions.shape[1] == 1 and raw_predictions.shape[2] >= 1:
        # Attention output shape: [segments, 1, tokens, embedding_dim].
        cls_tokens = raw_predictions[:, 0, 0, :]
        return cls_tokens.mean(axis=0).astype(np.float32), "mean_over_segments_cls_token"

    flattened = raw_predictions.reshape(raw_predictions.shape[0], -1)
    return flattened.mean(axis=0).astype(np.float32), "mean_over_segments_flattened"


def _rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + EPS))
    return float(20.0 * np.log10(max(rms, EPS)))


def _highest_rms_window(
    audio: np.ndarray,
    *,
    sample_rate: int,
    window_sec: float = 30.0,
    hop_sec: float = 1.0,
) -> tuple[np.ndarray, float, float, float]:
    """Return the highest-RMS sliding window and its start/end/RMS metadata."""
    total_samples = int(audio.size)
    if total_samples <= 0 or sample_rate <= 0:
        return audio, 0.0, 0.0, float("nan")

    window = min(total_samples, max(1, int(round(float(window_sec) * sample_rate))))
    hop = max(1, int(round(float(hop_sec) * sample_rate)))
    if total_samples <= window:
        return audio, 0.0, float(total_samples) / float(sample_rate), _rms_db(audio)

    starts = np.arange(0, total_samples - window + 1, hop, dtype=np.int64)
    final_start = int(total_samples - window)
    if starts.size == 0 or int(starts[-1]) != final_start:
        starts = np.append(starts, final_start)

    power = np.square(audio, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    window_energy = cumulative[starts + window] - cumulative[starts]
    best_start = int(starts[int(np.argmax(window_energy))])
    best_end = best_start + window
    best = audio[best_start:best_end]
    return (
        best,
        float(best_start) / float(sample_rate),
        float(best_end) / float(sample_rate),
        _rms_db(best),
    )


def extract_embedding_from_audio(
    audio: np.ndarray,
    model_file: Path,
    output_node: str,
    *,
    model: TensorflowPredictMAEST | None = None,
) -> tuple[np.ndarray, tuple[int, ...], str]:
    predictor = model or TensorflowPredictMAEST(graphFilename=str(model_file), output=output_node)
    raw_predictions = np.asarray(predictor(audio))
    embedding, reduction = _reduce_to_track_embedding(raw_predictions)
    return embedding, tuple(raw_predictions.shape), reduction


def extract_embedding(
    audio_file: Path,
    model_file: Path,
    output_node: str,
    *,
    model: TensorflowPredictMAEST | None = None,
) -> tuple[np.ndarray, tuple[int, ...], str]:
    audio = MonoLoader(filename=str(audio_file), sampleRate=16000, resampleQuality=4)()
    return extract_embedding_from_audio(audio, model_file, output_node, model=model)


def extract_peak_rms_embedding(
    audio_file: Path,
    model_file: Path,
    output_node: str,
    *,
    model: TensorflowPredictMAEST | None = None,
    window_sec: float = 30.0,
    hop_sec: float = 1.0,
) -> tuple[np.ndarray, tuple[int, ...], str, float, float, float]:
    sample_rate = 16000
    audio = MonoLoader(filename=str(audio_file), sampleRate=sample_rate, resampleQuality=4)()
    peak_audio, start_sec, end_sec, rms_db = _highest_rms_window(
        audio,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )
    embedding, raw_shape, reduction = extract_embedding_from_audio(
        peak_audio,
        model_file,
        output_node,
        model=model,
    )
    return embedding, raw_shape, reduction, start_sec, end_sec, rms_db


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_music_dir = project_root / "music"
    default_model_file = project_root / "models" / DEFAULT_MODEL_FILENAME
    default_output_file = default_music_dir / DEFAULT_OUTPUT_FILENAME

    parser = argparse.ArgumentParser(description="Extract MAEST embeddings with Essentia.")
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
        "--model-file",
        type=Path,
        default=default_model_file,
        help="Path to MAEST TensorFlow graph (.pb).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=default_output_file,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--output-node",
        default=DEFAULT_OUTPUT_NODE,
        help="TensorFlow output node used for embeddings.",
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

    audio_file = audio_file.expanduser().resolve()
    model_file = args.model_file.expanduser().resolve()
    output_file = args.output_file.expanduser()

    if not audio_file.exists():
        raise SystemExit(f"Audio file not found: {audio_file}")

    if not model_file.exists():
        raise SystemExit(
            f"MAEST model file not found: {model_file}\n"
            "Download a model from https://essentia.upf.edu/models.html#MAEST "
            "and pass it via --model-file."
        )

    embedding, raw_shape, reduction = extract_embedding(audio_file, model_file, args.output_node)

    payload = {
        "title": "MAEST Embedding (Essentia)",
        "filename": audio_file.name,
        "audio_file": _to_project_relpath(audio_file, project_root),
        "embedding_type": "maest",
        "model_name": model_file.stem,
        "model_file": _to_project_relpath(model_file, project_root),
        "output_node": args.output_node,
        "reduction": reduction,
        "raw_prediction_shape": list(raw_shape),
        "embedding_dimension": int(embedding.shape[0]),
        "embedding": embedding.tolist(),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print(f"Audio file: {_to_project_relpath(audio_file, project_root)}")
    print(f"Model file: {_to_project_relpath(model_file, project_root)}")
    print(f"Embedding dimension: {payload['embedding_dimension']}")
    print(f"Saved MAEST embedding JSON: {_to_project_relpath(output_file.resolve(), project_root)}")


if __name__ == "__main__":
    main()
