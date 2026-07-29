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
MAEST_SAMPLE_RATE = 16000
MAEST_MEL_HOP_SAMPLES = 256
MAEST_PATCH_SIZE_FRAMES = 1876
MAEST_MIN_INPUT_SAMPLES = MAEST_MEL_HOP_SAMPLES * MAEST_PATCH_SIZE_FRAMES


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


def _pad_short_maest_audio(audio: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad audio to the minimum MAEST patch length when a section is too short."""
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError("Cannot extract MAEST embedding from empty audio.")
    if signal.size >= MAEST_MIN_INPUT_SAMPLES:
        return signal, 0
    pad_samples = int(MAEST_MIN_INPUT_SAMPLES - signal.size)
    return np.pad(signal, (0, pad_samples), mode="constant"), pad_samples


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
    signal, padded_samples = _pad_short_maest_audio(audio)
    raw_predictions = np.asarray(predictor(signal))
    embedding, reduction = _reduce_to_track_embedding(raw_predictions)
    if padded_samples:
        padded_sec = padded_samples / float(MAEST_SAMPLE_RATE)
        reduction = f"{reduction}+silence_padded_short_input_{padded_sec:.3f}s"
    return embedding, tuple(raw_predictions.shape), reduction


def predict_discogs519_genres_from_audio(
    audio: np.ndarray,
    *,
    embedding_model: TensorflowPredictMAEST,
    genre_model: object,
) -> np.ndarray:
    """Run the official Discogs-519 head from MAEST's required final output.

    ``embedding_model`` must be configured with ``PartitionedCall/Identity_12``.
    The generic similarity embedding remains the seventh-layer output and is not
    suitable input for this head.
    """
    from essentia import Pool

    signal, _ = _pad_short_maest_audio(audio)
    final_layer_embeddings = np.asarray(embedding_model(signal))
    pool = Pool()
    pool.set("embeddings", final_layer_embeddings)
    prediction_pool = genre_model(pool)
    predictions = np.asarray(prediction_pool["PartitionedCall/Identity_1"], dtype=np.float32)
    if predictions.ndim == 0:
        raise ValueError("Discogs-519 genre head returned a scalar instead of class predictions.")
    if predictions.shape[-1] != 519:
        raise ValueError(
            "Discogs-519 genre head returned an unexpected class dimension: "
            f"{predictions.shape}. Expected the last dimension to be 519."
        )
    # The head returns one prediction vector per MAEST patch. Pool patches to one
    # fixed-width, track-level multi-label score vector.
    return predictions.reshape(-1, predictions.shape[-1]).mean(axis=0, dtype=np.float32)


def extract_peak_rms_embedding_from_audio(
    audio: np.ndarray,
    model_file: Path,
    output_node: str,
    *,
    model: TensorflowPredictMAEST | None = None,
    window_sec: float = 30.0,
    hop_sec: float = 1.0,
) -> tuple[np.ndarray, tuple[int, ...], str, float, float, float]:
    """Peak-RMS MAEST embedding without loading the audio a second time."""
    peak_audio, start_sec, end_sec, rms_db = _highest_rms_window(
        audio, sample_rate=MAEST_SAMPLE_RATE, window_sec=window_sec, hop_sec=hop_sec,
    )
    embedding, raw_shape, reduction = extract_embedding_from_audio(
        peak_audio, model_file, output_node, model=model,
    )
    return embedding, raw_shape, reduction, start_sec, end_sec, rms_db


def _cls_embedding(raw_prediction: np.ndarray) -> np.ndarray:
    """Return one embedding vector from a single MAEST inference window."""
    if raw_prediction.size == 0:
        raise ValueError("MAEST returned an empty prediction tensor for a segment.")

    if raw_prediction.ndim == 1:
        return raw_prediction.astype(np.float32)
    if raw_prediction.ndim == 2:
        return raw_prediction[0].astype(np.float32)
    if raw_prediction.ndim == 4 and raw_prediction.shape[0] >= 1 and raw_prediction.shape[1] == 1:
        # MAEST attention-layer output: [batch, 1, tokens, embedding_dim].
        return raw_prediction[0, 0, 0, :].astype(np.float32)

    return raw_prediction.reshape(raw_prediction.shape[0], -1)[0].astype(np.float32)


def _cls_embeddings(raw_predictions: np.ndarray) -> np.ndarray:
    """Return one CLS vector per MAEST inference patch."""
    if raw_predictions.size == 0:
        raise ValueError("MAEST returned an empty prediction tensor for a segment series.")
    if raw_predictions.ndim == 1:
        return raw_predictions.astype(np.float32)[None, :]
    if raw_predictions.ndim == 2:
        return raw_predictions.astype(np.float32)
    if raw_predictions.ndim == 4 and raw_predictions.shape[1] == 1:
        return raw_predictions[:, 0, 0, :].astype(np.float32)
    return raw_predictions.reshape(raw_predictions.shape[0], -1).astype(np.float32)


def extract_segment_embeddings_from_audio(
    audio: np.ndarray,
    model_file: Path,
    output_node: str,
    *,
    model: TensorflowPredictMAEST | None = None,
    sample_rate: int = 16000,
    window_sec: float = 30.0,
    hop_sec: float = 15.0,
    include_partial: bool = False,
) -> dict[str, np.ndarray | float | int | str]:
    """Extract a time-stamped MAEST CLS series with one inference per track.

    Essentia generates overlapping MAEST patches internally. ``window_sec`` must
    match the selected MAEST model duration (30 seconds for this project model);
    ``hop_sec`` is converted to the model's mel-frame hop, so its effective value
    is quantized to the nearest 256-sample frame.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if window_sec <= 0 or hop_sec <= 0:
        raise ValueError("window_sec and hop_sec must be positive.")

    if sample_rate != MAEST_SAMPLE_RATE:
        raise ValueError(f"MAEST requires {MAEST_SAMPLE_RATE} Hz audio, got {sample_rate}.")
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError("Cannot extract MAEST segment embeddings from empty audio.")

    model_window_sec = MAEST_PATCH_SIZE_FRAMES * MAEST_MEL_HOP_SAMPLES / MAEST_SAMPLE_RATE
    if not np.isclose(window_sec, model_window_sec, rtol=0.0, atol=0.05):
        raise ValueError(
            f"window_sec={window_sec:g} does not match this MAEST model's "
            f"{model_window_sec:.3f}-second patch duration."
        )
    patch_hop_frames = max(1, int(round(float(hop_sec) * MAEST_SAMPLE_RATE / MAEST_MEL_HOP_SAMPLES)))
    effective_hop_sec = patch_hop_frames * MAEST_MEL_HOP_SAMPLES / MAEST_SAMPLE_RATE
    last_patch_mode = "repeat" if include_partial else "discard"
    predictor = model or TensorflowPredictMAEST(
        graphFilename=str(model_file),
        output=output_node,
        patchSize=MAEST_PATCH_SIZE_FRAMES,
        patchHopSize=patch_hop_frames,
        lastPatchMode=last_patch_mode,
    )
    raw_predictions = np.asarray(predictor(signal))
    embeddings = _cls_embeddings(raw_predictions)
    start_sec = np.arange(embeddings.shape[0], dtype=np.float32) * np.float32(effective_hop_sec)
    duration_sec = float(signal.size) / float(sample_rate)
    end_sec = np.minimum(start_sec + np.float32(model_window_sec), duration_sec).astype(np.float32)
    raw_shape = np.asarray(raw_predictions.shape, dtype=np.int32)
    return {
        "embeddings": embeddings,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "center_sec": (start_sec + end_sec) * 0.5,
        "raw_prediction_shape": np.tile(raw_shape, (embeddings.shape[0], 1)),
        "sample_rate": int(sample_rate),
        "window_sec": float(window_sec),
        "hop_sec": float(effective_hop_sec),
        "last_window_mode": last_patch_mode,
    }


def extract_embedding(
    audio_file: Path,
    model_file: Path,
    output_node: str,
    *,
    model: TensorflowPredictMAEST | None = None,
) -> tuple[np.ndarray, tuple[int, ...], str]:
    audio = MonoLoader(filename=str(audio_file), sampleRate=16000, resampleQuality=4)()
    return extract_embedding_from_audio(audio, model_file, output_node, model=model)


def extract_segment_embeddings(
    audio_file: Path,
    model_file: Path,
    output_node: str,
    *,
    model: TensorflowPredictMAEST | None = None,
    window_sec: float = 30.0,
    hop_sec: float = 15.0,
    include_partial: bool = False,
) -> dict[str, np.ndarray | float | int | str]:
    """Load an audio file and return explicit, time-stamped MAEST segment embeddings."""
    sample_rate = 16000
    audio = MonoLoader(filename=str(audio_file), sampleRate=sample_rate, resampleQuality=4)()
    return extract_segment_embeddings_from_audio(
        audio,
        model_file,
        output_node,
        model=model,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
        include_partial=include_partial,
    )


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
    return extract_peak_rms_embedding_from_audio(
        audio, model_file, output_node, model=model, window_sec=window_sec, hop_sec=hop_sec,
    )


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
