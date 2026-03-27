"""DEAM valence/arousal prediction utilities using Essentia TensorFlow models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from essentia.standard import MonoLoader, TensorflowPredict2D, TensorflowPredictMusiCNN
except ImportError as exc:
    _IMPORT_ERROR = exc
    MonoLoader = None  # type: ignore[assignment]
    TensorflowPredict2D = None  # type: ignore[assignment]
    TensorflowPredictMusiCNN = None  # type: ignore[assignment]
else:
    _IMPORT_ERROR = None


DEFAULT_MUSICNN_OUTPUT = "model/dense/BiasAdd"
DEFAULT_VGGISH_OUTPUT = "model/vggish/embeddings"
DEFAULT_DEAM_OUTPUT = "model/Identity"
DEAM_VALUE_RANGE = (1.0, 9.0)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _to_project_relpath(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    """Return project-relative path when possible, else absolute path."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


def _validate_model_file(model_file: str | Path, name: str) -> Path:
    path = Path(model_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} model file not found: {path}")
    return path


def _require_essentia() -> None:
    if _IMPORT_ERROR is None:
        return
    raise ImportError(
        "Essentia TensorFlow predictors are unavailable. Install a TensorFlow-enabled "
        "Essentia build (for example: `uv add essentia-tensorflow`)."
    ) from _IMPORT_ERROR


def predict_deam_valence_arousal_musicnn(
    audio_file: str | Path,
    embedding_model_file: str | Path,
    regression_model_file: str | Path,
    embedding_output: str = DEFAULT_MUSICNN_OUTPUT,
    regression_output: str = DEFAULT_DEAM_OUTPUT,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    """Predict DEAM valence/arousal for one track using MusicNN embeddings.

    Returns frame/segment-level predictions plus track-level mean and std.
    """
    _require_essentia()

    audio_path = Path(audio_file).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")

    embedding_path = _validate_model_file(embedding_model_file, "Embedding")
    regression_path = _validate_model_file(regression_model_file, "Regression")

    audio = MonoLoader(filename=str(audio_path), sampleRate=sample_rate, resampleQuality=4)()
    embedding_model = TensorflowPredictMusiCNN(
        graphFilename=str(embedding_path),
        output=embedding_output,
    )
    embeddings = np.asarray(embedding_model(audio), dtype=np.float32)

    regression_model = TensorflowPredict2D(
        graphFilename=str(regression_path),
        output=regression_output,
    )
    predictions = np.asarray(regression_model(embeddings), dtype=np.float32)

    return _build_prediction_payload(
        title="DEAM Valence/Arousal Prediction (MusicNN embeddings)",
        audio_path=audio_path,
        embedding_path=embedding_path,
        regression_path=regression_path,
        embedding_output=embedding_output,
        regression_output=regression_output,
        embeddings=embeddings,
        predictions=predictions,
    )


def predict_deam_valence_arousal_vggish(
    audio_file: str | Path,
    embedding_model_file: str | Path,
    regression_model_file: str | Path,
    embedding_output: str = DEFAULT_VGGISH_OUTPUT,
    regression_output: str = DEFAULT_DEAM_OUTPUT,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    """Predict DEAM valence/arousal for one track using VGGish embeddings."""
    _require_essentia()

    audio_path = Path(audio_file).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")

    embedding_path = _validate_model_file(embedding_model_file, "Embedding")
    regression_path = _validate_model_file(regression_model_file, "Regression")

    try:
        from essentia.standard import TensorflowPredictVGGish
    except ImportError as exc:
        raise ImportError(
            "TensorflowPredictVGGish is unavailable in this Essentia build. "
            "Use a TensorFlow-enabled Essentia package."
        ) from exc

    audio = MonoLoader(filename=str(audio_path), sampleRate=sample_rate, resampleQuality=4)()
    embedding_model = TensorflowPredictVGGish(
        graphFilename=str(embedding_path),
        output=embedding_output,
    )
    embeddings = np.asarray(embedding_model(audio), dtype=np.float32)

    regression_model = TensorflowPredict2D(
        graphFilename=str(regression_path),
        output=regression_output,
    )
    predictions = np.asarray(regression_model(embeddings), dtype=np.float32)

    return _build_prediction_payload(
        title="DEAM Valence/Arousal Prediction (VGGish embeddings)",
        audio_path=audio_path,
        embedding_path=embedding_path,
        regression_path=regression_path,
        embedding_output=embedding_output,
        regression_output=regression_output,
        embeddings=embeddings,
        predictions=predictions,
    )


def predict_deam_valence_arousal_batch_musicnn(
    audio_files: Sequence[str | Path],
    embedding_model_file: str | Path,
    regression_model_file: str | Path,
    embedding_output: str = DEFAULT_MUSICNN_OUTPUT,
    regression_output: str = DEFAULT_DEAM_OUTPUT,
    sample_rate: int = 16000,
) -> list[dict[str, Any]]:
    """Predict DEAM valence/arousal for multiple audio files."""
    results: list[dict[str, Any]] = []
    for audio_file in audio_files:
        results.append(
            predict_deam_valence_arousal_musicnn(
                audio_file=audio_file,
                embedding_model_file=embedding_model_file,
                regression_model_file=regression_model_file,
                embedding_output=embedding_output,
                regression_output=regression_output,
                sample_rate=sample_rate,
            )
        )
    return results


def save_deam_prediction_json(prediction: dict[str, Any], output_file: str | Path) -> Path:
    """Save one prediction payload to JSON and return the output path."""
    output_path = Path(output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(prediction, f, indent=4)
    return output_path


def _build_prediction_payload(
    title: str,
    audio_path: Path,
    embedding_path: Path,
    regression_path: Path,
    embedding_output: str,
    regression_output: str,
    embeddings: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    if predictions.size == 0:
        raise ValueError("DEAM model returned an empty prediction tensor.")
    if predictions.ndim == 1:
        predictions = predictions[np.newaxis, :]
    if predictions.shape[-1] != 2:
        raise ValueError(
            f"Expected DEAM output dimension 2 ([valence, arousal]), got shape {predictions.shape}."
        )

    valence = predictions[:, 0]
    arousal = predictions[:, 1]

    return {
        "title": title,
        "filename": audio_path.name,
        "audio_file": _to_project_relpath(audio_path),
        "task": "deam_valence_arousal_regression",
        "dimensions": ["valence", "arousal"],
        "value_range": [DEAM_VALUE_RANGE[0], DEAM_VALUE_RANGE[1]],
        "models": {
            "embedding_model_file": _to_project_relpath(embedding_path),
            "embedding_output": embedding_output,
            "regression_model_file": _to_project_relpath(regression_path),
            "regression_output": regression_output,
        },
        "embedding_shape": list(embeddings.shape),
        "prediction_shape": list(predictions.shape),
        "segment_predictions": [
            {"valence": float(v), "arousal": float(a)} for v, a in zip(valence, arousal, strict=False)
        ],
        "track_prediction": {
            "valence": float(valence.mean()),
            "arousal": float(arousal.mean()),
            "valence_std": float(valence.std()),
            "arousal_std": float(arousal.std()),
            "num_segments": int(predictions.shape[0]),
        },
    }
