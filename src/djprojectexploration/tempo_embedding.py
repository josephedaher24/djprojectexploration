"""Tempo embedding utilities based on Essentia TempoCNN."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

try:
    from essentia.standard import MonoLoader, TempoCNN
except ImportError as exc:
    raise SystemExit(
        "Essentia TempoCNN extractor is unavailable. Install a TensorFlow-enabled "
        "Essentia build (for example: `uv add essentia-tensorflow`)."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPOCNN_MODEL_FILENAME = "deeptemp-k16-3.pb"
DEFAULT_TEMPOCNN_MODEL_URL = "https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k16-3.pb"
DEFAULT_TEMPOCNN_MODEL_FILE = PROJECT_ROOT / "models" / DEFAULT_TEMPOCNN_MODEL_FILENAME
DEFAULT_SAMPLE_RATE = 11025
DEFAULT_WINDOW_SEC = 12.0
DEFAULT_HOP_SEC = 6.0
DEFAULT_RMS_PERCENTILE = 20.0


def _to_project_relpath(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    """Return path relative to project_root when possible, else absolute path."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


def resolve_tempocnn_model_file(
    model_file: str | Path | None = None,
    *,
    auto_download: bool = False,
    model_url: str = DEFAULT_TEMPOCNN_MODEL_URL,
) -> Path:
    """Resolve a TempoCNN model file, optionally downloading it."""
    if model_file is not None:
        resolved = Path(model_file).expanduser().resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(
            f"TempoCNN model file not found: {resolved}. "
            f"Download from {model_url} or pass --auto-download-model."
        )

    candidates = [
        (PROJECT_ROOT / "models" / DEFAULT_TEMPOCNN_MODEL_FILENAME).resolve(),
        (PROJECT_ROOT / "models" / "tempo" / DEFAULT_TEMPOCNN_MODEL_FILENAME).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    target = candidates[0]
    if auto_download:
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(model_url, target)
        return target.resolve()

    raise FileNotFoundError(
        "TempoCNN model file not found. Tried:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + f"\nDownload from {model_url} or use auto_download=True."
    )


def compute_local_times(
    count: int,
    *,
    duration_sec: float,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
) -> np.ndarray:
    """Return TempoCNN local-window center times in seconds."""
    if count <= 0:
        return np.array([], dtype=np.float32)
    centers = (window_sec * 0.5) + np.arange(count, dtype=np.float32) * float(hop_sec)
    return np.clip(centers, 0.0, float(duration_sec)).astype(np.float32)


def compute_break_aware_tempo_confidence(
    *,
    global_bpm: float,
    local_bpm: np.ndarray,
    local_probs: np.ndarray,
    audio: np.ndarray,
    sr: int,
    local_times: np.ndarray | None = None,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    rms_percentile: float = DEFAULT_RMS_PERCENTILE,
) -> dict[str, Any]:
    """Compute one tempo-confidence score while down-weighting beatless sections."""
    tempo_hat = float(global_bpm)
    lbpm = np.asarray(local_bpm, dtype=np.float32)
    lprob = np.clip(np.asarray(local_probs, dtype=np.float32), 0.0, 1.0)

    win = max(1, int(round(float(window_sec) * sr)))
    hop = max(1, int(round(float(hop_sec) * sr)))
    duration = float(audio.size) / float(sr)

    if audio.size < win:
        rms = np.array([float(np.sqrt(np.mean(np.square(audio)) + 1e-12))], dtype=np.float32)
    else:
        starts = np.arange(0, audio.size - win + 1, hop, dtype=np.int64)
        rms = np.array(
            [float(np.sqrt(np.mean(np.square(audio[s : s + win])) + 1e-12)) for s in starts],
            dtype=np.float32,
        )

    n = int(min(lbpm.size, lprob.size, rms.size))
    if n == 0:
        return {
            "tempo_bpm": tempo_hat,
            "confidence": 0.0,
            "confidence_active_agreement": 0.0,
            "mean_prob_active": 0.0,
            "active_fraction": 0.0,
            "active_windows": 0,
            "total_windows": 0,
            "active_mask": np.array([], dtype=bool),
            "window_starts": np.array([], dtype=np.float32),
            "window_ends": np.array([], dtype=np.float32),
            "window_centers": np.array([], dtype=np.float32),
        }

    lbpm = lbpm[:n]
    lprob = lprob[:n]
    rms = rms[:n]

    if local_times is not None and len(local_times) >= n:
        centers = np.asarray(local_times[:n], dtype=np.float32)
    else:
        centers = compute_local_times(
            n,
            duration_sec=duration,
            window_sec=window_sec,
            hop_sec=hop_sec,
        )
    centers = np.clip(centers, 0.0, duration)

    half_window = 0.5 * float(window_sec)
    window_starts = np.clip(centers - half_window, 0.0, duration).astype(np.float32)
    window_ends = np.clip(centers + half_window, 0.0, duration).astype(np.float32)

    threshold = float(np.percentile(rms, float(rms_percentile)))
    active = rms > threshold
    active_float = active.astype(np.float32)
    active_fraction = float(np.mean(active_float))

    tol = max(3.0, 0.04 * tempo_hat)
    err = np.minimum.reduce(
        [
            np.abs(lbpm - tempo_hat),
            np.abs(lbpm - 0.5 * tempo_hat),
            np.abs(lbpm - 2.0 * tempo_hat),
        ]
    )
    agree = (err <= tol).astype(np.float32)

    weights = lprob * active_float
    if float(np.sum(weights)) > 0.0:
        conf_active = float(np.average(agree, weights=weights))
    else:
        conf_active = 0.0

    if np.any(active):
        mean_prob_active = float(np.mean(lprob[active]))
    else:
        mean_prob_active = 0.0

    confidence = float(0.7 * conf_active + 0.3 * mean_prob_active)

    return {
        "tempo_bpm": tempo_hat,
        "confidence": confidence,
        "confidence_active_agreement": conf_active,
        "mean_prob_active": mean_prob_active,
        "active_fraction": active_fraction,
        "active_windows": int(np.sum(active)),
        "total_windows": int(n),
        "active_mask": active,
        "window_starts": window_starts,
        "window_ends": window_ends,
        "window_centers": centers,
    }


def generate_tempo_embedding(
    audio_file: str | Path,
    *,
    model_file: str | Path | None = None,
    auto_download_model: bool = False,
    model_url: str = DEFAULT_TEMPOCNN_MODEL_URL,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    resample_quality: int = 4,
    snippet_length_sec: float | None = None,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    rms_percentile: float = DEFAULT_RMS_PERCENTILE,
) -> dict[str, Any]:
    """Generate a tempo embedding payload for one track."""
    audio_path = Path(audio_file).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    resolved_model_file = resolve_tempocnn_model_file(
        model_file=model_file,
        auto_download=auto_download_model,
        model_url=model_url,
    )

    audio = MonoLoader(filename=str(audio_path), sampleRate=int(sample_rate), resampleQuality=int(resample_quality))()
    if snippet_length_sec is not None:
        if snippet_length_sec <= 0:
            raise ValueError("snippet_length_sec must be positive when provided.")
        max_samples = int(float(snippet_length_sec) * int(sample_rate))
        audio = audio[: min(audio.size, max_samples)]

    duration_sec = float(audio.size) / float(sample_rate)
    global_bpm, local_bpm_raw, local_probs_raw = TempoCNN(graphFilename=str(resolved_model_file))(audio)
    global_bpm = float(global_bpm)
    local_bpm = np.asarray(local_bpm_raw, dtype=np.float32)
    local_probs = np.asarray(local_probs_raw, dtype=np.float32)
    local_times = compute_local_times(
        int(local_bpm.size),
        duration_sec=duration_sec,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )

    confidence_summary = compute_break_aware_tempo_confidence(
        global_bpm=global_bpm,
        local_bpm=local_bpm,
        local_probs=local_probs,
        audio=audio,
        sr=int(sample_rate),
        local_times=local_times,
        window_sec=window_sec,
        hop_sec=hop_sec,
        rms_percentile=rms_percentile,
    )

    embedding = np.array([global_bpm, float(confidence_summary["confidence"])], dtype=np.float32)

    return {
        "title": "Tempo Embedding (TempoCNN)",
        "filename": audio_path.name,
        "audio_file": _to_project_relpath(audio_path),
        "embedding_type": "tempo",
        "embedding_subtype": "tempocnn_break_aware",
        "embedding_dimension": int(embedding.size),
        "embedding": embedding.tolist(),
        "tempo_bpm": global_bpm,
        "confidence": float(confidence_summary["confidence"]),
        "confidence_active_agreement": float(confidence_summary["confidence_active_agreement"]),
        "mean_prob_active": float(confidence_summary["mean_prob_active"]),
        "active_fraction": float(confidence_summary["active_fraction"]),
        "active_windows": int(confidence_summary["active_windows"]),
        "total_windows": int(confidence_summary["total_windows"]),
        "local_bpm": local_bpm.tolist(),
        "local_probability": local_probs.tolist(),
        "local_times_sec": local_times.tolist(),
        "local_active_mask": confidence_summary["active_mask"].astype(np.int8).tolist(),
        "window_starts_sec": confidence_summary["window_starts"].tolist(),
        "window_ends_sec": confidence_summary["window_ends"].tolist(),
        "window_centers_sec": confidence_summary["window_centers"].tolist(),
        "sample_rate": int(sample_rate),
        "resample_quality": int(resample_quality),
        "snippet_length_sec": (None if snippet_length_sec is None else float(snippet_length_sec)),
        "window_sec": float(window_sec),
        "hop_sec": float(hop_sec),
        "rms_percentile": float(rms_percentile),
        "model_file": _to_project_relpath(resolved_model_file),
        "model_url": model_url,
    }

