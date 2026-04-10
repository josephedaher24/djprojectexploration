"""Chroma embedding utilities with optional beat-synchronous pooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from essentia.standard import (
    FrameGenerator,
    HPCP,
    Key,
    MonoLoader,
    OnsetRate,
    RhythmExtractor2013,
    SpectralPeaks,
    Spectrum,
    Windowing,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PITCH_CLASS_LABELS = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
ENHARMONIC_KEY_MAP = {
    "AB": "G#",
    "BB": "A#",
    "CB": "B",
    "DB": "C#",
    "EB": "D#",
    "FB": "E",
    "GB": "F#",
}


def _to_project_relpath(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    """Return project-relative path when possible, else absolute path."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


def compute_frame_chroma(
    audio: np.ndarray,
    sample_rate: int = 44100,
    frame_size: int = 4096,
    hop_size: int = 1024,
    chroma_bins: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frame chroma [num_frames, chroma_bins] and frame times [num_frames]."""
    windowing = Windowing(type="hann")
    spectrum = Spectrum(size=frame_size)
    spectral_peaks = SpectralPeaks(
        sampleRate=sample_rate,
        minFrequency=40,
        maxFrequency=5000,
        maxPeaks=80,
        magnitudeThreshold=1e-5,
        orderBy="magnitude",
    )
    hpcp = HPCP(
        size=chroma_bins,
        sampleRate=sample_rate,
        referenceFrequency=440,
        minFrequency=60,
        maxFrequency=2000,
        normalized="unitSum",
        harmonics=4,
        nonLinear=False,
        weightType="cosine",
    )

    chroma_frames: list[np.ndarray] = []
    for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        spec = spectrum(windowing(frame))
        freqs, mags = spectral_peaks(spec)
        if len(freqs) == 0:
            chroma_frames.append(np.zeros(chroma_bins, dtype=np.float32))
        else:
            chroma_frames.append(np.asarray(hpcp(freqs, mags), dtype=np.float32))

    if not chroma_frames:
        raise ValueError("No chroma frames were generated from the input audio.")

    chroma = np.vstack(chroma_frames)  # [frames, bins]
    frame_times = (np.arange(chroma.shape[0], dtype=np.float32) * hop_size) / sample_rate
    return chroma, frame_times


def detect_beats(audio: np.ndarray) -> tuple[float, np.ndarray]:
    bpm, beat_times, *_ = RhythmExtractor2013(method="multifeature")(audio)
    return float(bpm), np.asarray(beat_times, dtype=np.float32)


def estimate_key(audio: np.ndarray, sample_rate: int, frame_size: int, hop_size: int) -> tuple[str, str, float]:
    """Estimate key/scale/strength using an Essentia HPCP+Key pipeline."""
    windowing = Windowing(type="blackmanharris62")
    spectrum = Spectrum(size=frame_size)
    spectral_peaks = SpectralPeaks(
        orderBy="magnitude",
        magnitudeThreshold=1e-5,
        minFrequency=20,
        maxFrequency=3500,
        maxPeaks=60,
    )
    hpcp_key = HPCP(
        size=36,
        referenceFrequency=440,
        sampleRate=sample_rate,
        bandPreset=False,
        minFrequency=20,
        maxFrequency=3500,
        weightType="cosine",
        nonLinear=False,
        windowSize=1.0,
    )
    key_detector = Key(
        profileType="edma",
        numHarmonics=4,
        pcpSize=36,
        slope=0.6,
        usePolyphony=True,
        useThreeChords=True,
    )

    hpcp_frames: list[np.ndarray] = []
    for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        spec = spectrum(windowing(frame))
        freqs, mags = spectral_peaks(spec)
        if len(freqs) == 0:
            continue
        hpcp_frames.append(np.asarray(hpcp_key(freqs, mags), dtype=np.float32))

    if not hpcp_frames:
        return "N", "unknown", 0.0

    mean_hpcp = np.vstack(hpcp_frames).mean(axis=0).astype(np.float32)
    key, scale, strength, _relative_strength = key_detector(mean_hpcp)
    return str(key), str(scale), float(strength)


def _normalize_key_label(key_label: str) -> str:
    normalized = key_label.strip().upper()
    normalized = ENHARMONIC_KEY_MAP.get(normalized, normalized)
    return normalized


def key_to_feature_vector(key_label: str, scale: str, strength: float) -> np.ndarray:
    """Encode key estimate as numeric features appended to the chroma embedding.

    Output dims: 12 (tonic one-hot) + 1 (mode) + 1 (strength) = 14.
    mode: 1.0=major, -1.0=minor, 0.0=unknown.
    """
    tonic_features = np.zeros(len(PITCH_CLASS_LABELS), dtype=np.float32)
    normalized_key = _normalize_key_label(key_label)
    if normalized_key in PITCH_CLASS_LABELS:
        tonic_features[PITCH_CLASS_LABELS.index(normalized_key)] = 1.0

    scale_norm = scale.strip().lower()
    if scale_norm == "major":
        mode_value = 1.0
    elif scale_norm == "minor":
        mode_value = -1.0
    else:
        mode_value = 0.0

    strength_value = float(np.clip(strength, 0.0, 1.0))
    return np.concatenate(
        [tonic_features, np.array([mode_value, strength_value], dtype=np.float32)],
        axis=0,
    ).astype(np.float32)


def detect_first_onset(audio: np.ndarray) -> float | None:
    _, onset_times = OnsetRate()(audio)
    onset_times = np.atleast_1d(np.asarray(onset_times, dtype=np.float32))
    if onset_times.size == 0:
        return None
    return float(onset_times[0])


def build_beat_grid_from_bpm(audio_duration: float, bpm: float, phase_anchor: float) -> np.ndarray:
    beat_period = 60.0 / bpm
    first_beat = max(0.0, phase_anchor)
    while first_beat - beat_period >= 0.0:
        first_beat -= beat_period
    beat_times = np.arange(first_beat, audio_duration + beat_period, beat_period, dtype=np.float32)
    return beat_times[beat_times <= audio_duration]


def pool_chroma_over_beats(
    chroma_frames: np.ndarray,
    frame_times: np.ndarray,
    beat_times: np.ndarray,
    audio_duration: float,
) -> np.ndarray:
    """Average frame-level chroma within beat intervals.

    Returns shape [num_beat_intervals, chroma_bins].
    """
    if beat_times.size < 2:
        return chroma_frames

    interval_edges = np.append(beat_times, audio_duration).astype(np.float32)
    pooled: list[np.ndarray] = []
    for i in range(interval_edges.size - 1):
        start = interval_edges[i]
        end = interval_edges[i + 1]
        mask = (frame_times >= start) & (frame_times < end)
        if np.any(mask):
            pooled.append(chroma_frames[mask].mean(axis=0))
        else:
            midpoint = (start + end) * 0.5
            nearest_idx = int(np.argmin(np.abs(frame_times - midpoint)))
            pooled.append(chroma_frames[nearest_idx])

    return np.vstack(pooled)


def _unit_sum_normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array [N, bins], got shape {arr.shape}.")
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums = np.where(np.abs(row_sums) <= 1e-12, 1.0, row_sums)
    return arr / row_sums


def summarize_pitch_class_sequence(
    chroma_sequence: np.ndarray,
    *,
    center_baseline: float | None = 1.0 / 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and std of unit-sum-normalized chroma with optional centering."""
    seq = _unit_sum_normalize_rows(chroma_sequence)
    if center_baseline is not None:
        baseline = float(center_baseline)
        if not np.isfinite(baseline):
            raise ValueError("center_baseline must be finite when provided.")
        seq = seq - baseline

    chroma_mean = seq.mean(axis=0).astype(np.float32)
    chroma_std = seq.std(axis=0).astype(np.float32)
    return chroma_mean, chroma_std


def summarize_chroma_embedding(
    chroma_sequence: np.ndarray,
    *,
    center_baseline: float | None = 1.0 / 12.0,
) -> np.ndarray:
    """Compute mean+std chroma embedding and L2-normalize it."""
    chroma_mean, chroma_std = summarize_pitch_class_sequence(
        chroma_sequence,
        center_baseline=center_baseline,
    )
    embedding = np.concatenate([chroma_mean, chroma_std]).astype(np.float32)
    norm = float(np.linalg.norm(embedding))
    if norm > 0:
        embedding /= norm
    return embedding


def generate_chroma_embedding(
    audio_file: str | Path,
    sample_rate: int = 44100,
    frame_size: int = 4096,
    hop_size: int = 1024,
    chroma_bins: int = 12,
    bpm: float | None = None,
    onset_time_ms: float | None = None,
    include_key_features: bool = True,
    center_baseline: float | None = 1.0 / 12.0,
) -> dict[str, Any]:
    """Generate a chroma embedding with optional manual beat-grid controls."""
    audio_path = Path(audio_file).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if chroma_bins <= 0:
        raise ValueError("chroma_bins must be positive.")
    if bpm is not None and bpm <= 0:
        raise ValueError("bpm must be positive.")
    if onset_time_ms is not None and onset_time_ms < 0:
        raise ValueError("onset_time_ms must be >= 0.")
    if onset_time_ms is not None and bpm is None:
        raise ValueError("onset_time_ms requires bpm.")
    if center_baseline is not None and not np.isfinite(float(center_baseline)):
        raise ValueError("center_baseline must be finite when provided.")

    audio = MonoLoader(filename=str(audio_path), sampleRate=sample_rate, resampleQuality=4)()
    audio_duration = audio.size / sample_rate
    chroma_frames, frame_times = compute_frame_chroma(
        audio=audio,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        chroma_bins=chroma_bins,
    )

    if bpm is not None:
        if onset_time_ms is not None:
            phase_anchor = min(max(0.0, onset_time_ms / 1000.0), audio_duration)
            beat_source = "manual_bpm_manual_onset_anchor"
        else:
            detected_anchor = detect_first_onset(audio)
            phase_anchor = detected_anchor if detected_anchor is not None else 0.0
            beat_source = "manual_bpm_first_onset_anchor"
        beat_times = build_beat_grid_from_bpm(audio_duration=audio_duration, bpm=bpm, phase_anchor=phase_anchor)
        used_bpm = float(bpm)
    else:
        used_bpm, beat_times = detect_beats(audio)
        phase_anchor = None
        beat_source = "detected_rhythmextractor2013_multifeature"

    beat_chroma = pool_chroma_over_beats(
        chroma_frames=chroma_frames,
        frame_times=frame_times,
        beat_times=beat_times,
        audio_duration=audio_duration,
    )
    pitch_class_mean, pitch_class_std = summarize_pitch_class_sequence(
        beat_chroma,
        center_baseline=center_baseline,
    )
    base_embedding = np.concatenate([pitch_class_mean, pitch_class_std]).astype(np.float32)
    base_norm = float(np.linalg.norm(base_embedding))
    if base_norm > 0:
        base_embedding /= base_norm
    detected_key, detected_scale, key_strength = estimate_key(
        audio=audio,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
    )

    if include_key_features:
        key_features = key_to_feature_vector(
            key_label=detected_key,
            scale=detected_scale,
            strength=key_strength,
        )
        embedding = np.concatenate([base_embedding, key_features]).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding /= norm
        embedding_type = "chroma_mean_std_plus_key"
        title = "Beat-Synchronous Chroma Embedding (Mean+Std + Key)"
    else:
        key_features = np.zeros(0, dtype=np.float32)
        embedding = base_embedding
        embedding_type = "chroma_mean_std"
        title = "Beat-Synchronous Chroma Embedding (Mean+Std)"

    return {
        "title": title,
        "filename": audio_path.name,
        "audio_file": _to_project_relpath(audio_path),
        "embedding_type": embedding_type,
        "chroma_bins": int(chroma_bins),
        "base_embedding_dimension": int(base_embedding.shape[0]),
        "key_feature_dimension": int(key_features.shape[0]),
        "embedding_dimension": int(embedding.shape[0]),
        "embedding": embedding.tolist(),
        "pitch_class_mean": pitch_class_mean.tolist(),
        "pitch_class_std": pitch_class_std.tolist(),
        "key_estimate": {
            "key": detected_key,
            "scale": detected_scale,
            "strength": float(key_strength),
        },
        "beat_pooling": {
            "bpm": float(used_bpm),
            "beat_count": int(beat_times.size),
            "beat_source": beat_source,
            "phase_anchor_seconds": None if phase_anchor is None else float(phase_anchor),
        },
        "config": {
            "sample_rate": int(sample_rate),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "manual_bpm": None if bpm is None else float(bpm),
            "manual_onset_time_ms": None if onset_time_ms is None else float(onset_time_ms),
            "include_key_features": bool(include_key_features),
            "center_baseline": None if center_baseline is None else float(center_baseline),
        },
    }


def test_losingit_embedding() -> dict[str, Any]:
    """Helper test for music/losingit.mp3 with bpm=125 and onset_time_ms=0."""
    project_root = Path(__file__).resolve().parents[2]
    audio_file = project_root / "music" / "losingit.mp3"
    output_file = project_root / "music" / "losingit_chroma_embedding_bpm125_onset0.json"

    result = generate_chroma_embedding(audio_file=audio_file, bpm=125.0, onset_time_ms=0.0)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"Audio file: {audio_file}")
    print("Manual BPM: 125.0")
    print("Manual onset anchor: 0 ms")
    print(f"Embedding dimension: {result['embedding_dimension']}")
    print(f"Saved test embedding: {output_file}")
    return result
