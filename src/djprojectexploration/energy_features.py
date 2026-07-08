"""Energy-model feature extraction for labeled DJ track CSVs.

The extractor writes an enriched CSV: original metadata and labels are preserved,
and deterministic audio/tempo features are appended for later model fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from djprojectexploration.tracklists import PROJECT_ROOT, optional_float, to_project_relpath

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "energy_features"
DEFAULT_ENERGY_EMBEDDING_DIR = PROJECT_ROOT / "data" / "energy_embeddings"
DEFAULT_ENERGY_MODEL_DIR = PROJECT_ROOT / "data" / "energy_models"
DEFAULT_TEMPO_EMBEDDING_DIR = PROJECT_ROOT / "data" / "tempo_embeddings"
DEFAULT_ENERGY_TARGET_MIN = 1.0
DEFAULT_ENERGY_TARGET_MAX = 9.0
EPS = 1e-12

SPECTRAL_BANDS_HZ = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 120.0),
    "low_mid": (120.0, 250.0),
    "mid": (250.0, 2000.0),
    "high": (2000.0, 8000.0),
}
FULL_ENERGY_MODEL_NAME = "full_tweedie_ridge_aries_ara"
FULL_ENERGY_FEATURE_KEYS = [
    "bpm",
    "spectral_centroid_hz",
    "spectral_tilt_db_per_oct",
    "spectral_rolloff_05_hz",
    "low_band_energy_ratio",
    "low_band_periodicity",
    "dynamic_range_db",
    "crest_factor_db",
    "onset_density",
    "onset_magnitude",
    "rhythmic_entropy",
    "band_flux_low",
    "band_flux_mid",
    "band_flux_high",
    "downbeat_strength_ratio",
    "phrase_consistency_4bar",
    "phrase_consistency_1bar",
    "hpss_percussive_ratio",
    "bassline_onset_density",
    "spectral_spread_hz",
    "section_transition_rate",
    "integrated_loudness_lufs",
    "short_term_loudness_mean_lufs",
    "short_term_loudness_std_lu",
    "short_term_loudness_range_lu",
    "rms_mean_db",
    "rms_median_db",
    "rms_std_db",
    "rms_iqr_db",
    "rms_range_db",
    "rms_var",
]
ENERGY_NPZ_FIELDNAMES = [
    "mix_name",
    "track_number",
    "title",
    "artists",
    "genre",
    "mp3_name",
    "energy",
    "bpm",
    "glm_energy_pred",
    "glm_energy_oof_pred",
    "glm_energy_residual",
    "glm_energy_oof_residual",
]


@dataclass(frozen=True)
class EnergyTrack:
    """Normalized row from a labeled energy CSV."""

    row_index: int
    row: dict[str, str]
    track_number: int
    title: str
    artists: str
    filename: str
    audio_path: Path
    genre: str
    labeled_bpm: float | None
    energy: float | None


@dataclass(frozen=True)
class EnergyModelResult:
    model_name: str
    feature_names: list[str]
    predictions: np.ndarray
    oof_predictions: np.ndarray
    best_alpha: float
    baseline_mae: float
    oof_mae: float
    oof_r2: float
    train_mae: float
    train_r2: float
    available: bool
    estimator: Any | None = None


def _project_relpath(path: Path) -> str:
    return to_project_relpath(path)


def _optional_float(value: Any) -> float | None:
    number = optional_float(value)
    if number is None:
        return None
    return number if np.isfinite(number) else None


def _safe_feature(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.8g}"
    if isinstance(value, np.floating):
        number = float(value)
        if not np.isfinite(number):
            return ""
        return f"{number:.8g}"
    return value


def _resolve_audio_path(
    row: dict[str, str],
    *,
    csv_dir: Path,
    music_dir: Path | None,
) -> Path | None:
    filepath = (row.get("filepath") or row.get("location") or "").strip()
    if filepath:
        path = Path(filepath).expanduser()
        if not path.is_absolute():
            path = csv_dir / path
        return path

    filename = (
        row.get("mp3_name")
        or row.get("filename")
        or row.get("file")
        or row.get("name")
        or row.get("title")
        or ""
    ).strip()
    if not filename:
        return None

    rel = Path(filename).expanduser()
    if rel.is_absolute():
        return rel

    candidates = [csv_dir / rel]
    if music_dir is not None:
        candidates.append(music_dir / rel.name)
    candidates.extend(
        [
            PROJECT_ROOT / rel,
            PROJECT_ROOT / "music" / rel.name,
            csv_dir / rel.name,
        ]
    )
    for candidate in candidates:
        if candidate.expanduser().exists():
            return candidate
    return candidates[0]


def load_energy_tracks(
    input_csv: str | Path,
    *,
    music_dir: str | Path | None = None,
    skip_missing_audio: bool = False,
) -> list[EnergyTrack]:
    """Load a labeled energy CSV and resolve each track's audio path."""
    csv_path = Path(input_csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    resolved_music_dir = None if music_dir is None else Path(music_dir).expanduser().resolve()
    tracks: list[EnergyTrack] = []
    skipped: list[tuple[int, str]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=1):
            track_number_raw = (row.get("track_number") or row.get("#") or "").strip()
            if track_number_raw:
                try:
                    track_number = int(track_number_raw)
                except ValueError:
                    track_number = row_index
            else:
                track_number = row_index

            audio_path = _resolve_audio_path(row, csv_dir=csv_path.parent, music_dir=resolved_music_dir)
            if audio_path is None:
                skipped.append((row_index, "<unresolved>"))
                if skip_missing_audio:
                    continue
                raise ValueError(f"Could not resolve audio path for row {row_index} in {csv_path}")

            resolved_audio = audio_path.expanduser().resolve()
            if not resolved_audio.exists():
                skipped.append((row_index, str(resolved_audio)))
                if skip_missing_audio:
                    continue
                raise FileNotFoundError(f"Audio file not found for row {row_index}: {resolved_audio}")

            title = (row.get("title") or row.get("name") or resolved_audio.stem).strip()
            artists = (row.get("artists") or row.get("artist") or "").strip()
            filename = (row.get("mp3_name") or row.get("filename") or resolved_audio.name).strip()
            genre = (row.get("genre") or "").strip()
            labeled_bpm = _optional_float(row.get("bpm"))
            energy = _optional_float(row.get("energy"))

            tracks.append(
                EnergyTrack(
                    row_index=row_index,
                    row=dict(row),
                    track_number=track_number,
                    title=title,
                    artists=artists,
                    filename=filename,
                    audio_path=resolved_audio,
                    genre=genre,
                    labeled_bpm=labeled_bpm,
                    energy=energy,
                )
            )

    if not tracks:
        detail = " All rows were missing audio files." if skipped else ""
        raise RuntimeError(f"No usable tracks found in {csv_path}.{detail}")
    return tracks


def default_output_file(input_csv: str | Path, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    csv_path = Path(input_csv).expanduser()
    return Path(output_dir).expanduser().resolve() / f"{csv_path.stem}_energy_features.csv"


def default_tempo_embeddings_file(input_csv: str | Path) -> Path | None:
    csv_path = Path(input_csv).expanduser()
    candidates = [
        DEFAULT_TEMPO_EMBEDDING_DIR / f"{csv_path.stem}.npz",
        DEFAULT_TEMPO_EMBEDDING_DIR / f"{csv_path.stem.replace('_energy', '_tracks')}.npz",
        DEFAULT_TEMPO_EMBEDDING_DIR / f"{csv_path.stem.replace('_tracks', '')}_tracks.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _decode_np_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_tempo_embedding_lookup(path: str | Path | None) -> dict[str, dict[str, float]]:
    """Load TempoCNN playlist NPZ values keyed by lowercase basename."""
    if path is None:
        return {}
    npz_path = Path(path).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"Tempo embeddings NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    filenames = [_decode_np_string(v) for v in data["filenames"]]
    fields = {
        "tempo_embedding_bpm": "tempo_bpm",
        "tempo_embedding_confidence": "tempo_confidence",
        "tempo_embedding_confidence_active_agreement": "tempo_confidence_active_agreement",
        "tempo_embedding_mean_active_probability": "tempo_mean_active_probability",
        "tempo_embedding_active_fraction": "tempo_active_fraction",
        "tempo_embedding_active_windows": "tempo_active_windows",
        "tempo_embedding_total_windows": "tempo_total_windows",
    }

    lookup: dict[str, dict[str, float]] = {}
    for i, filename in enumerate(filenames):
        key = Path(filename).name.lower()
        row: dict[str, float] = {}
        for output_name, npz_name in fields.items():
            if npz_name in data.files and i < len(data[npz_name]):
                row[output_name] = _safe_feature(data[npz_name][i])
        lookup[key] = row
    return lookup


def _compute_tempo_embedding_features(
    audio_path: Path,
    *,
    model_file: str | Path | None,
    auto_download_model: bool,
) -> dict[str, float]:
    from djprojectexploration.tempo_embedding import generate_tempo_embedding

    payload = generate_tempo_embedding(
        audio_file=audio_path,
        model_file=model_file,
        auto_download_model=auto_download_model,
    )
    return {
        "tempo_embedding_bpm": _safe_feature(payload.get("tempo_bpm")),
        "tempo_embedding_confidence": _safe_feature(payload.get("confidence")),
        "tempo_embedding_confidence_active_agreement": _safe_feature(
            payload.get("confidence_active_agreement")
        ),
        "tempo_embedding_mean_active_probability": _safe_feature(payload.get("mean_prob_active")),
        "tempo_embedding_active_fraction": _safe_feature(payload.get("active_fraction")),
        "tempo_embedding_active_windows": _safe_feature(payload.get("active_windows")),
        "tempo_embedding_total_windows": _safe_feature(payload.get("total_windows")),
    }


def _band_mask(freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    return (freqs >= float(low_hz)) & (freqs < float(high_hz))


def _db_from_power(power: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(power, EPS))


def _mean_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _std_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else float("nan")


def _var_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.var(arr)) if arr.size else float("nan")


def _spectral_tilt_db_per_oct(freqs: np.ndarray, mean_power: np.ndarray) -> float:
    valid = (freqs >= 30.0) & (freqs <= 12000.0) & np.isfinite(mean_power) & (mean_power > 0.0)
    if int(np.sum(valid)) < 3:
        return float("nan")
    x = np.log2(freqs[valid])
    y = _db_from_power(mean_power[valid])
    slope, _intercept = np.polyfit(x, y, deg=1)
    return float(slope)


def _chunk_slices(total_samples: int, sr: int, section_sec: float) -> list[slice]:
    chunk = max(1, int(round(float(section_sec) * sr)))
    slices = [slice(start, min(total_samples, start + chunk)) for start in range(0, total_samples, chunk)]
    return [s for s in slices if s.stop - s.start > max(1, sr)]


def _loudness_lufs(y: np.ndarray, sr: int) -> float:
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(int(sr))
        return _safe_feature(meter.integrated_loudness(np.asarray(y, dtype=np.float64)))
    except Exception:
        rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)) + EPS))
        return float(20.0 * np.log10(max(rms, EPS)))


def _audio_duration_sec(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate > 0 and info.frames > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return float("nan")


def _section_features(
    y: np.ndarray,
    *,
    sr: int,
    hop_length: int,
    section_sec: float,
) -> dict[str, float]:
    slices = _chunk_slices(int(y.size), int(sr), float(section_sec))
    if not slices:
        slices = [slice(0, int(y.size))]

    section_loudness: list[float] = []
    section_rms_db: list[float] = []
    section_onset_density: list[float] = []
    section_centroid: list[float] = []

    for chunk in slices:
        yc = y[chunk]
        duration = float(yc.size) / float(sr)
        if duration <= 0:
            continue

        loudness = _loudness_lufs(yc, sr)
        rms = float(np.sqrt(np.mean(np.square(yc, dtype=np.float64)) + EPS))
        rms_db = float(20.0 * np.log10(max(rms, EPS)))
        onset_env = librosa.onset.onset_strength(y=yc, sr=sr, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        centroid = librosa.feature.spectral_centroid(y=yc, sr=sr, hop_length=hop_length)

        section_loudness.append(loudness)
        section_rms_db.append(rms_db)
        section_onset_density.append(float(len(onsets)) / duration)
        section_centroid.append(_mean_or_nan(centroid))

    loudness_arr = np.asarray(section_loudness, dtype=np.float64)
    rms_arr = np.asarray(section_rms_db, dtype=np.float64)
    onset_arr = np.asarray(section_onset_density, dtype=np.float64)
    centroid_arr = np.asarray(section_centroid, dtype=np.float64)

    if loudness_arr.size and np.any(np.isfinite(loudness_arr)):
        peak_idx = int(np.nanargmax(loudness_arr))
    elif rms_arr.size and np.any(np.isfinite(rms_arr)):
        peak_idx = int(np.nanargmax(rms_arr))
    else:
        peak_idx = 0

    median_loudness = float(np.nanmedian(loudness_arr)) if loudness_arr.size else float("nan")
    median_rms = float(np.nanmedian(rms_arr)) if rms_arr.size else float("nan")
    median_centroid = float(np.nanmedian(centroid_arr)) if centroid_arr.size else float("nan")

    peak_loudness = _safe_feature(loudness_arr[peak_idx]) if loudness_arr.size else float("nan")
    peak_rms = _safe_feature(rms_arr[peak_idx]) if rms_arr.size else float("nan")
    peak_centroid = _safe_feature(centroid_arr[peak_idx]) if centroid_arr.size else float("nan")

    return {
        "max_30s_loudness_lufs": _mean_or_nan(np.array([np.nanmax(loudness_arr)])),
        "max_30s_onset_density": _mean_or_nan(np.array([np.nanmax(onset_arr)])),
        "peak_section_spectral_centroid_hz": peak_centroid,
        "peak_section_loudness_lufs": peak_loudness,
        "median_section_loudness_lufs": median_loudness,
        "peak_section_loudness_minus_median_lu": peak_loudness - median_loudness,
        "peak_section_rms_db": peak_rms,
        "median_section_rms_db": median_rms,
        "peak_section_rms_minus_median_db": peak_rms - median_rms,
        "peak_section_centroid_minus_median_hz": peak_centroid - median_centroid,
        "section_count_30s": float(len(slices)),
    }


def _beat_features(
    y: np.ndarray,
    *,
    sr: int,
    hop_length: int,
    frame_length: int,
    tempo_bpm: float | None,
) -> dict[str, float]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    beat_kwargs: dict[str, Any] = {
        "onset_envelope": onset_env,
        "sr": sr,
        "hop_length": hop_length,
        "trim": False,
    }
    if tempo_bpm is not None and np.isfinite(tempo_bpm) and tempo_bpm > 0:
        beat_kwargs["bpm"] = float(tempo_bpm)

    try:
        _tempo, beat_frames = librosa.beat.beat_track(**beat_kwargs)
    except Exception:
        beat_frames = np.array([], dtype=int)

    if beat_frames.size < 4:
        return {
            "beat_count": float(beat_frames.size),
            "beat_rms_mean_db": float("nan"),
            "beat_rms_std_db": float("nan"),
            "downbeat_offbeat_energy_ratio": float("nan"),
            "downbeat_phase_index": float("nan"),
            "four_bar_phrase_energy_variance": float("nan"),
        }

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    beat_frames = beat_frames[beat_frames < rms.size]
    if beat_frames.size < 4:
        return {
            "beat_count": float(beat_frames.size),
            "beat_rms_mean_db": float("nan"),
            "beat_rms_std_db": float("nan"),
            "downbeat_offbeat_energy_ratio": float("nan"),
            "downbeat_phase_index": float("nan"),
            "four_bar_phrase_energy_variance": float("nan"),
        }

    beat_rms = rms[beat_frames]
    beat_rms_db = 20.0 * np.log10(np.maximum(beat_rms, EPS))

    phase_means = []
    for phase in range(4):
        values = beat_rms[phase::4]
        phase_means.append(float(np.mean(values)) if values.size else float("nan"))
    if np.any(np.isfinite(phase_means)):
        downbeat_phase = int(np.nanargmax(np.asarray(phase_means, dtype=np.float64)))
        downbeat_values = beat_rms[downbeat_phase::4]
        offbeat_mask = np.ones(beat_rms.size, dtype=bool)
        offbeat_mask[downbeat_phase::4] = False
        offbeat_values = beat_rms[offbeat_mask]
        ratio = float(np.mean(downbeat_values) / max(float(np.mean(offbeat_values)), EPS))
    else:
        downbeat_phase = -1
        ratio = float("nan")

    phrase_len = 16
    phrase_means = [
        float(np.mean(beat_rms[i : i + phrase_len]))
        for i in range(0, beat_rms.size - phrase_len + 1, phrase_len)
    ]

    return {
        "beat_count": float(beat_frames.size),
        "beat_rms_mean_db": _mean_or_nan(beat_rms_db),
        "beat_rms_std_db": _std_or_nan(beat_rms_db),
        "downbeat_offbeat_energy_ratio": ratio,
        "downbeat_phase_index": float(downbeat_phase),
        "four_bar_phrase_energy_variance": _var_or_nan(np.asarray(phrase_means, dtype=np.float64)),
    }


def extract_audio_energy_features(
    audio_path: str | Path,
    *,
    sample_rate: int = 22050,
    analysis_seconds: float | None = None,
    tempo_bpm: float | None = None,
    section_sec: float = 30.0,
) -> dict[str, float]:
    """Extract deterministic audio features for perceived energy modeling."""
    path = Path(audio_path).expanduser().resolve()
    y, sr = librosa.load(
        path,
        sr=int(sample_rate),
        mono=True,
        duration=None if analysis_seconds is None else float(analysis_seconds),
    )
    y = np.asarray(y, dtype=np.float32)
    analysis_duration = float(y.size) / float(sr) if sr > 0 else 0.0
    if y.size == 0 or analysis_duration <= 0:
        raise RuntimeError(f"Loaded empty audio for {path}")
    source_duration = _audio_duration_sec(path)

    n_fft = 4096
    hop_length = 512
    frame_length = 2048

    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    power = np.square(stft, dtype=np.float64)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mean_power = np.mean(power, axis=1)
    total_power = float(np.sum(mean_power) + EPS)

    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
    rolloff_05 = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.05)
    rolloff_85 = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = 20.0 * np.log10(np.maximum(rms, EPS))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    onset_density = float(len(onsets)) / max(analysis_duration, EPS)
    onset_magnitude = _mean_or_nan(onset_env[onsets]) if len(onsets) else 0.0

    features: dict[str, float] = {
        "duration_sec": source_duration,
        "analysis_duration_sec": analysis_duration,
        "spectral_centroid_hz": _mean_or_nan(centroid),
        "spectral_spread_hz": _mean_or_nan(bandwidth),
        "spectral_tilt_db_per_oct": _spectral_tilt_db_per_oct(freqs, mean_power),
        "spectral_rolloff_05_hz": _mean_or_nan(rolloff_05),
        "spectral_rolloff_85_hz": _mean_or_nan(rolloff_85),
        "dynamic_range_db": float(np.nanpercentile(rms_db, 95) - np.nanpercentile(rms_db, 10)),
        "rms_mean_db": _mean_or_nan(rms_db),
        "rms_median_db": float(np.nanmedian(rms_db)),
        "rms_std_db": _std_or_nan(rms_db),
        "rms_iqr_db": float(np.nanpercentile(rms_db, 75) - np.nanpercentile(rms_db, 25)),
        "top_quartile_rms": float(np.mean(rms[rms >= np.nanpercentile(rms, 75)])),
        "top_quartile_rms_db": float(
            20.0 * np.log10(max(float(np.mean(rms[rms >= np.nanpercentile(rms, 75)])), EPS))
        ),
        "crest_factor_db": float(
            20.0 * np.log10(max(float(np.max(np.abs(y))), EPS))
            - 20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(y, dtype=np.float64)) + EPS)), EPS))
        ),
        "onset_density": onset_density,
        "onset_magnitude": onset_magnitude,
        "integrated_loudness_lufs": _loudness_lufs(y, sr),
    }

    for band_name, (low_hz, high_hz) in SPECTRAL_BANDS_HZ.items():
        mask = _band_mask(freqs, low_hz, high_hz)
        band_power = np.sum(mean_power[mask]) if np.any(mask) else float("nan")
        features[f"{band_name}_band_energy_ratio"] = float(band_power / total_power)

        band_by_frame = np.sum(power[mask, :], axis=0) if np.any(mask) else np.array([], dtype=np.float64)
        if band_by_frame.size > 1:
            flux = np.maximum(np.diff(np.sqrt(np.maximum(band_by_frame, 0.0))), 0.0)
            features[f"band_flux_{band_name}"] = _mean_or_nan(flux)
        else:
            features[f"band_flux_{band_name}"] = float("nan")

    low_mask = _band_mask(freqs, 20.0, 120.0)
    features["low_band_energy_ratio"] = float(np.sum(mean_power[low_mask]) / total_power)

    features.update(
        _beat_features(
            y,
            sr=sr,
            hop_length=hop_length,
            frame_length=frame_length,
            tempo_bpm=tempo_bpm,
        )
    )
    features.update(
        _section_features(
            y,
            sr=sr,
            hop_length=hop_length,
            section_sec=section_sec,
        )
    )
    return features


def extract_energy_feature_rows(
    tracks: list[EnergyTrack],
    *,
    tempo_lookup: dict[str, dict[str, float]] | None = None,
    compute_missing_tempo: bool = False,
    tempo_model_file: str | Path | None = None,
    auto_download_tempo_model: bool = False,
    sample_rate: int = 22050,
    analysis_seconds: float | None = None,
    section_sec: float = 30.0,
) -> list[dict[str, Any]]:
    """Extract feature rows for a list of normalized tracks."""
    tempo_lookup = tempo_lookup or {}
    rows: list[dict[str, Any]] = []
    total = len(tracks)
    for index, track in enumerate(tracks, start=1):
        print(f"[{index}/{total}] Extracting energy features: {track.title}")
        output_row: dict[str, Any] = dict(track.row)
        output_row.update(
            {
                "track_number": track.track_number,
                "title": track.title,
                "artists": track.artists,
                "filename": track.filename,
                "audio_path": _project_relpath(track.audio_path),
                "genre": track.genre,
                "labeled_bpm": "" if track.labeled_bpm is None else track.labeled_bpm,
                "labeled_bpm_missing": bool(track.labeled_bpm is None or track.labeled_bpm <= 0),
                "energy": "" if track.energy is None else track.energy,
            }
        )

        tempo_features = dict(tempo_lookup.get(Path(track.filename).name.lower(), {}))
        if not tempo_features:
            tempo_features = dict(tempo_lookup.get(track.audio_path.name.lower(), {}))
        if not tempo_features and compute_missing_tempo:
            tempo_features = _compute_tempo_embedding_features(
                track.audio_path,
                model_file=tempo_model_file,
                auto_download_model=auto_download_tempo_model,
            )
        output_row.update(tempo_features)

        tempo_bpm = _optional_float(output_row.get("tempo_embedding_bpm"))
        audio_features = extract_audio_energy_features(
            track.audio_path,
            sample_rate=sample_rate,
            analysis_seconds=analysis_seconds,
            tempo_bpm=tempo_bpm,
            section_sec=section_sec,
        )
        output_row.update(audio_features)
        rows.append(output_row)
    return rows


def write_feature_csv(rows: list[dict[str, Any]], output_file: str | Path) -> Path:
    """Write rows to CSV using a stable union of encountered columns."""
    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred_prefix = [
        "track_number",
        "title",
        "name",
        "artists",
        "artist",
        "album",
        "genre",
        "filename",
        "mp3_name",
        "filepath",
        "audio_path",
        "energy",
        "labeled_bpm",
        "labeled_bpm_missing",
        "tempo_embedding_bpm",
        "tempo_embedding_confidence",
    ]
    for field in preferred_prefix:
        if any(field in row for row in rows) and field not in seen:
            fieldnames.append(field)
            seen.add(field)
    for row in rows:
        for field in row.keys():
            if field not in seen:
                fieldnames.append(field)
                seen.add(field)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})
    return output_path


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    mix = str(row.get("mix_name") or row.get("mix_slug") or row.get("source") or "").strip().lower()
    filename = str(row.get("mp3_name") or row.get("filename") or row.get("filepath") or row.get("audio_path") or "").strip()
    return mix, Path(filename).name.lower()


def _mix_name_from_path(path: Path) -> str:
    stem = path.stem
    for suffix in ("_energy_features", "_tracks"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.replace("_", "-")


def _read_feature_csvs(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value).expanduser().resolve()
        mix_name = _mix_name_from_path(path)
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                out = dict(row)
                out.setdefault("mix_name", mix_name)
                rows.append(out)
    return rows


def _energy_value(row: dict[str, Any]) -> float:
    return _safe_feature(row.get("energy"))


def _feature_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [[_safe_feature(row.get(feature)) for feature in feature_names] for row in rows],
        dtype=np.float64,
    )


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum(np.square(y - pred)))
    ss_tot = float(np.sum(np.square(y - float(np.mean(y)))))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fit_full_energy_model(rows: list[dict[str, Any]], *, min_valid_features: int = 4) -> EnergyModelResult:
    """Fit the replaceable full energy model from labeled feature rows."""
    y_all = np.asarray([_energy_value(row) for row in rows], dtype=np.float64)
    X_all = _feature_matrix(rows, FULL_ENERGY_FEATURE_KEYS)
    train_mask = np.isfinite(y_all) & np.any(np.isfinite(X_all), axis=1)
    if int(np.sum(train_mask)) < 6:
        nan_pred = np.full(len(rows), np.nan, dtype=np.float64)
        return EnergyModelResult(
            model_name=FULL_ENERGY_MODEL_NAME,
            feature_names=[],
            predictions=nan_pred,
            oof_predictions=nan_pred.copy(),
            best_alpha=float("nan"),
            baseline_mae=float("nan"),
            oof_mae=float("nan"),
            oof_r2=float("nan"),
            train_mae=float("nan"),
            train_r2=float("nan"),
            available=False,
        )

    valid_feature_mask = np.sum(np.isfinite(X_all[train_mask]), axis=0) >= int(min_valid_features)
    feature_names = [name for name, keep in zip(FULL_ENERGY_FEATURE_KEYS, valid_feature_mask) if keep]
    if not feature_names:
        nan_pred = np.full(len(rows), np.nan, dtype=np.float64)
        return EnergyModelResult(
            model_name=FULL_ENERGY_MODEL_NAME,
            feature_names=[],
            predictions=nan_pred,
            oof_predictions=nan_pred.copy(),
            best_alpha=float("nan"),
            baseline_mae=float("nan"),
            oof_mae=float("nan"),
            oof_r2=float("nan"),
            train_mae=float("nan"),
            train_r2=float("nan"),
            available=False,
        )

    X = X_all[:, valid_feature_mask]
    X_train = X[train_mask]
    y_train = y_all[train_mask]

    try:
        from sklearn.base import clone
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import TweedieRegressor
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import KFold, RepeatedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        import warnings
    except ImportError as exc:
        raise ImportError("The full energy model requires scikit-learn. It is normally installed via umap-learn.") from exc

    base_model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        TweedieRegressor(
            power=0,
            link="identity",
            alpha=1.0,
            fit_intercept=True,
            max_iter=10000,
            tol=1e-7,
        ),
    )
    alpha_grid = np.logspace(-3, 2, 26)
    n_splits = min(5, int(y_train.size))
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=20, random_state=42)
    alpha_scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for alpha in alpha_grid:
            fold_mae = []
            for train_idx, test_idx in cv.split(X_train, y_train):
                model = clone(base_model)
                model.set_params(tweedieregressor__alpha=float(alpha))
                model.fit(X_train[train_idx], y_train[train_idx])
                pred = model.predict(X_train[test_idx])
                fold_mae.append(mean_absolute_error(y_train[test_idx], pred))
            alpha_scores.append(float(np.mean(fold_mae)))
    best_alpha = float(alpha_grid[int(np.argmin(np.asarray(alpha_scores, dtype=np.float64)))])

    oof_train = np.full(y_train.shape, np.nan, dtype=np.float64)
    diagnostic_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for train_idx, test_idx in diagnostic_cv.split(X_train, y_train):
            model = clone(base_model)
            model.set_params(tweedieregressor__alpha=best_alpha)
            model.fit(X_train[train_idx], y_train[train_idx])
            oof_train[test_idx] = model.predict(X_train[test_idx])

    final_model = clone(base_model)
    final_model.set_params(tweedieregressor__alpha=best_alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        final_model.fit(X_train, y_train)
    train_pred = final_model.predict(X_train)
    all_pred = final_model.predict(X)

    oof_all = np.full(len(rows), np.nan, dtype=np.float64)
    train_positions = np.flatnonzero(train_mask)
    oof_all[train_positions] = oof_train
    baseline = np.full_like(y_train, float(np.mean(y_train)), dtype=np.float64)
    return EnergyModelResult(
        model_name=FULL_ENERGY_MODEL_NAME,
        feature_names=feature_names,
        predictions=np.asarray(all_pred, dtype=np.float64),
        oof_predictions=oof_all,
        best_alpha=best_alpha,
        baseline_mae=float(mean_absolute_error(y_train, baseline)),
        oof_mae=float(mean_absolute_error(y_train, oof_train)),
        oof_r2=_r2_score(y_train, oof_train),
        train_mae=float(mean_absolute_error(y_train, train_pred)),
        train_r2=_r2_score(y_train, train_pred),
        available=True,
        estimator=final_model,
    )


def default_energy_model_path(
    output_dir: str | Path = DEFAULT_ENERGY_MODEL_DIR,
    *,
    name: str = FULL_ENERGY_MODEL_NAME,
) -> Path:
    return Path(output_dir).expanduser().resolve() / f"{name}.joblib"


def _model_metadata(
    result: EnergyModelResult,
    *,
    training_rows: int,
    labeled_rows: int,
    target_min: float,
    target_max: float,
) -> dict[str, Any]:
    return {
        "model_name": result.model_name,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "feature_names": list(result.feature_names),
        "training_rows": int(training_rows),
        "labeled_rows": int(labeled_rows),
        "target_min": float(target_min),
        "target_max": float(target_max),
        "best_alpha": float(result.best_alpha),
        "baseline_mae": float(result.baseline_mae),
        "oof_mae": float(result.oof_mae),
        "oof_r2": float(result.oof_r2),
        "train_mae": float(result.train_mae),
        "train_r2": float(result.train_r2),
    }


def save_energy_model_artifact(
    result: EnergyModelResult,
    output_file: str | Path,
    *,
    training_rows: int,
    labeled_rows: int,
    target_min: float,
    target_max: float,
    metadata_file: str | Path | None = None,
) -> Path:
    """Persist a fitted energy model pipeline and sidecar metadata."""
    if not result.available or result.estimator is None:
        raise RuntimeError("Cannot save an unavailable energy model.")
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Saving energy models requires joblib.") from exc

    output_path = Path(output_file).expanduser().resolve()
    metadata_path = (
        Path(metadata_file).expanduser().resolve()
        if metadata_file is not None
        else output_path.with_suffix(".json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = _model_metadata(
        result,
        training_rows=training_rows,
        labeled_rows=labeled_rows,
        target_min=target_min,
        target_max=target_max,
    )
    artifact = {
        "schema_version": 1,
        "metadata": metadata,
        "estimator": result.estimator,
        "feature_names": list(result.feature_names),
        "model_name": result.model_name,
    }
    joblib.dump(artifact, output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved energy model: {_project_relpath(output_path)}")
    print(f"Saved energy model metadata: {_project_relpath(metadata_path)}")
    return output_path


def load_energy_model_artifact(model_file: str | Path) -> dict[str, Any]:
    """Load a frozen energy model artifact created by save_energy_model_artifact."""
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Loading energy models requires joblib.") from exc

    path = Path(model_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Energy model artifact not found: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError(f"Energy model artifact has unexpected format: {path}")
    if "estimator" not in artifact or "feature_names" not in artifact:
        raise ValueError(f"Energy model artifact is missing estimator/feature_names: {path}")
    return artifact


def train_energy_model_from_feature_csvs(
    feature_csvs: list[str | Path],
    *,
    output_file: str | Path | None = None,
    metadata_file: str | Path | None = None,
    name: str = FULL_ENERGY_MODEL_NAME,
    output_dir: str | Path = DEFAULT_ENERGY_MODEL_DIR,
    target_min: float = DEFAULT_ENERGY_TARGET_MIN,
    target_max: float = DEFAULT_ENERGY_TARGET_MAX,
) -> Path:
    """Train and freeze a reusable full energy model from labeled feature CSVs."""
    rows = _read_feature_csvs(feature_csvs)
    if not rows:
        raise RuntimeError("No rows found in energy feature CSV inputs.")
    result = fit_full_energy_model(rows)
    y_all = np.asarray([_energy_value(row) for row in rows], dtype=np.float64)
    labeled_values = y_all[np.isfinite(y_all)]
    labeled_rows = int(labeled_values.size)
    if not result.available:
        raise RuntimeError(
            "Energy model could not be trained. Provide feature CSVs with at least 6 labeled `energy` rows."
        )
    model_path = (
        Path(output_file).expanduser().resolve()
        if output_file is not None
        else default_energy_model_path(output_dir, name=name)
    )
    save_energy_model_artifact(
        result,
        model_path,
        training_rows=len(rows),
        labeled_rows=labeled_rows,
        target_min=float(target_min),
        target_max=float(target_max),
        metadata_file=metadata_file,
    )
    print(f"Training rows: {len(rows)}")
    print(f"Labeled rows: {labeled_rows}")
    print(f"Features retained: {len(result.feature_names)}")
    print(f"Best alpha: {result.best_alpha:.4g}")
    print(f"OOF MAE: {result.oof_mae:.3f}; train MAE: {result.train_mae:.3f}")
    return model_path


def predict_with_energy_model_artifact(rows: list[dict[str, Any]], model_file: str | Path) -> EnergyModelResult:
    """Apply a frozen energy model artifact to feature rows."""
    artifact = load_energy_model_artifact(model_file)
    feature_names = [str(name) for name in artifact["feature_names"]]
    estimator = artifact["estimator"]
    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
    X = _feature_matrix(rows, feature_names)
    predictions = np.asarray(estimator.predict(X), dtype=np.float64)
    target_min = _safe_feature(metadata.get("target_min"))
    target_max = _safe_feature(metadata.get("target_max"))
    if np.isfinite(target_min) and np.isfinite(target_max) and target_max >= target_min:
        predictions = np.clip(predictions, target_min, target_max)
    oof_predictions = np.full(len(rows), np.nan, dtype=np.float64)
    return EnergyModelResult(
        model_name=str(artifact.get("model_name") or metadata.get("model_name") or FULL_ENERGY_MODEL_NAME),
        feature_names=feature_names,
        predictions=predictions,
        oof_predictions=oof_predictions,
        best_alpha=_safe_feature(metadata.get("best_alpha")),
        baseline_mae=_safe_feature(metadata.get("baseline_mae")),
        oof_mae=_safe_feature(metadata.get("oof_mae")),
        oof_r2=_safe_feature(metadata.get("oof_r2")),
        train_mae=_safe_feature(metadata.get("train_mae")),
        train_r2=_safe_feature(metadata.get("train_r2")),
        available=True,
        estimator=estimator,
    )


def default_energy_npz_path(output_dir: str | Path = DEFAULT_ENERGY_EMBEDDING_DIR, *, name: str = "energy_features") -> Path:
    return Path(output_dir).expanduser().resolve() / f"{name}.npz"


def create_energy_embedding_npz(
    feature_csvs: list[str | Path],
    *,
    output_file: str | Path | None = None,
    manifest_file: str | Path | None = None,
    name: str = "energy_features",
    output_dir: str | Path = DEFAULT_ENERGY_EMBEDDING_DIR,
    model: str = "full",
    model_file: str | Path | None = None,
) -> Path:
    """Create the canonical app-consumable energy NPZ from energy feature CSV rows."""
    rows = _read_feature_csvs(feature_csvs)
    if not rows:
        raise RuntimeError("No rows found in energy feature CSV inputs.")
    if model != "full":
        raise ValueError("Only model='full' is currently implemented.")

    model_result = (
        predict_with_energy_model_artifact(rows, model_file)
        if model_file is not None
        else fit_full_energy_model(rows)
    )
    feature_names = list(model_result.feature_names or FULL_ENERGY_FEATURE_KEYS)
    embeddings = _feature_matrix(rows, feature_names).astype(np.float32)

    output_path = Path(output_file).expanduser().resolve() if output_file is not None else default_energy_npz_path(output_dir, name=name)
    manifest_path = (
        Path(manifest_file).expanduser().resolve()
        if manifest_file is not None
        else output_path.with_name(f"{output_path.stem}_manifest.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    energies = np.asarray([_energy_value(row) for row in rows], dtype=np.float32)
    predictions = np.asarray(model_result.predictions, dtype=np.float32)
    oof_predictions = np.asarray(model_result.oof_predictions, dtype=np.float32)
    residual = energies.astype(np.float64) - predictions.astype(np.float64)
    oof_residual = energies.astype(np.float64) - oof_predictions.astype(np.float64)
    created_utc = datetime.now(tz=timezone.utc).isoformat()

    np.savez_compressed(
        output_path,
        embedding_type=np.array("energy_features", dtype=np.str_),
        created_utc=np.array(created_utc, dtype=np.str_),
        model_name=np.array(model_result.model_name, dtype=np.str_),
        num_tracks=np.array(len(rows), dtype=np.int32),
        embedding_dimension=np.array(int(embeddings.shape[1]), dtype=np.int32),
        embeddings=embeddings,
        feature_names=np.asarray(feature_names, dtype=np.str_),
        mix_name=np.asarray([str(row.get("mix_name", "")) for row in rows], dtype=np.str_),
        track_number=np.asarray([str(row.get("track_number", "")) for row in rows], dtype=np.str_),
        title=np.asarray([str(row.get("title") or row.get("name") or "") for row in rows], dtype=np.str_),
        artists=np.asarray([str(row.get("artists") or row.get("artist") or "") for row in rows], dtype=np.str_),
        genre=np.asarray([str(row.get("genre") or "") for row in rows], dtype=np.str_),
        mp3_name=np.asarray([str(row.get("mp3_name") or row.get("filename") or "") for row in rows], dtype=np.str_),
        energy=energies,
        bpm=np.asarray([_safe_feature(row.get("bpm") or row.get("labeled_bpm") or row.get("tempo_embedding_bpm")) for row in rows], dtype=np.float32),
        glm_available=np.array(bool(model_result.available), dtype=np.bool_),
        glm_best_alpha=np.array(float(model_result.best_alpha), dtype=np.float32),
        glm_baseline_mae=np.array(float(model_result.baseline_mae), dtype=np.float32),
        glm_oof_mae=np.array(float(model_result.oof_mae), dtype=np.float32),
        glm_oof_r2=np.array(float(model_result.oof_r2), dtype=np.float32),
        glm_train_mae=np.array(float(model_result.train_mae), dtype=np.float32),
        glm_train_r2=np.array(float(model_result.train_r2), dtype=np.float32),
        glm_energy_pred=predictions,
        glm_energy_oof_pred=oof_predictions,
        glm_energy_residual=np.asarray(residual, dtype=np.float32),
        glm_energy_oof_residual=np.asarray(oof_residual, dtype=np.float32),
    )

    manifest_fields = list(ENERGY_NPZ_FIELDNAMES)
    for feature in feature_names:
        if feature not in manifest_fields:
            manifest_fields.append(feature)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = dict(row)
            out["glm_energy_pred"] = predictions[i]
            out["glm_energy_oof_pred"] = oof_predictions[i]
            out["glm_energy_residual"] = residual[i]
            out["glm_energy_oof_residual"] = oof_residual[i]
            writer.writerow({field: _csv_value(out.get(field, "")) for field in manifest_fields})

    print(f"Saved energy feature NPZ: {_project_relpath(output_path)}")
    print(f"Saved energy feature manifest: {_project_relpath(manifest_path)}")
    print(f"Tracks: {len(rows)}")
    print(f"Features retained: {len(feature_names)}")
    print(f"Full energy model available: {bool(model_result.available)}")
    if model_file is not None:
        print(f"Energy model artifact: {_project_relpath(Path(model_file).expanduser().resolve())}")
    if model_result.available:
        print(f"Best alpha: {model_result.best_alpha:.4g}")
        print(f"OOF MAE: {model_result.oof_mae:.3f}; train MAE: {model_result.train_mae:.3f}")
    return output_path


def create_energy_feature_csv(
    input_csv: str | Path,
    *,
    output_file: str | Path | None = None,
    music_dir: str | Path | None = None,
    tempo_embeddings: str | Path | None = None,
    compute_missing_tempo: bool = False,
    tempo_model_file: str | Path | None = None,
    auto_download_tempo_model: bool = False,
    skip_missing_audio: bool = False,
    sample_rate: int = 22050,
    analysis_seconds: float | None = None,
    section_sec: float = 30.0,
    max_tracks: int | None = None,
) -> Path:
    """Create an enriched energy-feature CSV from a labeled track CSV."""
    tracks = load_energy_tracks(
        input_csv,
        music_dir=music_dir,
        skip_missing_audio=skip_missing_audio,
    )
    if max_tracks is not None:
        tracks = tracks[: int(max_tracks)]

    tempo_file = Path(tempo_embeddings).expanduser().resolve() if tempo_embeddings is not None else None
    if tempo_file is None:
        tempo_file = default_tempo_embeddings_file(input_csv)

    tempo_lookup = load_tempo_embedding_lookup(tempo_file) if tempo_file is not None else {}
    rows = extract_energy_feature_rows(
        tracks,
        tempo_lookup=tempo_lookup,
        compute_missing_tempo=compute_missing_tempo,
        tempo_model_file=tempo_model_file,
        auto_download_tempo_model=auto_download_tempo_model,
        sample_rate=sample_rate,
        analysis_seconds=analysis_seconds,
        section_sec=section_sec,
    )

    resolved_output = default_output_file(input_csv) if output_file is None else Path(output_file)
    saved_path = write_feature_csv(rows, resolved_output)
    print(f"Saved energy feature CSV: {_project_relpath(saved_path)}")
    print(f"Tracks: {len(rows)}")
    if tempo_file is not None:
        print(f"Tempo embeddings: {_project_relpath(tempo_file)}")
    elif not compute_missing_tempo:
        print("Tempo embeddings: none found; tempo_embedding_* fields were not added.")
    return saved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract energy-model features to CSV.")
    parser.add_argument("input_csv", type=Path, help="Labeled track CSV.")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--music-dir", type=Path, default=None)
    parser.add_argument("--tempo-embeddings", type=Path, default=None, help="Tempo playlist NPZ.")
    parser.add_argument("--compute-missing-tempo", action="store_true")
    parser.add_argument("--tempo-model-file", type=Path, default=None)
    parser.add_argument("--auto-download-tempo-model", action="store_true")
    parser.add_argument("--skip-missing-audio", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument(
        "--analysis-seconds",
        type=float,
        default=None,
        help="Optional seconds from the start of each track to analyze.",
    )
    parser.add_argument("--section-sec", type=float, default=30.0)
    parser.add_argument("--max-tracks", type=int, default=None)
    return parser.parse_args()


def parse_npz_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create canonical energy NPZ from energy feature CSV files.")
    parser.add_argument(
        "feature_csv",
        nargs="+",
        type=Path,
        help="Energy feature CSV(s), such as data/energy_features/<stem>_energy_features.csv.",
    )
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--manifest-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ENERGY_EMBEDDING_DIR)
    parser.add_argument("--name", default="energy_features", help="Default output stem when --output-file is omitted.")
    parser.add_argument("--model", default="full", choices=["full"], help="Energy model spec to use.")
    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="Frozen energy model artifact to apply instead of fitting from the input rows.",
    )
    return parser.parse_args(argv)


def parse_train_model_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and freeze an energy model from labeled energy feature CSVs.")
    parser.add_argument(
        "feature_csv",
        nargs="+",
        type=Path,
        help="Labeled energy feature CSV(s). Use djprojectexploration-energy-features first if needed.",
    )
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--metadata-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ENERGY_MODEL_DIR)
    parser.add_argument("--name", default=FULL_ENERGY_MODEL_NAME)
    parser.add_argument("--target-min", type=float, default=DEFAULT_ENERGY_TARGET_MIN)
    parser.add_argument("--target-max", type=float, default=DEFAULT_ENERGY_TARGET_MAX)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    create_energy_feature_csv(
        args.input_csv,
        output_file=args.output_file,
        music_dir=args.music_dir,
        tempo_embeddings=args.tempo_embeddings,
        compute_missing_tempo=bool(args.compute_missing_tempo),
        tempo_model_file=args.tempo_model_file,
        auto_download_tempo_model=bool(args.auto_download_tempo_model),
        skip_missing_audio=bool(args.skip_missing_audio),
        sample_rate=int(args.sample_rate),
        analysis_seconds=args.analysis_seconds,
        section_sec=float(args.section_sec),
        max_tracks=args.max_tracks,
    )


def energy_npz_main(argv: list[str] | None = None) -> int:
    args = parse_npz_args(argv)
    create_energy_embedding_npz(
        [Path(path) for path in args.feature_csv],
        output_file=args.output_file,
        manifest_file=args.manifest_file,
        name=str(args.name),
        output_dir=args.output_dir,
        model=str(args.model),
        model_file=args.model_file,
    )
    return 0


def energy_train_model_main(argv: list[str] | None = None) -> int:
    args = parse_train_model_args(argv)
    train_energy_model_from_feature_csvs(
        [Path(path) for path in args.feature_csv],
        output_file=args.output_file,
        metadata_file=args.metadata_file,
        name=str(args.name),
        output_dir=args.output_dir,
        target_min=float(args.target_min),
        target_max=float(args.target_max),
    )
    return 0


if __name__ == "__main__":
    main()
