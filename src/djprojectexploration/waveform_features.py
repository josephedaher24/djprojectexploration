"""Precompute multi-resolution waveform summaries for playlist tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from djprojectexploration.audio_snippets import (
    DEFAULT_MIDDLE_FRACTION,
    DEFAULT_SCAN_HOP_SECONDS,
    DEFAULT_SNIPPET_SECONDS,
    select_rms_focused_window,
)
from djprojectexploration.playlist_embedding_pipeline import (
    _default_npz_name,
    _metadata_arrays,
)
from djprojectexploration.tracklists import (
    PROJECT_ROOT,
    PlaylistTrack,
    load_playlist_tracks,
    to_project_relpath as _to_project_relpath,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "waveform_features"
DEFAULT_SAMPLE_RATE = 22_050
DEFAULT_ROW_BINS = 512
DEFAULT_PREVIEW_BINS = 4096
DEFAULT_DETAIL_BINS = 32_768
DEFAULT_BAND_BINS = 8192
WAVEFORM_FEATURE_VERSION = 3


def _string_array(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=np.str_)


def _safe_stem(value: str) -> str:
    stem = Path(value).stem or "track"
    cleaned = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in stem).strip(" ._-")
    return cleaned or "track"


def _json_asset_path(*, json_dir: Path, track: PlaylistTrack) -> Path:
    digest = hashlib.sha1(str(track.audio_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return json_dir / f"{_safe_stem(track.mp3_name)}_{digest}.json"


def _normalize(values: np.ndarray, *, floor: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    if arr.size == 0:
        return arr
    scale = float(np.percentile(arr, 99.5))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.max(arr))
    if not np.isfinite(scale) or scale <= 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr / scale, 0.0, 1.0)
    # Slight expansion keeps quieter sections visible without flattening transients.
    arr = np.sqrt(arr).astype(np.float32)
    if floor > 0.0:
        arr = np.where(arr > 0.0, np.maximum(arr, floor), 0.0)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _quantize(values: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    return np.rint(arr * 255.0).astype(np.uint8)


def _quantize_signed(values: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)
    return np.rint(arr * 127.0).astype(np.int8)


def _peak_profile(abs_audio: np.ndarray, bins: int) -> np.ndarray:
    bins = max(8, int(bins))
    values = np.asarray(abs_audio, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(bins, dtype=np.float32)
    edges = np.linspace(0, values.size, bins + 1, dtype=np.int64)
    profile = np.zeros(bins, dtype=np.float32)
    for i in range(bins):
        start = int(edges[i])
        end = int(edges[i + 1])
        if end <= start:
            end = min(values.size, start + 1)
        segment = values[start:end]
        profile[i] = float(np.max(segment)) if segment.size else 0.0
    return _normalize(profile, floor=0.025)


def _minmax_profile(audio: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    bins = max(8, int(bins))
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0:
        empty = np.zeros(bins, dtype=np.float32)
        return empty, empty

    edges = np.linspace(0, values.size, bins + 1, dtype=np.int64)
    low = np.zeros(bins, dtype=np.float32)
    high = np.zeros(bins, dtype=np.float32)
    for i in range(bins):
        start = int(edges[i])
        end = int(edges[i + 1])
        if end <= start:
            end = min(values.size, start + 1)
        segment = values[start:end]
        if segment.size:
            low[i] = float(np.min(segment))
            high[i] = float(np.max(segment))

    envelope = np.maximum(np.abs(low), np.abs(high))
    scale = float(np.percentile(envelope, 99.5)) if envelope.size else 0.0
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.max(envelope)) if envelope.size else 0.0
    if not np.isfinite(scale) or scale <= 1e-9:
        return np.zeros_like(low), np.zeros_like(high)
    return (
        np.clip(low / scale, -1.0, 1.0).astype(np.float32),
        np.clip(high / scale, -1.0, 1.0).astype(np.float32),
    )


def _rms_profile(audio: np.ndarray, bins: int) -> np.ndarray:
    bins = max(8, int(bins))
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(bins, dtype=np.float32)
    edges = np.linspace(0, values.size, bins + 1, dtype=np.int64)
    profile = np.zeros(bins, dtype=np.float32)
    for i in range(bins):
        start = int(edges[i])
        end = int(edges[i + 1])
        if end <= start:
            end = min(values.size, start + 1)
        segment = values[start:end]
        if segment.size:
            profile[i] = float(np.sqrt(np.mean(np.square(segment, dtype=np.float64))))
    return _normalize(profile, floor=0.025)


def _interp_profile(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    bins = max(8, int(bins))
    if values.size == 0:
        return np.zeros(bins, dtype=np.float32)
    if values.size == 1:
        return np.full(bins, float(values[0]), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, bins, dtype=np.float32)
    return np.interp(x_new, x_old, values).astype(np.float32)


def _band_profiles(audio: np.ndarray, *, sample_rate: int, bins: int) -> dict[str, np.ndarray]:
    import librosa

    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if y.size == 0:
        empty = np.zeros(max(8, int(bins)), dtype=np.float32)
        return {"low": empty, "mid": empty, "high": empty}

    n_fft = 2048
    hop_length = 512
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)).astype(np.float32)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    band_defs = {
        "low": (20.0, 250.0),
        "mid": (250.0, 4000.0),
        "high": (4000.0, float(sample_rate) * 0.5),
    }
    profiles: dict[str, np.ndarray] = {}
    for name, (lo, hi) in band_defs.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            env = np.zeros(stft.shape[1], dtype=np.float32)
        else:
            env = np.sqrt(np.mean(np.square(stft[mask, :], dtype=np.float64), axis=0)).astype(np.float32)
        profiles[name] = _normalize(_interp_profile(env, bins), floor=0.02)
    return profiles


def generate_waveform_feature_payload(
    audio_path: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    row_bins: int = DEFAULT_ROW_BINS,
    preview_bins: int = DEFAULT_PREVIEW_BINS,
    detail_bins: int = DEFAULT_DETAIL_BINS,
    band_bins: int = DEFAULT_BAND_BINS,
) -> dict[str, Any]:
    """Generate compact waveform summaries for a single audio file."""
    import librosa

    resolved = Path(audio_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved}")

    audio, sr = librosa.load(str(resolved), sr=int(sample_rate), mono=True)
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration_seconds = float(y.size / float(sr)) if sr > 0 else 0.0
    abs_y = np.abs(y)
    preview_start_idx, preview_end_idx, preview_rms = select_rms_focused_window(
        y,
        int(sr),
        snippet_seconds=DEFAULT_SNIPPET_SECONDS,
        middle_fraction=DEFAULT_MIDDLE_FRACTION,
        hop_seconds=DEFAULT_SCAN_HOP_SECONDS,
    )
    preview_start_seconds = preview_start_idx / float(sr) if sr > 0 else 0.0
    preview_end_seconds = preview_end_idx / float(sr) if sr > 0 else 0.0

    row = _rms_profile(y, row_bins)
    preview = _peak_profile(abs_y, preview_bins)
    detail = _peak_profile(abs_y, detail_bins)
    preview_min, preview_max = _minmax_profile(y, preview_bins)
    detail_min, detail_max = _minmax_profile(y, detail_bins)
    bands = _band_profiles(y, sample_rate=int(sr), bins=band_bins)

    return {
        "version": WAVEFORM_FEATURE_VERSION,
        "source_audio_path": str(resolved),
        "sample_rate": int(sr),
        "duration_seconds": duration_seconds,
        "format": "uint8_0_255",
        "signed_format": "int8_-127_127",
        "row_bins": int(row_bins),
        "preview_bins": int(preview_bins),
        "detail_bins": int(detail_bins),
        "band_bins": int(band_bins),
        "preview_start_seconds": float(preview_start_seconds),
        "preview_end_seconds": float(preview_end_seconds),
        "preview_duration_seconds": float(max(0.0, preview_end_seconds - preview_start_seconds)),
        "preview_score": float(preview_rms),
        "preview_method": "rms_middle_scan",
        "row_peaks": _quantize(row),
        "preview_peaks": _quantize(preview),
        "detail_peaks": _quantize(detail),
        "preview_min": _quantize_signed(preview_min),
        "preview_max": _quantize_signed(preview_max),
        "detail_min": _quantize_signed(detail_min),
        "detail_max": _quantize_signed(detail_max),
        "band_low": _quantize(bands["low"]),
        "band_mid": _quantize(bands["mid"]),
        "band_high": _quantize(bands["high"]),
    }


def _json_payload_for_track(track: PlaylistTrack, payload: dict[str, Any], *, track_id: str) -> dict[str, Any]:
    return {
        "version": int(payload["version"]),
        "track_id": track_id,
        "track_number": int(track.track_number),
        "title": track.title,
        "artists": track.artists,
        "filename": track.mp3_name,
        "source_audio_path": _to_project_relpath(Path(str(payload["source_audio_path"]))),
        "sample_rate": int(payload["sample_rate"]),
        "duration_seconds": float(payload["duration_seconds"]),
        "format": "uint8_0_255",
        "signed_format": "int8_-127_127",
        "row_bins": int(payload["row_bins"]),
        "preview_bins": int(payload["preview_bins"]),
        "detail_bins": int(payload["detail_bins"]),
        "band_bins": int(payload["band_bins"]),
        "preview_start_seconds": float(payload.get("preview_start_seconds", 0.0)),
        "preview_end_seconds": float(payload.get("preview_end_seconds", 0.0)),
        "preview_duration_seconds": float(payload.get("preview_duration_seconds", 0.0)),
        "preview_score": float(payload.get("preview_score", 0.0)),
        "preview_method": str(payload.get("preview_method") or "rms_middle_scan"),
        "row_peaks": np.asarray(payload["row_peaks"], dtype=np.uint8).tolist(),
        "preview_peaks": np.asarray(payload["preview_peaks"], dtype=np.uint8).tolist(),
        "detail_peaks": np.asarray(payload["detail_peaks"], dtype=np.uint8).tolist(),
        "envelope": {
            "preview_min": np.asarray(payload["preview_min"], dtype=np.int8).tolist(),
            "preview_max": np.asarray(payload["preview_max"], dtype=np.int8).tolist(),
            "detail_min": np.asarray(payload["detail_min"], dtype=np.int8).tolist(),
            "detail_max": np.asarray(payload["detail_max"], dtype=np.int8).tolist(),
        },
        "bands": {
            "low": np.asarray(payload["band_low"], dtype=np.uint8).tolist(),
            "mid": np.asarray(payload["band_mid"], dtype=np.uint8).tolist(),
            "high": np.asarray(payload["band_high"], dtype=np.uint8).tolist(),
        },
    }


def create_waveform_playlist_features_npz(
    tracklist_csv: str | Path,
    *,
    music_dir: str | Path | None = None,
    output_file: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    json_dir: str | Path | None = None,
    mix_slug: str | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    row_bins: int = DEFAULT_ROW_BINS,
    preview_bins: int = DEFAULT_PREVIEW_BINS,
    detail_bins: int = DEFAULT_DETAIL_BINS,
    band_bins: int = DEFAULT_BAND_BINS,
    skip_missing_audio: bool = False,
    force: bool = False,
) -> Path:
    """Build multi-resolution waveform summaries for a playlist and save a collection NPZ."""
    csv_path = Path(tracklist_csv).expanduser().resolve()
    tracks = load_playlist_tracks(
        csv_path,
        music_dir=music_dir,
        skip_missing_audio=skip_missing_audio,
    )
    slug = mix_slug or csv_path.parent.name

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_file = (
        Path(output_file).expanduser().resolve()
        if output_file is not None
        else resolved_output_dir / _default_npz_name(csv_path)
    )
    resolved_json_dir = (
        Path(json_dir).expanduser().resolve()
        if json_dir is not None
        else resolved_output_dir / slug
    )
    resolved_json_dir.mkdir(parents=True, exist_ok=True)

    row_rows: list[np.ndarray] = []
    preview_rows: list[np.ndarray] = []
    detail_rows: list[np.ndarray] = []
    preview_min_rows: list[np.ndarray] = []
    preview_max_rows: list[np.ndarray] = []
    detail_min_rows: list[np.ndarray] = []
    detail_max_rows: list[np.ndarray] = []
    band_low_rows: list[np.ndarray] = []
    band_mid_rows: list[np.ndarray] = []
    band_high_rows: list[np.ndarray] = []
    json_paths: list[str] = []
    durations: list[float] = []
    sample_rates: list[int] = []
    preview_starts: list[float] = []
    preview_ends: list[float] = []
    preview_durations: list[float] = []
    preview_scores: list[float] = []
    preview_methods: list[str] = []
    track_ids: list[str] = []

    for i, track in enumerate(tracks, start=1):
        track_id = f"{slug}:{track.track_number}"
        json_path = _json_asset_path(json_dir=resolved_json_dir, track=track)
        if json_path.exists() and not force:
            try:
                cached = json.loads(json_path.read_text(encoding="utf-8"))
                envelope = cached.get("envelope") or {}
                payload = {
                    "version": int(cached.get("version", WAVEFORM_FEATURE_VERSION)),
                    "source_audio_path": str(track.audio_path),
                    "sample_rate": int(cached.get("sample_rate", sample_rate)),
                    "duration_seconds": float(cached.get("duration_seconds", 0.0)),
                    "format": "uint8_0_255",
                    "signed_format": "int8_-127_127",
                    "row_bins": int(cached.get("row_bins", row_bins)),
                    "preview_bins": int(cached.get("preview_bins", preview_bins)),
                    "detail_bins": int(cached.get("detail_bins", detail_bins)),
                    "band_bins": int(cached.get("band_bins", band_bins)),
                    "preview_start_seconds": float(cached.get("preview_start_seconds", 0.0)),
                    "preview_end_seconds": float(cached.get("preview_end_seconds", 0.0)),
                    "preview_duration_seconds": float(cached.get("preview_duration_seconds", 0.0)),
                    "preview_score": float(cached.get("preview_score", 0.0)),
                    "preview_method": str(cached.get("preview_method") or "rms_middle_scan"),
                    "row_peaks": np.asarray(cached.get("row_peaks", []), dtype=np.uint8),
                    "preview_peaks": np.asarray(cached.get("preview_peaks", []), dtype=np.uint8),
                    "detail_peaks": np.asarray(cached.get("detail_peaks", []), dtype=np.uint8),
                    "preview_min": np.asarray(envelope.get("preview_min", []), dtype=np.int8),
                    "preview_max": np.asarray(envelope.get("preview_max", []), dtype=np.int8),
                    "detail_min": np.asarray(envelope.get("detail_min", []), dtype=np.int8),
                    "detail_max": np.asarray(envelope.get("detail_max", []), dtype=np.int8),
                    "band_low": np.asarray((cached.get("bands") or {}).get("low", []), dtype=np.uint8),
                    "band_mid": np.asarray((cached.get("bands") or {}).get("mid", []), dtype=np.uint8),
                    "band_high": np.asarray((cached.get("bands") or {}).get("high", []), dtype=np.uint8),
                }
                if (
                    payload["version"] != WAVEFORM_FEATURE_VERSION
                    or payload["row_peaks"].size != int(row_bins)
                    or payload["preview_peaks"].size != int(preview_bins)
                    or payload["detail_peaks"].size != int(detail_bins)
                    or payload["preview_min"].size != int(preview_bins)
                    or payload["preview_max"].size != int(preview_bins)
                    or payload["detail_min"].size != int(detail_bins)
                    or payload["detail_max"].size != int(detail_bins)
                    or payload["band_low"].size != int(band_bins)
                ):
                    raise ValueError("cached waveform shape mismatch")
            except Exception:
                payload = generate_waveform_feature_payload(
                    track.audio_path,
                    sample_rate=sample_rate,
                    row_bins=row_bins,
                    preview_bins=preview_bins,
                    detail_bins=detail_bins,
                    band_bins=band_bins,
                )
                json_path.write_text(
                    json.dumps(_json_payload_for_track(track, payload, track_id=track_id), separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                )
        else:
            payload = generate_waveform_feature_payload(
                track.audio_path,
                sample_rate=sample_rate,
                row_bins=row_bins,
                preview_bins=preview_bins,
                detail_bins=detail_bins,
                band_bins=band_bins,
            )
            json_path.write_text(
                json.dumps(_json_payload_for_track(track, payload, track_id=track_id), separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )

        row_rows.append(np.asarray(payload["row_peaks"], dtype=np.uint8))
        preview_rows.append(np.asarray(payload["preview_peaks"], dtype=np.uint8))
        detail_rows.append(np.asarray(payload["detail_peaks"], dtype=np.uint8))
        preview_min_rows.append(np.asarray(payload["preview_min"], dtype=np.int8))
        preview_max_rows.append(np.asarray(payload["preview_max"], dtype=np.int8))
        detail_min_rows.append(np.asarray(payload["detail_min"], dtype=np.int8))
        detail_max_rows.append(np.asarray(payload["detail_max"], dtype=np.int8))
        band_low_rows.append(np.asarray(payload["band_low"], dtype=np.uint8))
        band_mid_rows.append(np.asarray(payload["band_mid"], dtype=np.uint8))
        band_high_rows.append(np.asarray(payload["band_high"], dtype=np.uint8))
        json_paths.append(_to_project_relpath(json_path))
        durations.append(float(payload["duration_seconds"]))
        sample_rates.append(int(payload["sample_rate"]))
        preview_starts.append(float(payload.get("preview_start_seconds", 0.0)))
        preview_ends.append(float(payload.get("preview_end_seconds", 0.0)))
        preview_durations.append(float(payload.get("preview_duration_seconds", 0.0)))
        preview_scores.append(float(payload.get("preview_score", 0.0)))
        preview_methods.append(str(payload.get("preview_method") or "rms_middle_scan"))
        track_ids.append(track_id)
        print(f"[{i:03d}/{len(tracks):03d}] waveform {track_id} {track.title}")

    metadata = _metadata_arrays(tracks)
    created_utc = datetime.now(tz=timezone.utc).isoformat()
    npz_payload: dict[str, Any] = {
        "feature_type": np.array("waveform", dtype=np.str_),
        "waveform_feature_version": np.array(WAVEFORM_FEATURE_VERSION, dtype=np.int32),
        "playlist_csv": np.array(_to_project_relpath(csv_path), dtype=np.str_),
        "created_utc": np.array(created_utc, dtype=np.str_),
        "num_tracks": np.array(len(tracks), dtype=np.int32),
        "track_ids": _string_array(track_ids),
        "waveform_json_paths": _string_array(json_paths),
        "duration_seconds": np.asarray(durations, dtype=np.float32),
        "sample_rate": np.asarray(sample_rates, dtype=np.int32),
        "preview_start_seconds": np.asarray(preview_starts, dtype=np.float32),
        "preview_end_seconds": np.asarray(preview_ends, dtype=np.float32),
        "preview_duration_seconds": np.asarray(preview_durations, dtype=np.float32),
        "preview_score": np.asarray(preview_scores, dtype=np.float32),
        "preview_method": _string_array(preview_methods),
        "row_bins": np.array(int(row_bins), dtype=np.int32),
        "preview_bins": np.array(int(preview_bins), dtype=np.int32),
        "detail_bins": np.array(int(detail_bins), dtype=np.int32),
        "band_bins": np.array(int(band_bins), dtype=np.int32),
        "row_peaks": np.vstack(row_rows).astype(np.uint8),
        "preview_peaks": np.vstack(preview_rows).astype(np.uint8),
        "detail_peaks": np.vstack(detail_rows).astype(np.uint8),
        "preview_min": np.vstack(preview_min_rows).astype(np.int8),
        "preview_max": np.vstack(preview_max_rows).astype(np.int8),
        "detail_min": np.vstack(detail_min_rows).astype(np.int8),
        "detail_max": np.vstack(detail_max_rows).astype(np.int8),
        "band_low": np.vstack(band_low_rows).astype(np.uint8),
        "band_mid": np.vstack(band_mid_rows).astype(np.uint8),
        "band_high": np.vstack(band_high_rows).astype(np.uint8),
    }
    npz_payload.update(metadata)

    resolved_output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(resolved_output_file, **npz_payload)
    print(f"Saved waveform collection: {_to_project_relpath(resolved_output_file)}")
    print(f"Tracks: {len(tracks)}")
    print(f"JSON dir: {_to_project_relpath(resolved_json_dir)}")
    return resolved_output_file


def _norm_token(value: str) -> str:
    text = str(value or "").strip().lower().replace("\\", "/")
    name = text.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    return "".join(ch for ch in stem if ch.isalnum())


def _dequantize(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return []
    if float(np.nanmax(arr)) > 1.0:
        arr = arr / 255.0
    return [round(float(np.clip(v, 0.0, 1.0)), 4) for v in arr]


def _dequantize_signed(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return []
    if float(np.nanmax(np.abs(arr))) > 1.001:
        arr = arr / 127.0
    return [round(float(np.clip(v, -1.0, 1.0)), 4) for v in arr]


def load_waveform_feature_lookup(npz_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a waveform NPZ and return metadata keyed by normalized filename."""
    resolved = Path(npz_path).expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as data:
        filenames = [str(v) for v in np.asarray(data["filenames"], dtype=np.str_)]
        json_paths = [str(v) for v in np.asarray(data["waveform_json_paths"], dtype=np.str_)]
        row_peaks = np.asarray(data["row_peaks"], dtype=np.uint8)
        preview_peaks = np.asarray(data["preview_peaks"], dtype=np.uint8) if "preview_peaks" in data.files else None
        preview_min = np.asarray(data["preview_min"], dtype=np.int8) if "preview_min" in data.files else None
        preview_max = np.asarray(data["preview_max"], dtype=np.int8) if "preview_max" in data.files else None
        detail_bins = int(np.asarray(data["detail_bins"]).reshape(-1)[0]) if "detail_bins" in data.files else 0
        band_bins = int(np.asarray(data["band_bins"]).reshape(-1)[0]) if "band_bins" in data.files else 0
        durations = np.asarray(data["duration_seconds"], dtype=np.float32) if "duration_seconds" in data.files else None
        preview_starts = np.asarray(data["preview_start_seconds"], dtype=np.float32) if "preview_start_seconds" in data.files else None
        preview_ends = np.asarray(data["preview_end_seconds"], dtype=np.float32) if "preview_end_seconds" in data.files else None
        preview_scores = np.asarray(data["preview_score"], dtype=np.float32) if "preview_score" in data.files else None
        preview_methods = np.asarray(data["preview_method"], dtype=np.str_) if "preview_method" in data.files else None

        lookup: dict[str, dict[str, Any]] = {}
        for i, filename in enumerate(filenames):
            json_path = Path(json_paths[i])
            if not json_path.is_absolute():
                json_path = (PROJECT_ROOT / json_path).resolve()
            entry = {
                "waveform_path": str(json_path),
                "waveform_peaks": _dequantize(row_peaks[i]),
                "waveform_preview_peaks": _dequantize(preview_peaks[i]) if preview_peaks is not None else [],
                "waveform_preview_min": _dequantize_signed(preview_min[i]) if preview_min is not None else [],
                "waveform_preview_max": _dequantize_signed(preview_max[i]) if preview_max is not None else [],
                "waveform_detail_bins": detail_bins,
                "waveform_band_bins": band_bins,
            }
            if durations is not None and i < durations.size and np.isfinite(float(durations[i])):
                entry["duration_seconds"] = float(durations[i])
            if preview_starts is not None and i < preview_starts.size and np.isfinite(float(preview_starts[i])):
                entry["preview_start_seconds"] = float(preview_starts[i])
            if preview_ends is not None and i < preview_ends.size and np.isfinite(float(preview_ends[i])):
                entry["preview_end_seconds"] = float(preview_ends[i])
            if preview_scores is not None and i < preview_scores.size and np.isfinite(float(preview_scores[i])):
                entry["preview_score"] = float(preview_scores[i])
            if preview_methods is not None and i < preview_methods.size:
                entry["preview_method"] = str(preview_methods[i])
            lookup[_norm_token(filename)] = entry
            lookup[Path(filename).name.lower()] = entry
    return lookup


def default_waveform_npz_path(tracklist_csv: str | Path, *, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    csv_path = Path(tracklist_csv).expanduser().resolve()
    return Path(output_dir).expanduser().resolve() / _default_npz_name(csv_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute detailed waveform summaries for a playlist.")
    parser.add_argument("tracklist_csv", type=Path, help="Playlist track CSV.")
    parser.add_argument("--music-dir", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-dir", type=Path, default=None)
    parser.add_argument("--mix-slug", type=str, default=None)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--row-bins", type=int, default=DEFAULT_ROW_BINS)
    parser.add_argument("--preview-bins", type=int, default=DEFAULT_PREVIEW_BINS)
    parser.add_argument("--detail-bins", type=int, default=DEFAULT_DETAIL_BINS)
    parser.add_argument("--band-bins", type=int, default=DEFAULT_BAND_BINS)
    parser.add_argument("--skip-missing-audio", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute existing per-track JSON files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    create_waveform_playlist_features_npz(
        args.tracklist_csv,
        music_dir=args.music_dir,
        output_file=args.output_file,
        output_dir=args.output_dir,
        json_dir=args.json_dir,
        mix_slug=args.mix_slug,
        sample_rate=int(args.sample_rate),
        row_bins=int(args.row_bins),
        preview_bins=int(args.preview_bins),
        detail_bins=int(args.detail_bins),
        band_bins=int(args.band_bins),
        skip_missing_audio=bool(args.skip_missing_audio),
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
