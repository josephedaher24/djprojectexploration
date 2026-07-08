"""Groove embedding utilities based on TempoCNN and multiband onset flux."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
from essentia.standard import MonoLoader, TempoCNN

from djprojectexploration.tempo_embedding import (
    DEFAULT_TEMPOCNN_MODEL_URL,
    resolve_tempocnn_model_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_TEMPOCNN_SAMPLE_RATE = 11025
DEFAULT_HOP_LENGTH = 256
DEFAULT_N_FFT = 2048
DEFAULT_SNIPPET_LENGTH_SEC: float | None = None
DEFAULT_SUBDIVISIONS_PER_BEAT = 4
DEFAULT_BEATS_PER_BAR = 4
DEFAULT_PHRASE_BARS = 1
DEFAULT_POOLING_MODE = "mean"
DEFAULT_POOLING_TOPK = 3
DEFAULT_BANDS_HZ: dict[str, tuple[float, float]] = {
    "low": (40.0, 180.0),
    "mid": (180.0, 2000.0),
    "high": (2000.0, 10000.0),
}
DEFAULT_PHASE_ALIGN_LOW_MID_WEIGHTS = (0.3, 0.7)


def _to_project_relpath(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    """Return project-relative path when possible, else absolute path."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


def bpm_from_beat_times(beat_times: np.ndarray, fallback_bpm: float) -> float:
    """Estimate BPM from median inter-beat interval, falling back when needed."""
    beats = np.asarray(beat_times, dtype=np.float32)
    if beats.size >= 2:
        ibi = np.diff(beats.astype(np.float64))
        ibi = ibi[np.isfinite(ibi) & (ibi > 1e-6)]
        if ibi.size:
            return float(60.0 / np.median(ibi))

    if np.isfinite(fallback_bpm) and fallback_bpm > 0:
        return float(fallback_bpm)
    return float("nan")


def build_beat_grid_from_bpm_onset(
    audio_duration: float,
    bpm: float,
    first_onset_sec: float,
) -> np.ndarray:
    """Build regular beat ticks from BPM and a phase/onset anchor."""
    if bpm <= 0:
        raise ValueError("bpm must be positive.")

    beat_period = 60.0 / float(bpm)
    first_beat = float(np.clip(first_onset_sec, 0.0, float(audio_duration)))
    while first_beat - beat_period >= 0.0:
        first_beat -= beat_period

    beat_times = np.arange(first_beat, float(audio_duration) + beat_period, beat_period, dtype=np.float32)
    beat_times = beat_times[beat_times <= float(audio_duration)]

    if beat_times.size < 2:
        beat_times = np.arange(0.0, float(audio_duration) + beat_period, beat_period, dtype=np.float32)
        beat_times = beat_times[beat_times <= float(audio_duration)]

    return beat_times.astype(np.float32)


def normalize_to_unit_peak(x: np.ndarray) -> np.ndarray:
    """Normalize a 1D signal by absolute peak."""
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    peak = float(np.max(np.abs(arr)))
    if peak <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / peak).astype(np.float32)


def estimate_global_onset_phase_shift_score(
    beat_times: np.ndarray,
    onset_times: np.ndarray,
    onset_env: np.ndarray,
    *,
    max_shift_sec: float = 0.18,
    step_sec: float = 0.002,
) -> tuple[float, float]:
    """Find the global beat-grid shift that best samples onset-envelope peaks."""
    beats = np.asarray(beat_times, dtype=np.float32)
    times = np.asarray(onset_times, dtype=np.float32)
    env = normalize_to_unit_peak(np.asarray(onset_env, dtype=np.float32))

    if beats.size == 0 or times.size == 0 or env.size == 0:
        return 0.0, float("-inf")

    max_shift_sec = float(max(0.0, max_shift_sec))
    step_sec = float(max(1e-4, step_sec))
    shifts = np.arange(-max_shift_sec, max_shift_sec + step_sec, step_sec, dtype=np.float32)
    tmin = float(times[0])
    tmax = float(times[-1])

    best_shift = 0.0
    best_score = float("-inf")
    for shift in shifts:
        query = beats + float(shift)
        valid = (query >= tmin) & (query <= tmax)
        if not np.any(valid):
            continue
        score = float(np.mean(np.interp(query[valid], times, env)))
        if score > best_score:
            best_score = score
            best_shift = float(shift)

    return best_shift, best_score


def prepend_missing_start_beats(
    beat_times: np.ndarray,
    duration_sec: float,
    fallback_bpm: float,
    *,
    max_missing_beats: int = 2,
    snap_tolerance_sec: float = 0.12,
) -> tuple[np.ndarray, int]:
    """Prepend 1-2 near-zero beats when a regular grid starts one beat late."""
    beats = np.asarray(beat_times, dtype=np.float32)
    beats = beats[np.isfinite(beats)]
    beats = np.unique(beats)
    beats = beats[(beats >= 0.0) & (beats <= float(duration_sec))]

    if beats.size == 0:
        return beats, 0

    if beats.size >= 2:
        ibi = np.diff(beats.astype(np.float64))
        ibi = ibi[np.isfinite(ibi) & (ibi > 1e-6)]
        beat_period = float(np.median(ibi)) if ibi.size else float("nan")
    elif np.isfinite(fallback_bpm) and fallback_bpm > 0:
        beat_period = float(60.0 / fallback_bpm)
    else:
        beat_period = float("nan")

    if not np.isfinite(beat_period) or beat_period <= 0:
        return beats, 0

    max_missing_beats = int(max(0, max_missing_beats))
    snap_tolerance_sec = float(max(0.0, snap_tolerance_sec))
    first = float(beats[0])

    for n in range(1, max_missing_beats + 1):
        candidate = first - n * beat_period
        if abs(candidate) <= snap_tolerance_sec:
            prep = first - beat_period * np.arange(n, 0, -1, dtype=np.float32)
            prep = np.clip(prep, 0.0, float(duration_sec)).astype(np.float32)
            merged = np.unique(np.concatenate([prep, beats]).astype(np.float32))
            return merged, int(n)

    return beats, 0


def band_onset_flux(
    stft_magnitude: np.ndarray,
    frequencies_hz: np.ndarray,
    band_low_hz: float,
    band_high_hz: float,
) -> np.ndarray:
    """Compute normalized positive spectral flux for one frequency band."""
    mask = (frequencies_hz >= band_low_hz) & (frequencies_hz < band_high_hz)
    if not np.any(mask):
        return np.zeros(stft_magnitude.shape[1], dtype=np.float32)

    band_mag = np.asarray(stft_magnitude[mask], dtype=np.float32)
    positive_flux = np.maximum(np.diff(band_mag, axis=1), 0.0)
    env = positive_flux.mean(axis=0).astype(np.float32)
    env = np.pad(env, (1, 0), mode="constant")
    env /= float(np.max(env)) + 1e-8
    return env.astype(np.float32)


def beat_sync_multiband_tensor(
    beat_times: np.ndarray,
    frame_times: np.ndarray,
    band_envs: list[np.ndarray],
    *,
    subdivisions: int = DEFAULT_SUBDIVISIONS_PER_BEAT,
    pooling_mode: str = DEFAULT_POOLING_MODE,
    pooling_topk: int = DEFAULT_POOLING_TOPK,
    normalize_per_beat: bool = True,
) -> np.ndarray:
    """Pool multiband onset envelopes into [beat, subdivision, band]."""
    subdivisions = int(subdivisions)
    if subdivisions <= 0:
        raise ValueError("subdivisions must be a positive integer.")

    pooling_mode = str(pooling_mode).strip().lower()
    if pooling_mode not in {"center", "mean", "max", "topk_mean"}:
        raise ValueError("pooling_mode must be one of {'center', 'mean', 'max', 'topk_mean'}.")

    pooling_topk = int(max(1, pooling_topk))
    beats = np.asarray(beat_times, dtype=np.float32)
    times = np.asarray(frame_times, dtype=np.float32)
    n_intervals = max(0, int(beats.size) - 1)
    tensor = np.zeros((n_intervals, subdivisions, len(band_envs)), dtype=np.float32)

    for i in range(n_intervals):
        start = float(beats[i])
        end = float(beats[i + 1])
        if end <= start:
            continue

        edges = start + (np.arange(subdivisions + 1, dtype=np.float32) / subdivisions) * (end - start)
        centers = start + ((np.arange(subdivisions, dtype=np.float32) + 0.5) / subdivisions) * (end - start)
        for b, env_raw in enumerate(band_envs):
            env = np.asarray(env_raw, dtype=np.float32)
            if pooling_mode == "center":
                tensor[i, :, b] = np.interp(centers, times, env).astype(np.float32)
                continue

            pooled = np.zeros(subdivisions, dtype=np.float32)
            for s in range(subdivisions):
                left = float(edges[s])
                right = float(edges[s + 1])
                if s == subdivisions - 1:
                    mask = (times >= left) & (times <= right)
                else:
                    mask = (times >= left) & (times < right)

                values = env[mask]
                if values.size == 0:
                    pooled[s] = float(np.interp(centers[s], times, env))
                elif pooling_mode == "mean":
                    pooled[s] = float(np.mean(values))
                elif pooling_mode == "max":
                    pooled[s] = float(np.max(values))
                else:
                    k = int(min(pooling_topk, values.size))
                    pooled[s] = float(np.mean(np.sort(values)[-k:]))

            tensor[i, :, b] = pooled

    if normalize_per_beat:
        tensor /= np.sum(tensor, axis=(1, 2), keepdims=True) + 1e-8
    return tensor.astype(np.float32)


def summarize_groove_profiles(
    beat_sync_tensor: np.ndarray,
    *,
    subdivisions_per_beat: int = DEFAULT_SUBDIVISIONS_PER_BEAT,
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
    phrase_bars: int = DEFAULT_PHRASE_BARS,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return mean beat profile, mean phrase profile, and complete phrase count."""
    tensor = np.asarray(beat_sync_tensor, dtype=np.float32)
    subdivisions_per_beat = int(subdivisions_per_beat)
    phrase_beats = max(1, int(beats_per_bar) * int(phrase_bars))
    phrase_subdivisions = subdivisions_per_beat * phrase_beats
    n_bands = int(tensor.shape[2]) if tensor.ndim == 3 else 0

    if tensor.size == 0:
        beat_profile = np.zeros((subdivisions_per_beat, n_bands), dtype=np.float32)
        phrase_profile = np.zeros((phrase_subdivisions, n_bands), dtype=np.float32)
        return beat_profile, phrase_profile, 0

    beat_profile = tensor.mean(axis=0).astype(np.float32)
    n_complete_beats = (tensor.shape[0] // phrase_beats) * phrase_beats
    if n_complete_beats >= phrase_beats:
        phrase_tensor = tensor[:n_complete_beats].reshape(
            -1,
            phrase_beats,
            subdivisions_per_beat,
            n_bands,
        )
        phrase_tensor = phrase_tensor.reshape(phrase_tensor.shape[0], phrase_subdivisions, n_bands)
        phrase_profile = phrase_tensor.mean(axis=0).astype(np.float32)
        n_complete_phrases = int(phrase_tensor.shape[0])
    else:
        phrase_profile = np.tile(beat_profile, (phrase_beats, 1)).astype(np.float32)
        n_complete_phrases = 0

    return beat_profile, phrase_profile, n_complete_phrases


def _tempocnn_tempo(
    audio_path: Path,
    *,
    model_file: str | Path | None,
    auto_download_model: bool,
    model_url: str,
    sample_rate: int,
    snippet_length_sec: float | None,
) -> tuple[float, np.ndarray, np.ndarray, Path]:
    resolved_model = resolve_tempocnn_model_file(
        model_file=model_file,
        auto_download=auto_download_model,
        model_url=model_url,
    )
    audio_tc = MonoLoader(filename=str(audio_path), sampleRate=int(sample_rate), resampleQuality=4)()
    if snippet_length_sec is not None:
        audio_tc = audio_tc[: int(float(snippet_length_sec) * int(sample_rate))]
    global_bpm, local_bpm_raw, local_prob_raw = TempoCNN(graphFilename=str(resolved_model))(audio_tc)
    return (
        float(global_bpm),
        np.asarray(local_bpm_raw, dtype=np.float32),
        np.asarray(local_prob_raw, dtype=np.float32),
        resolved_model,
    )


def _choose_phase_alignment(
    *,
    beat_times: np.ndarray,
    onset_env: np.ndarray,
    onset_env_times: np.ndarray,
    stft_mag: np.ndarray,
    freqs_hz: np.ndarray,
    frame_times: np.ndarray,
    bands_hz: dict[str, tuple[float, float]],
    requested_mode: str,
    low_mid_weights: tuple[float, float],
    adaptive_margin: float,
    max_shift_sec: float,
    step_sec: float,
) -> tuple[str, float, dict[str, float], dict[str, float]]:
    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "full": (
            np.asarray(onset_env_times, dtype=np.float32),
            normalize_to_unit_peak(np.asarray(onset_env, dtype=np.float32)),
        )
    }

    low_band = bands_hz.get("low")
    mid_band = bands_hz.get("mid")
    if low_band is not None and mid_band is not None:
        low_env = band_onset_flux(stft_mag, freqs_hz, *low_band)
        mid_env = band_onset_flux(stft_mag, freqs_hz, *mid_band)
        low_w, mid_w = low_mid_weights
        low_mid_env = normalize_to_unit_peak((float(low_w) * low_env + float(mid_w) * mid_env).astype(np.float32))
        if low_mid_env.size == frame_times.size and np.any(low_mid_env > 0):
            candidates["low_mid"] = (frame_times, low_mid_env)

    scores: dict[str, float] = {}
    shifts: dict[str, float] = {}
    for mode_name, (candidate_times, candidate_env) in candidates.items():
        shift, score = estimate_global_onset_phase_shift_score(
            beat_times=beat_times,
            onset_times=candidate_times,
            onset_env=candidate_env,
            max_shift_sec=max_shift_sec,
            step_sec=step_sec,
        )
        scores[mode_name] = float(score)
        shifts[mode_name] = float(shift)

    requested = str(requested_mode).strip().lower()
    if requested not in {"full", "low_mid", "adaptive"}:
        raise ValueError("phase_align_mode must be one of {'full', 'low_mid', 'adaptive'}.")

    if requested == "full":
        mode = "full"
    elif requested == "low_mid":
        mode = "low_mid" if "low_mid" in shifts else "full"
    elif "full" in scores and "low_mid" in scores:
        mode = "low_mid" if scores["low_mid"] >= scores["full"] + float(adaptive_margin) else "full"
    elif "low_mid" in scores:
        mode = "low_mid"
    else:
        mode = "full"

    return mode, float(shifts.get(mode, 0.0)), scores, shifts


def generate_groove_embedding(
    audio_file: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    tempocnn_sample_rate: int = DEFAULT_TEMPOCNN_SAMPLE_RATE,
    model_file: str | Path | None = None,
    auto_download_model: bool = False,
    model_url: str = DEFAULT_TEMPOCNN_MODEL_URL,
    snippet_length_sec: float | None = DEFAULT_SNIPPET_LENGTH_SEC,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_fft: int = DEFAULT_N_FFT,
    bands_hz: dict[str, tuple[float, float]] | None = None,
    manual_bpm: float | None = None,
    onset_time_sec: float | None = 0.0,
    auto_phase_align: bool = True,
    phase_align_mode: str = "adaptive",
    phase_align_max_shift_sec: float = 0.18,
    phase_align_step_sec: float = 0.002,
    phase_align_low_mid_weights: tuple[float, float] = DEFAULT_PHASE_ALIGN_LOW_MID_WEIGHTS,
    phase_align_adaptive_margin: float = 0.01,
    auto_prepend_start_beats: bool = True,
    max_missing_start_beats: int = 2,
    start_beat_snap_tolerance_sec: float = 0.12,
    subdivisions_per_beat: int = DEFAULT_SUBDIVISIONS_PER_BEAT,
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
    phrase_bars: int = DEFAULT_PHRASE_BARS,
    pooling_mode: str = DEFAULT_POOLING_MODE,
    pooling_topk: int = DEFAULT_POOLING_TOPK,
    profile_mode: str = "phrase",
    normalize_per_beat: bool = True,
) -> dict[str, Any]:
    """Generate a beat/phrase-synchronous multiband groove embedding for one track."""
    audio_path = Path(audio_file).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if manual_bpm is not None and manual_bpm <= 0:
        raise ValueError("manual_bpm must be positive.")
    if onset_time_sec is not None and onset_time_sec < 0:
        raise ValueError("onset_time_sec must be >= 0.")

    profile_mode = str(profile_mode).strip().lower()
    if profile_mode not in {"beat", "phrase"}:
        raise ValueError("profile_mode must be one of {'beat', 'phrase'}.")

    bands = dict(DEFAULT_BANDS_HZ if bands_hz is None else bands_hz)
    if not bands:
        raise ValueError("At least one frequency band is required.")

    audio = MonoLoader(filename=str(audio_path), sampleRate=int(sample_rate), resampleQuality=4)()
    if snippet_length_sec is not None:
        if snippet_length_sec <= 0:
            raise ValueError("snippet_length_sec must be positive when provided.")
        audio = audio[: int(float(snippet_length_sec) * int(sample_rate))]
    audio = np.asarray(audio, dtype=np.float32)
    duration_sec = float(audio.size) / float(sample_rate)

    stft_mag = np.abs(librosa.stft(audio, n_fft=int(n_fft), hop_length=int(hop_length), center=True))
    freqs_hz = librosa.fft_frequencies(sr=int(sample_rate), n_fft=int(n_fft))
    frame_times = librosa.frames_to_time(
        np.arange(stft_mag.shape[1]),
        sr=int(sample_rate),
        hop_length=int(hop_length),
    ).astype(np.float32)

    tempocnn_model_file: Path | None = None
    local_bpm = np.array([], dtype=np.float32)
    local_probability = np.array([], dtype=np.float32)
    tempocnn_bpm = float("nan")

    has_explicit_phase_anchor = onset_time_sec is not None
    phase_anchor_sec = 0.0 if onset_time_sec is None else float(onset_time_sec)
    if manual_bpm is None:
        tempocnn_bpm, local_bpm, local_probability, tempocnn_model_file = _tempocnn_tempo(
            audio_path,
            model_file=model_file,
            auto_download_model=auto_download_model,
            model_url=model_url,
            sample_rate=int(tempocnn_sample_rate),
            snippet_length_sec=snippet_length_sec,
        )
        bpm_seed = float(tempocnn_bpm) if np.isfinite(tempocnn_bpm) and tempocnn_bpm > 0 else 120.0
        beat_source = "tempocnn_grid"
    else:
        bpm_seed = float(manual_bpm)
        beat_source = "manual_bpm_grid"

    beat_times = build_beat_grid_from_bpm_onset(
        audio_duration=duration_sec,
        bpm=bpm_seed,
        first_onset_sec=phase_anchor_sec,
    )

    phase_shift_sec = 0.0
    phase_align_mode_used = "none"
    phase_align_search_mode = "explicit_onset_local" if has_explicit_phase_anchor else "missing_onset_full_beat"
    phase_align_search_max_shift_sec = float(phase_align_max_shift_sec)
    phase_align_scores: dict[str, float] = {}
    phase_align_shifts: dict[str, float] = {}
    if auto_phase_align and beat_times.size >= 2:
        if not has_explicit_phase_anchor and np.isfinite(bpm_seed) and bpm_seed > 0:
            beat_period_sec = 60.0 / float(bpm_seed)
            phase_align_search_max_shift_sec = max(
                float(phase_align_step_sec),
                0.5 * beat_period_sec,
            )
        onset_env = librosa.onset.onset_strength(y=audio, sr=int(sample_rate), hop_length=int(hop_length))
        onset_env_times = librosa.times_like(onset_env, sr=int(sample_rate), hop_length=int(hop_length))
        phase_align_mode_used, phase_shift_sec, phase_align_scores, phase_align_shifts = _choose_phase_alignment(
            beat_times=beat_times,
            onset_env=onset_env,
            onset_env_times=onset_env_times,
            stft_mag=stft_mag,
            freqs_hz=freqs_hz,
            frame_times=frame_times,
            bands_hz=bands,
            requested_mode=phase_align_mode,
            low_mid_weights=phase_align_low_mid_weights,
            adaptive_margin=float(phase_align_adaptive_margin),
            max_shift_sec=phase_align_search_max_shift_sec,
            step_sec=float(phase_align_step_sec),
        )
        shifted = beat_times + float(phase_shift_sec)
        shifted = shifted[(shifted >= 0.0) & (shifted <= duration_sec)]
        shifted = np.unique(shifted.astype(np.float32))
        if shifted.size >= 2:
            beat_times = shifted
            beat_source = f"{beat_source}_phase_aligned_{phase_align_mode_used}"
            if not has_explicit_phase_anchor:
                beat_source = f"{beat_source}_full_beat_search"

    prepended_count = 0
    if auto_prepend_start_beats:
        beat_times, prepended_count = prepend_missing_start_beats(
            beat_times=beat_times,
            duration_sec=duration_sec,
            fallback_bpm=bpm_seed,
            max_missing_beats=int(max_missing_start_beats),
            snap_tolerance_sec=float(start_beat_snap_tolerance_sec),
        )
        if prepended_count > 0:
            beat_source = f"{beat_source}_prepended_start_{prepended_count}"

    band_names = list(bands.keys())
    band_envs = [band_onset_flux(stft_mag, freqs_hz, *bands[name]) for name in band_names]
    beat_sync_tensor = beat_sync_multiband_tensor(
        beat_times=beat_times,
        frame_times=frame_times,
        band_envs=band_envs,
        subdivisions=int(subdivisions_per_beat),
        pooling_mode=pooling_mode,
        pooling_topk=int(pooling_topk),
        normalize_per_beat=bool(normalize_per_beat),
    )
    beat_profile, phrase_profile, n_complete_phrases = summarize_groove_profiles(
        beat_sync_tensor,
        subdivisions_per_beat=int(subdivisions_per_beat),
        beats_per_bar=int(beats_per_bar),
        phrase_bars=int(phrase_bars),
    )
    selected_profile = phrase_profile if profile_mode == "phrase" else beat_profile
    embedding = selected_profile.reshape(-1).astype(np.float32)
    bpm_from_grid = bpm_from_beat_times(beat_times, fallback_bpm=bpm_seed)

    return {
        "title": "Groove Embedding (TempoCNN + Beat-Synchronous Multiband Onsets)",
        "filename": audio_path.name,
        "audio_file": _to_project_relpath(audio_path),
        "embedding_type": "groove",
        "embedding_subtype": f"{profile_mode}_synchronous_multiband_onset_flux",
        "embedding_dimension": int(embedding.size),
        "embedding": embedding.tolist(),
        "band_names": band_names,
        "bands_hz": {name: [float(lo), float(hi)] for name, (lo, hi) in bands.items()},
        "beat_profile": beat_profile.tolist(),
        "phrase_profile": phrase_profile.tolist(),
        "beat_sync_tensor_shape": list(beat_sync_tensor.shape),
        "beat_pooling": {
            "bpm": float(bpm_from_grid),
            "bpm_seed": float(bpm_seed),
            "tempocnn_bpm": None if not np.isfinite(tempocnn_bpm) else float(tempocnn_bpm),
            "beat_count": int(beat_times.size),
            "beat_source": beat_source,
            "phase_anchor_seconds": float(phase_anchor_sec),
            "phase_shift_seconds": float(phase_shift_sec),
            "phase_align_mode": phase_align_mode_used,
            "phase_align_search_mode": phase_align_search_mode,
            "phase_align_search_max_shift_seconds": float(phase_align_search_max_shift_sec),
            "phase_align_scores": phase_align_scores,
            "phase_align_shifts": phase_align_shifts,
            "prepended_start_beats": int(prepended_count),
            "complete_phrases": int(n_complete_phrases),
        },
        "local_bpm": local_bpm.tolist(),
        "local_probability": local_probability.tolist(),
        "config": {
            "sample_rate": int(sample_rate),
            "tempocnn_sample_rate": int(tempocnn_sample_rate),
            "tempocnn_model_file": None if tempocnn_model_file is None else _to_project_relpath(tempocnn_model_file),
            "tempocnn_model_url": model_url,
            "snippet_length_sec": None if snippet_length_sec is None else float(snippet_length_sec),
            "hop_length": int(hop_length),
            "n_fft": int(n_fft),
            "manual_bpm": None if manual_bpm is None else float(manual_bpm),
            "onset_time_sec": None if onset_time_sec is None else float(onset_time_sec),
            "auto_phase_align": bool(auto_phase_align),
            "phase_align_mode": phase_align_mode,
            "phase_align_max_shift_sec": float(phase_align_max_shift_sec),
            "phase_align_missing_onset_search": "full_beat",
            "phase_align_step_sec": float(phase_align_step_sec),
            "phase_align_low_mid_weights": [float(v) for v in phase_align_low_mid_weights],
            "phase_align_adaptive_margin": float(phase_align_adaptive_margin),
            "auto_prepend_start_beats": bool(auto_prepend_start_beats),
            "max_missing_start_beats": int(max_missing_start_beats),
            "start_beat_snap_tolerance_sec": float(start_beat_snap_tolerance_sec),
            "subdivisions_per_beat": int(subdivisions_per_beat),
            "beats_per_bar": int(beats_per_bar),
            "phrase_bars": int(phrase_bars),
            "pooling_mode": pooling_mode,
            "pooling_topk": int(pooling_topk),
            "profile_mode": profile_mode,
            "normalize_per_beat": bool(normalize_per_beat),
        },
    }


def resolve_band_weight_vector(
    band_names: list[str],
    band_weights: dict[str, float] | np.ndarray,
) -> np.ndarray:
    """Resolve named or positional band weights for groove-profile comparison."""
    if isinstance(band_weights, dict):
        weights = np.array([float(band_weights.get(str(name), 1.0)) for name in band_names], dtype=np.float64)
    else:
        weights = np.asarray(band_weights, dtype=np.float64).reshape(-1)
        if weights.size != len(band_names):
            raise ValueError(f"band_weights size ({weights.size}) must match band count ({len(band_names)}).")
    if np.any(~np.isfinite(weights)):
        raise ValueError("band_weights must be finite.")
    if np.any(weights < 0.0):
        raise ValueError("band_weights must be non-negative.")
    if np.sum(weights) <= 0.0:
        raise ValueError("band_weights must not all be zero.")
    return weights


def weighted_cosine_profile(
    profile_a: np.ndarray,
    profile_b: np.ndarray,
    *,
    band_weights: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute notebook-compatible weighted cosine similarity between profiles."""
    a = np.asarray(profile_a, dtype=np.float64)
    b = np.asarray(profile_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Profile shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 2:
        raise ValueError(f"Expected 2D profiles, got shape {a.shape}")

    weights = np.asarray(band_weights, dtype=np.float64).reshape(1, -1)
    if weights.shape[1] != a.shape[1]:
        raise ValueError(f"Band-weight length ({weights.shape[1]}) does not match profile bands ({a.shape[1]}).")

    weighted_dot = float(np.sum(weights * a * b))
    weighted_norm_a = float(np.sum(weights * a * a))
    weighted_norm_b = float(np.sum(weights * b * b))
    denom = np.sqrt(weighted_norm_a * weighted_norm_b) + float(eps)
    return float(weighted_dot / denom)


def pairwise_weighted_cosine_matrix(
    profile_items: list[dict[str, Any]],
    *,
    band_weights: dict[str, float] | np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute pairwise weighted cosine similarity for generated groove profiles."""
    if len(profile_items) == 0:
        raise ValueError("profile_items is empty.")

    band_names = np.asarray(profile_items[0]["band_names"], dtype=object).astype(str).tolist()
    weight_vector = resolve_band_weight_vector(band_names, band_weights)

    n = len(profile_items)
    similarities = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        similarities[i, i] = 1.0
        for j in range(i + 1, n):
            sim = weighted_cosine_profile(
                np.asarray(profile_items[i]["profile"], dtype=np.float32),
                np.asarray(profile_items[j]["profile"], dtype=np.float32),
                band_weights=weight_vector,
                eps=eps,
            )
            similarities[i, j] = sim
            similarities[j, i] = sim

    return similarities, weight_vector, band_names
