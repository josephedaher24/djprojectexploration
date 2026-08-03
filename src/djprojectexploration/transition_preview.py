"""Render prototype beat-aligned transition previews from playlist CSV metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import librosa
import numpy as np
from scipy import signal
import soundfile as sf

from djprojectexploration.tracklists import (
    PROJECT_ROOT,
    PlaylistTrack,
    load_playlist_tracks,
    optional_float,
    read_csv_rows,
)
from djprojectexploration.loudness_matching import (
    DEFAULT_TARGET_LUFS,
    MAX_COMPENSATION_DB,
    LoudnessMeasurement,
    normalize_mode,
)


DEFAULT_TRACKLIST = PROJECT_ROOT / "music" / "aries-mix" / "aries_mix_tracks.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "transitions"
DEFAULT_SAMPLE_RATE = 44100
PREVIEW_SAMPLE_RATE = DEFAULT_SAMPLE_RATE
TRANSITION_QUALITY_PROFILES = {
    "preview": {
        "sample_rate": PREVIEW_SAMPLE_RATE,
        "rubberband_flag": "--fast",
        "stretch_algorithm": "rubberband-r2-duration",
    },
    "final": {
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "rubberband_flag": "--fine",
        "stretch_algorithm": "rubberband-r3-duration",
    },
}
TRANSITION_QUALITY_CHOICES = tuple(TRANSITION_QUALITY_PROFILES)
DEFAULT_VOLUME_MODE = "overlap-crossfade"
DEFAULT_EQ_MODE = "none"
DEFAULT_FILTER_MODE = "none"
MIX_HEADROOM_DB = -3.0
MIX_HEADROOM_GAIN = float(10.0 ** (MIX_HEADROOM_DB / 20.0))
# Bump whenever the assembled preview gain staging changes.  This keeps old
# partially-trimmed previews out of the render cache.
RENDER_MIX_VERSION = "global-headroom-fx-loudness-v4"
AUTOMATION_FADER_LAW_VERSION = "monotone-cubic-v1"
OVERLAP_CROSSFADE_FLOOR_DB = -6.0
OVERLAP_CROSSFADE_FLOOR_GAIN = float(10.0 ** (OVERLAP_CROSSFADE_FLOOR_DB / 20.0))
TRANSITION_PRESETS = {
    "custom": {},
    "auto": {
        "volume_mode": "overlap-crossfade",
        "eq_mode": "center-bass-swap",
        "filter_mode": "none",
    },
    "fade": {
        "volume_mode": "crossfade",
        "eq_mode": "none",
        "filter_mode": "none",
    },
    "blend": {
        "volume_mode": "overlap",
        "eq_mode": "center-bass-swap",
        "filter_mode": "none",
    },
    "rise": {
        "volume_mode": "smooth-crossfade",
        "eq_mode": "start-bass-swap",
        "filter_mode": "high-pass-filter-in",
    },
    "wave": {
        "volume_mode": "smooth-crossfade",
        "eq_mode": "long-bass-cut",
        "filter_mode": "low-pass-filter-out",
    },
    "cut": {
        "volume_mode": "center-cut",
        "eq_mode": "center-bass-swap",
        "filter_mode": "none",
    },
}
VOLUME_MODES = {
    "crossfade",
    "overlap-crossfade",
    "smooth-crossfade",
    "overlap",
    "fade-in-fade-out",
    "cut-in-fade-out",
    "fade-in-cut-out",
    "center-cut",
}
EQ_MODES = {
    "none",
    "start-bass-swap",
    "end-bass-swap",
    "center-bass-swap",
    "long-bass-cut",
}
FILTER_MODES = {
    "none",
    "low-pass-filter-out",
    "low-pass-filter-in",
    "high-pass-filter-out",
    "high-pass-filter-in",
}
AUTOMATION_CURVES = {"linear", "smooth", "exponential"}
AUTOMATION_LANES = ("volume", "eq_low", "eq_mid", "eq_high", "filter", "delay", "reverb")
AUTOMATION_DECKS = ("from", "to")
AUTOMATION_LIMITS = {
    # Volume uses the same full DAW-style control range as the isolator lanes.
    # The digital mute floor is represented as -96 dB and _db_to_gain turns it
    # into exact silence; the final limiter safely handles the +6 dB endpoint.
    "volume": (-96.0, 6.0),
    "eq_low": (-96.0, 6.0),
    "eq_mid": (-96.0, 6.0),
    "eq_high": (-96.0, 6.0),
    "filter": (-1.0, 1.0),
    "delay": (0.0, 1.0),
    "reverb": (0.0, 1.0),
}
ISOLATOR_LOW_CUTOFF_HZ = 275.0
ISOLATOR_HIGH_CUTOFF_HZ = 3000.0
ISOLATOR_KILL_DB = -96.0
TRUE_PEAK_CEILING_DBTP = -0.3
FILTER_MIN_HZ = 20.0
FILTER_MAX_HZ = 20000.0
FILTER_EDGE_KILL_START = 0.975
FILTER_EDGE_KILL_END = 0.995
DELAY_BEAT_DIVISIONS = (0.125, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0)
DEFAULT_DELAY_BEATS = 0.5
DEFAULT_DELAY_TONE = 0.5
DEFAULT_REVERB_DECAY_SECONDS = 1.2
DEFAULT_REVERB_TONE = 0.5
FADER_ANCHOR_POSITIONS = np.asarray([0.0, 0.25, 0.5, 0.85, 1.0], dtype=np.float64)
FADER_ANCHOR_DB = np.asarray([-96.0, -30.0, -13.0, 0.0, 6.0], dtype=np.float64)
CUE_COLUMN_BY_NAME = {
    "IN 1": "in_1_seconds",
    "IN 2": "in_2_seconds",
    "OUT 1": "out_1_seconds",
    "OUT 2": "out_2_seconds",
}


@dataclass(frozen=True)
class RenderedTransition:
    preview_path: Path
    metadata_path: Path
    waveform_path: Path
    from_track_number: int
    to_track_number: int
    from_title: str
    to_title: str
    from_bpm: float
    to_bpm: float
    from_onset_time: float
    to_onset_time: float
    from_start_seconds: float
    to_start_seconds: float
    duration_seconds: float
    overlap_bars: int
    front_padding_bars: int
    back_padding_bars: int
    from_nudge_beats: float
    to_nudge_beats: float
    from_beatgrid_ms: float
    to_beatgrid_ms: float
    from_pitch_shift: int
    to_pitch_shift: int
    beats_per_bar: int
    target_sample_rate: int
    quality: str
    b_time_stretch_rate: float
    preset: str
    volume_mode: str
    eq_mode: str
    filter_mode: str


@dataclass(frozen=True)
class PreparedLiveTransition:
    """Two tempo-/pitch-corrected deck windows for browser-side mixing."""

    transition_id: str
    from_path: Path
    to_path: Path
    sample_rate: int
    duration_seconds: float
    transport_start_seconds: float
    transport_end_seconds: float
    transition_start_seconds: float
    transition_end_seconds: float
    loop_start_seconds: float
    loop_end_seconds: float
    overlap_bars: int
    front_padding_bars: int
    back_padding_bars: int
    guard_bars: int
    beats_per_bar: int
    from_bpm: float
    to_bpm: float
    b_time_stretch_rate: float


def _slug(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    text = "_".join(part for part in text.split("_") if part)
    return text[:72] or "track"


def _shift_camelot_key(key: str, semitones: int) -> str:
    text = str(key or "").strip().upper()
    if len(text) < 2 or text[-1] not in {"A", "B"}:
        return text
    try:
        number = int(text[:-1])
    except ValueError:
        return text
    if number < 1 or number > 12:
        return text
    shifted = ((number - 1 + (7 * int(semitones))) % 12) + 1
    return f"{shifted}{text[-1]}"


def _track_tokens(track: PlaylistTrack) -> set[str]:
    filename = Path(track.mp3_name)
    values = {
        str(track.track_number),
        track.title,
        track.mp3_name,
        filename.name,
        filename.stem,
    }
    return {v.strip().lower() for v in values if v and v.strip()}


def _read_tracklist_rows(tracklist_csv: Path) -> list[dict[str, str]]:
    return read_csv_rows(tracklist_csv)


def _default_cue_table(tracklist_csv: Path) -> Path:
    stem = tracklist_csv.stem
    if stem.endswith("_tracks_rekordbox"):
        stem = stem[: -len("_tracks_rekordbox")]
    elif stem.endswith("_tracks"):
        stem = stem[: -len("_tracks")]
    return tracklist_csv.with_name(f"{stem}_cues.csv")


def _read_cue_rows(cue_table: Path | None) -> list[dict[str, str]]:
    if cue_table is None or not cue_table.exists():
        return []
    return read_csv_rows(cue_table)


def _find_track(tracks: list[PlaylistTrack], query: str) -> PlaylistTrack:
    needle = query.strip().lower()
    if not needle:
        raise ValueError("Track query cannot be empty.")

    exact = [track for track in tracks if needle in _track_tokens(track)]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        names = ", ".join(f"{t.track_number}:{t.title}" for t in exact[:8])
        raise ValueError(f"Track query '{query}' matched multiple tracks: {names}")

    partial = [
        track
        for track in tracks
        if needle in track.title.lower()
        or needle in track.mp3_name.lower()
        or needle in Path(track.mp3_name).stem.lower()
    ]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        names = ", ".join(f"{t.track_number}:{t.title}" for t in partial[:8])
        raise ValueError(f"Track query '{query}' matched multiple tracks: {names}")

    raise ValueError(f"Could not find track matching '{query}'.")


def _row_for_track(rows: list[dict[str, str]], track: PlaylistTrack) -> dict[str, str]:
    track_tokens = _track_tokens(track)
    for row in rows:
        values = {
            str(row.get("track_number", "")).strip().lower(),
            str(row.get("title", "")).strip().lower(),
            str(row.get("mp3_name", "")).strip().lower(),
            Path(str(row.get("mp3_name", "")).strip()).name.lower(),
            Path(str(row.get("mp3_name", "")).strip()).stem.lower(),
        }
        if track_tokens.intersection(v for v in values if v):
            return row
    return {}


def _cue_seconds_from_table(
    cue_rows: list[dict[str, str]],
    track: PlaylistTrack,
    cue_name: str | None,
) -> float | None:
    if cue_name is None:
        return None
    normalized = " ".join(cue_name.strip().upper().split())
    track_tokens = _track_tokens(track)
    for row in cue_rows:
        row_tokens = {
            str(row.get("track_number", "")).strip().lower(),
            str(row.get("title", "")).strip().lower(),
            str(row.get("mp3_name", "")).strip().lower(),
            Path(str(row.get("mp3_name", "")).strip()).name.lower(),
            Path(str(row.get("mp3_name", "")).strip()).stem.lower(),
        }
        if not track_tokens.intersection(v for v in row_tokens if v):
            continue
        if " ".join(str(row.get("cue_name", "")).strip().upper().split()) == normalized:
            return optional_float(row.get("start_seconds"))
    return None


def _cue_seconds_from_wide_columns(row: dict[str, str], cue_name: str | None) -> float | None:
    if cue_name is None:
        return None
    normalized = " ".join(cue_name.strip().upper().split())
    column = CUE_COLUMN_BY_NAME.get(normalized)
    if column is None:
        return None
    return optional_float(row.get(column))


def _cue_seconds(
    *,
    cue_rows: list[dict[str, str]],
    track: PlaylistTrack,
    row: dict[str, str],
    cue_name: str | None,
) -> float | None:
    if cue_name is None:
        return None
    return _cue_seconds_from_table(cue_rows, track, cue_name) or _cue_seconds_from_wide_columns(row, cue_name)


def _required_float(value: float | None, *, label: str, track: PlaylistTrack) -> float:
    if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{label} must be a positive number for {track.track_number}: {track.title}")
    return float(value)


def _onset_time(track: PlaylistTrack) -> float:
    if track.onset_time is None or not math.isfinite(float(track.onset_time)):
        return 0.0
    return max(0.0, float(track.onset_time))


def _load_segment(
    path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
) -> np.ndarray:
    offset = max(0.0, float(start_seconds))
    duration = max(0.01, float(duration_seconds))
    audio, _ = librosa.load(
        str(path),
        sr=sample_rate,
        mono=False,
        offset=offset,
        duration=duration,
    )
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.shape[0] > audio.shape[1]:
        audio = audio.T
    if audio.size == 0 or audio.shape[1] == 0:
        raise RuntimeError(f"No audio loaded from {path} at {offset:.3f}s")
    return audio


def _load_segment_with_padding(
    path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
) -> np.ndarray:
    leading_seconds = max(0.0, -float(start_seconds))
    offset = max(0.0, float(start_seconds))
    load_duration = max(0.01, float(duration_seconds) - leading_seconds)
    audio = _load_segment(
        path,
        start_seconds=offset,
        duration_seconds=load_duration,
        sample_rate=sample_rate,
    )
    leading_samples = int(round(leading_seconds * sample_rate))
    target_samples = max(1, int(round(float(duration_seconds) * sample_rate)))
    if leading_samples > 0:
        audio = np.pad(audio, ((0, 0), (leading_samples, 0)), mode="constant")
    return _fit_length(audio, target_samples)


def _fit_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
    target_samples = max(1, int(target_samples))
    if audio.shape[1] == target_samples:
        return audio
    if audio.shape[1] > target_samples:
        return audio[:, :target_samples]
    pad = target_samples - audio.shape[1]
    return np.pad(audio, ((0, 0), (0, pad)), mode="constant")


def _time_stretch_multichannel(
    audio: np.ndarray,
    *,
    rate: float,
    target_samples: int,
    sample_rate: int,
    quality: str,
    pitch_shift_semitones: int = 0,
) -> np.ndarray:
    rate = float(rate)
    pitch_shift_semitones = int(pitch_shift_semitones)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"Invalid time-stretch rate: {rate}")
    if abs(rate - 1.0) < 1e-6 and pitch_shift_semitones == 0:
        return _fit_length(audio, target_samples)

    rubberband = shutil.which("rubberband")
    if rubberband is None:
        raise RuntimeError(
            "Rubber Band CLI was not found on PATH. Install it first, for example with "
            "`brew install rubberband` on macOS."
        )

    target_duration = target_samples / float(sample_rate)
    with tempfile.TemporaryDirectory(prefix="dj_transition_rubberband_") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "input.wav"
        output_path = tmp_dir / "output.wav"
        sf.write(str(input_path), audio.T, sample_rate, format="WAV", subtype="FLOAT")
        try:
            subprocess.run(
                [
                    rubberband,
                    "--quiet",
                    str(TRANSITION_QUALITY_PROFILES[quality]["rubberband_flag"]),
                    "--duration",
                    f"{target_duration:.12f}",
                    *(["--pitch", str(pitch_shift_semitones)] if pitch_shift_semitones else []),
                    str(input_path),
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or exc.stdout or "").strip()
            detail = f": {stderr}" if stderr else ""
            raise RuntimeError(f"Rubber Band audio transform failed{detail}") from exc
        stretched, _ = sf.read(str(output_path), dtype="float32", always_2d=True)
    return _fit_length(stretched.T, target_samples)


def _norm_mode(value: str) -> str:
    return "-".join(str(value).strip().lower().replace("_", "-").split())


def _resolve_modes(
    *,
    preset: str,
    volume_mode: str | None,
    eq_mode: str | None,
    filter_mode: str | None,
) -> tuple[str, str, str, str]:
    preset_key = _norm_mode(preset or "custom")
    if preset_key not in TRANSITION_PRESETS:
        valid = ", ".join(sorted(TRANSITION_PRESETS))
        raise ValueError(f"Unsupported preset '{preset}'. Expected one of: {valid}")

    defaults = TRANSITION_PRESETS[preset_key]
    volume = _norm_mode(volume_mode or defaults.get("volume_mode", DEFAULT_VOLUME_MODE))
    eq = _norm_mode(eq_mode or defaults.get("eq_mode", DEFAULT_EQ_MODE))
    filt = _norm_mode(filter_mode or defaults.get("filter_mode", DEFAULT_FILTER_MODE))

    if volume not in VOLUME_MODES:
        raise ValueError(f"Unsupported volume mode '{volume}'. Expected one of: {', '.join(sorted(VOLUME_MODES))}")
    if eq not in EQ_MODES:
        raise ValueError(f"Unsupported EQ mode '{eq}'. Expected one of: {', '.join(sorted(EQ_MODES))}")
    if filt not in FILTER_MODES:
        raise ValueError(f"Unsupported filter mode '{filt}'. Expected one of: {', '.join(sorted(FILTER_MODES))}")
    return preset_key, volume, eq, filt


def _resolve_quality(value: str | None, sample_rate: int | None) -> tuple[str, int]:
    quality = _norm_mode(value or "preview")
    if quality not in TRANSITION_QUALITY_PROFILES:
        valid = ", ".join(TRANSITION_QUALITY_CHOICES)
        raise ValueError(f"Unsupported transition quality '{value}'. Expected one of: {valid}")
    profile_rate = int(TRANSITION_QUALITY_PROFILES[quality]["sample_rate"])
    resolved_rate = profile_rate if sample_rate is None else max(8000, int(sample_rate))
    return quality, resolved_rate


def _smoothstep(x: np.ndarray) -> np.ndarray:
    return (x * x * (3.0 - (2.0 * x))).astype(np.float32)


def _default_lane_points(lane: str, total_beats: float) -> list[dict[str, float | str]]:
    default_value = 0.0
    return [
        {"beat": 0.0, "value": default_value, "curve": "linear"},
        {"beat": float(total_beats), "value": default_value, "curve": "linear"},
    ]


def _normalize_automation(
    automation: Mapping[str, Any] | None,
    *,
    total_beats: float,
) -> dict[str, dict[str, list[dict[str, float | str]]]] | None:
    """Validate and canonicalize explicit transition automation points.

    Explicit automation is intentionally optional so command-line and older API
    callers retain the pre-existing mode-based rendering behavior.
    """
    if automation is None:
        return None
    if not isinstance(automation, Mapping):
        raise ValueError("Automation must be an object containing from/to deck lanes.")

    normalized: dict[str, dict[str, list[dict[str, float | str]]]] = {}
    for deck in AUTOMATION_DECKS:
        raw_deck = automation.get(deck, {})
        if not isinstance(raw_deck, Mapping):
            raise ValueError(f"Automation deck '{deck}' must be an object.")
        deck_lanes: dict[str, list[dict[str, float | str]]] = {}
        for lane in AUTOMATION_LANES:
            raw_points = raw_deck.get(lane)
            if raw_points is None:
                deck_lanes[lane] = _default_lane_points(lane, total_beats)
                continue
            if not isinstance(raw_points, list) or len(raw_points) < 2:
                raise ValueError(f"Automation lane '{deck}.{lane}' needs at least two points.")
            low, high = AUTOMATION_LIMITS[lane]
            points: list[dict[str, float | str]] = []
            for raw_point in raw_points:
                if not isinstance(raw_point, Mapping):
                    raise ValueError(f"Automation point in '{deck}.{lane}' must be an object.")
                try:
                    beat = float(raw_point.get("beat"))
                    value = float(raw_point.get("value"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Automation point in '{deck}.{lane}' needs numeric beat and value.") from exc
                if not math.isfinite(beat) or not math.isfinite(value):
                    raise ValueError(f"Automation point in '{deck}.{lane}' must be finite.")
                if beat < -1e-6 or beat > total_beats + 1e-6:
                    raise ValueError(f"Automation point beat in '{deck}.{lane}' must be inside the overlap.")
                curve = _norm_mode(str(raw_point.get("curve") or "linear"))
                if curve not in AUTOMATION_CURVES:
                    raise ValueError(f"Unsupported automation curve '{curve}'.")
                points.append(
                    {
                        "beat": float(np.clip(beat, 0.0, total_beats)),
                        "value": float(np.clip(value, low, high)),
                        "curve": curve,
                    }
                )
            # Python's sort is stable: the order of equal-time points encodes a
            # deliberate left-to-right discontinuity (before value, after value).
            points.sort(key=lambda point: float(point["beat"]))
            if len(points) < 2:
                raise ValueError(f"Automation lane '{deck}.{lane}' needs at least two points.")
            if abs(float(points[0]["beat"])) > 1e-6 or abs(float(points[-1]["beat"]) - total_beats) > 1e-6:
                raise ValueError(f"Automation lane '{deck}.{lane}' must start at beat 0 and end at the overlap end.")
            group_start = 0
            while group_start < len(points):
                beat = float(points[group_start]["beat"])
                group_end = group_start + 1
                while group_end < len(points) and abs(float(points[group_end]["beat"]) - beat) < 1e-6:
                    points[group_end]["beat"] = beat
                    group_end += 1
                group_size = group_end - group_start
                if group_size > 2:
                    raise ValueError(f"Automation lane '{deck}.{lane}' allows at most two points at one beat.")
                if group_size == 2 and (abs(beat) < 1e-6 or abs(beat - total_beats) < 1e-6):
                    raise ValueError(f"Automation jump in '{deck}.{lane}' must be inside the overlap, not at an endpoint.")
                group_start = group_end
            points[0]["beat"] = 0.0
            points[-1]["beat"] = float(total_beats)
            deck_lanes[lane] = points
        normalized[deck] = deck_lanes
    return normalized


def _curve_progress(x: np.ndarray, curve: str) -> np.ndarray:
    if curve == "linear":
        return x.astype(np.float32)
    if curve == "smooth":
        return _smoothstep(x)
    if curve == "exponential":
        return (np.expm1(4.0 * x) / math.expm1(4.0)).astype(np.float32)
    raise AssertionError(f"Unhandled automation curve: {curve}")


def _fader_slopes() -> np.ndarray:
    """Monotone PCHIP slopes for the editor/renderer fader anchor curve."""
    x = FADER_ANCHOR_POSITIONS
    y = FADER_ANCHOR_DB
    delta = np.diff(y) / np.diff(x)
    slopes = np.empty_like(y)
    slopes[0] = delta[0]
    slopes[-1] = delta[-1]
    for index in range(1, len(y) - 1):
        left, right = delta[index - 1], delta[index]
        if left * right <= 0.0:
            slopes[index] = 0.0
            continue
        left_width = x[index] - x[index - 1]
        right_width = x[index + 1] - x[index]
        first_weight = (2.0 * right_width) + left_width
        second_weight = right_width + (2.0 * left_width)
        slopes[index] = (first_weight + second_weight) / ((first_weight / left) + (second_weight / right))
    return slopes


FADER_ANCHOR_SLOPES = _fader_slopes()


def _fader_position_to_db(position: np.ndarray | float) -> np.ndarray:
    """Map normalized fader position to dB with a smooth monotone cubic."""
    pos = np.clip(np.asarray(position, dtype=np.float64), 0.0, 1.0)
    segment = np.clip(np.searchsorted(FADER_ANCHOR_POSITIONS, pos, side="right") - 1, 0, len(FADER_ANCHOR_POSITIONS) - 2)
    x0, x1 = FADER_ANCHOR_POSITIONS[segment], FADER_ANCHOR_POSITIONS[segment + 1]
    y0, y1 = FADER_ANCHOR_DB[segment], FADER_ANCHOR_DB[segment + 1]
    width = x1 - x0
    t = (pos - x0) / width
    t2, t3 = t * t, t * t * t
    output = (
        ((2.0 * t3) - (3.0 * t2) + 1.0) * y0
        + ((t3 - (2.0 * t2) + t) * width * FADER_ANCHOR_SLOPES[segment])
        + ((-2.0 * t3) + (3.0 * t2)) * y1
        + ((t3 - t2) * width * FADER_ANCHOR_SLOPES[segment + 1])
    )
    return output.astype(np.float32)


def _db_to_fader_position(value_db: np.ndarray | float) -> np.ndarray:
    """Bounded inverse of :func:`_fader_position_to_db` for pointer/DSP parity."""
    values = np.clip(np.asarray(value_db, dtype=np.float64), FADER_ANCHOR_DB[0], FADER_ANCHOR_DB[-1])
    low = np.zeros_like(values)
    high = np.ones_like(values)
    for _ in range(28):
        midpoint = (low + high) * 0.5
        below = _fader_position_to_db(midpoint) < values
        low = np.where(below, midpoint, low)
        high = np.where(below, high, midpoint)
    return ((low + high) * 0.5).astype(np.float32)


def _evaluate_points(
    points: list[dict[str, float | str]],
    *,
    beats: np.ndarray,
    lane: str | None = None,
) -> np.ndarray:
    times = np.asarray([float(point["beat"]) for point in points], dtype=np.float32)
    values = np.asarray([float(point["value"]) for point in points], dtype=np.float32)
    segment_index = np.searchsorted(times, beats, side="right") - 1
    out = np.full(beats.shape, values[0], dtype=np.float32)
    after_last = segment_index >= len(points) - 1
    out[after_last] = values[-1]
    valid = (segment_index >= 0) & (segment_index < len(points) - 1)
    if not np.any(valid):
        return out
    left_index = segment_index[valid]
    start = times[left_index]
    end = times[left_index + 1]
    span = np.maximum(1e-9, end - start)
    local = np.clip((beats[valid] - start) / span, 0.0, 1.0)
    result = values[left_index].copy()
    curves = np.asarray([str(point["curve"]) for point in points], dtype=object)
    for curve in AUTOMATION_CURVES:
        curve_mask = curves[left_index] == curve
        if not np.any(curve_mask):
            continue
        progress = _curve_progress(local[curve_mask], curve)
        indices = left_index[curve_mask]
        if lane == "volume" or (lane or "").startswith("eq_"):
            start_position = _db_to_fader_position(values[indices])
            end_position = _db_to_fader_position(values[indices + 1])
            result[curve_mask] = _fader_position_to_db(start_position + ((end_position - start_position) * progress))
        else:
            result[curve_mask] = values[indices] + ((values[indices + 1] - values[indices]) * progress)
    out[valid] = result
    return out


def _db_to_gain(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    gain = np.power(10.0, arr / 20.0).astype(np.float32)
    gain[arr <= ISOLATOR_KILL_DB + 0.5] = 0.0
    return gain


def _nearest_delay_division(value: float | None) -> float:
    """Resolve a user-facing beat selector to one supported musical division."""
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_DELAY_BEATS
    return min(DELAY_BEAT_DIVISIONS, key=lambda option: abs(option - numeric))


def _fx_settings(
    *,
    from_delay_beats: float | None,
    to_delay_beats: float | None,
    from_delay_tone: float | None,
    to_delay_tone: float | None,
    from_reverb_decay: float | None,
    to_reverb_decay: float | None,
    from_reverb_tone: float | None,
    to_reverb_tone: float | None,
) -> dict[str, dict[str, float]]:
    """Canonical compact DJ-unit settings, shared by cache and DSP paths."""
    def tone(value: float | None, default: float) -> float:
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except (TypeError, ValueError):
            return default

    def decay(value: float | None) -> float:
        try:
            return float(np.clip(float(value), 0.25, 4.0))
        except (TypeError, ValueError):
            return DEFAULT_REVERB_DECAY_SECONDS

    return {
        "from": {
            "delay_beats": _nearest_delay_division(from_delay_beats),
            "delay_tone": tone(from_delay_tone, DEFAULT_DELAY_TONE),
            "reverb_decay": decay(from_reverb_decay),
            "reverb_tone": tone(from_reverb_tone, DEFAULT_REVERB_TONE),
        },
        "to": {
            "delay_beats": _nearest_delay_division(to_delay_beats),
            "delay_tone": tone(to_delay_tone, DEFAULT_DELAY_TONE),
            "reverb_decay": decay(to_reverb_decay),
            "reverb_tone": tone(to_reverb_tone, DEFAULT_REVERB_TONE),
        },
    }


def _normalize_fx_settings(value: Mapping[str, Any] | None) -> dict[str, dict[str, float]]:
    raw = value if isinstance(value, Mapping) else {}
    from_raw = raw.get("from", {}) if isinstance(raw.get("from", {}), Mapping) else {}
    to_raw = raw.get("to", {}) if isinstance(raw.get("to", {}), Mapping) else {}
    return _fx_settings(
        from_delay_beats=from_raw.get("delay_beats"),
        to_delay_beats=to_raw.get("delay_beats"),
        from_delay_tone=from_raw.get("delay_tone"),
        to_delay_tone=to_raw.get("delay_tone"),
        from_reverb_decay=from_raw.get("reverb_decay"),
        to_reverb_decay=to_raw.get("reverb_decay"),
        from_reverb_tone=from_raw.get("reverb_tone"),
        to_reverb_tone=to_raw.get("reverb_tone"),
    )


def _explicit_automation_curves(
    automation: dict[str, dict[str, list[dict[str, float | str]]]],
    *,
    samples: int,
    total_beats: float,
) -> dict[str, Any]:
    beats = np.linspace(0.0, total_beats, samples, endpoint=True, dtype=np.float32)
    values: dict[str, Any] = {"x": beats / max(total_beats, 1e-9), "explicit": True, "spec": automation}
    for deck, prefix in (("from", "a"), ("to", "b")):
        volume_db = _evaluate_points(automation[deck]["volume"], beats=beats, lane="volume")
        values[f"{prefix}_gain"] = _db_to_gain(volume_db)
        values[f"{prefix}_volume_db"] = volume_db
        for band in ("low", "mid", "high"):
            lane = f"eq_{band}"
            values[f"{prefix}_eq_{band}_db"] = _evaluate_points(automation[deck][lane], beats=beats, lane=lane)
        values[f"{prefix}_filter_amount"] = _evaluate_points(automation[deck]["filter"], beats=beats, lane="filter")
        values[f"{prefix}_delay_depth"] = _evaluate_points(automation[deck]["delay"], beats=beats, lane="delay")
        values[f"{prefix}_reverb_depth"] = _evaluate_points(automation[deck]["reverb"], beats=beats, lane="reverb")
    return values


def _automation_curves(
    *,
    samples: int,
    sample_rate: int,
    volume_mode: str,
    eq_mode: str,
    filter_mode: str,
) -> dict[str, np.ndarray | str | float | None]:
    x = np.linspace(0.0, 1.0, samples, endpoint=True, dtype=np.float32)
    half = x < 0.5
    silence = np.float32(0.0)
    unity = np.float32(1.0)

    if volume_mode == "crossfade":
        a_gain = np.cos(x * np.pi * 0.5).astype(np.float32)
        b_gain = np.sin(x * np.pi * 0.5).astype(np.float32)
    elif volume_mode == "overlap-crossfade":
        floor = np.float32(OVERLAP_CROSSFADE_FLOOR_GAIN)
        a_gain = (floor + ((1.0 - floor) * np.cos(x * np.pi * 0.5))).astype(np.float32)
        b_gain = (floor + ((1.0 - floor) * np.sin(x * np.pi * 0.5))).astype(np.float32)
    elif volume_mode == "smooth-crossfade":
        t = _smoothstep(x)
        a_gain = np.sqrt(np.maximum(1.0 - t, 0.0)).astype(np.float32)
        b_gain = np.sqrt(np.maximum(t, 0.0)).astype(np.float32)
    elif volume_mode == "overlap":
        a_gain = np.ones(samples, dtype=np.float32)
        b_gain = np.ones(samples, dtype=np.float32)
    elif volume_mode == "fade-in-fade-out":
        a_gain = np.where(half, unity, 1.0 - ((x - 0.5) * 2.0)).astype(np.float32)
        b_gain = np.where(half, x * 2.0, unity).astype(np.float32)
    elif volume_mode == "cut-in-fade-out":
        a_gain = np.where(half, unity, 1.0 - ((x - 0.5) * 2.0)).astype(np.float32)
        b_gain = np.ones(samples, dtype=np.float32)
    elif volume_mode == "fade-in-cut-out":
        a_gain = np.ones(samples, dtype=np.float32)
        b_gain = np.where(half, x * 2.0, unity).astype(np.float32)
    elif volume_mode == "center-cut":
        a_gain = np.where(half, unity, silence).astype(np.float32)
        b_gain = np.where(half, silence, unity).astype(np.float32)
    else:
        raise AssertionError(f"Unhandled volume mode: {volume_mode}")

    low_cut_gain = np.float32(0.15)
    a_low_gain = np.ones(samples, dtype=np.float32)
    b_low_gain = np.ones(samples, dtype=np.float32)
    if eq_mode == "start-bass-swap":
        a_low_gain[:] = low_cut_gain
    elif eq_mode == "end-bass-swap":
        b_low_gain[:] = low_cut_gain
    elif eq_mode == "center-bass-swap":
        a_low_gain = np.where(half, unity, silence).astype(np.float32)
        b_low_gain = np.where(half, silence, unity).astype(np.float32)
    elif eq_mode == "long-bass-cut":
        a_low_gain[:] = low_cut_gain
        b_low_gain[:] = low_cut_gain
    elif eq_mode != "none":
        raise AssertionError(f"Unhandled EQ mode: {eq_mode}")

    low_cutoff = 180.0
    min_cutoff = 320.0
    max_cutoff = min(18000.0, (sample_rate * 0.5) - 100.0)
    min_hp = 30.0
    max_hp = min(2400.0, (sample_rate * 0.5) - 100.0)
    filter_a_type: str | None = None
    filter_b_type: str | None = None
    filter_a_cutoff: np.ndarray | None = None
    filter_b_cutoff: np.ndarray | None = None

    if filter_mode == "low-pass-filter-out":
        filter_a_type = "lowpass"
        filter_a_cutoff = (max_cutoff - ((max_cutoff - min_cutoff) * x)).astype(np.float32)
    elif filter_mode == "low-pass-filter-in":
        filter_b_type = "lowpass"
        filter_b_cutoff = (min_cutoff + ((max_cutoff - min_cutoff) * x)).astype(np.float32)
    elif filter_mode == "high-pass-filter-out":
        filter_a_type = "highpass"
        filter_a_cutoff = (min_hp + ((max_hp - min_hp) * x)).astype(np.float32)
    elif filter_mode == "high-pass-filter-in":
        filter_b_type = "highpass"
        filter_b_cutoff = (max_hp - ((max_hp - min_hp) * x)).astype(np.float32)
    elif filter_mode != "none":
        raise AssertionError(f"Unhandled filter mode: {filter_mode}")

    return {
        "x": x,
        "a_gain": a_gain,
        "b_gain": b_gain,
        "a_low_gain": a_low_gain,
        "b_low_gain": b_low_gain,
        "low_cutoff": low_cutoff,
        "filter_a_type": filter_a_type,
        "filter_b_type": filter_b_type,
        "filter_a_cutoff": filter_a_cutoff,
        "filter_b_cutoff": filter_b_cutoff,
        "a_delay_depth": np.zeros(samples, dtype=np.float32),
        "b_delay_depth": np.zeros(samples, dtype=np.float32),
        "a_reverb_depth": np.zeros(samples, dtype=np.float32),
        "b_reverb_depth": np.zeros(samples, dtype=np.float32),
    }


def _apply_low_band_gain(audio: np.ndarray, *, sample_rate: int, cutoff_hz: float, low_gain: np.ndarray) -> np.ndarray:
    if np.allclose(low_gain, 1.0):
        return audio
    sos = signal.butter(4, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    low = signal.sosfiltfilt(sos, audio, axis=1).astype(np.float32)
    high = audio - low
    return (high + (low * low_gain[np.newaxis, :])).astype(np.float32)


def _apply_variable_filter(
    audio: np.ndarray,
    *,
    sample_rate: int,
    filter_type: str | None,
    cutoff: np.ndarray | None,
    block_size: int = 4096,
) -> np.ndarray:
    if filter_type is None or cutoff is None:
        return audio
    samples = audio.shape[1]
    if samples <= 1:
        return audio

    hop = max(256, block_size // 2)
    window = signal.windows.hann(block_size, sym=False).astype(np.float32)
    out = np.zeros_like(audio, dtype=np.float32)
    weight = np.zeros(samples, dtype=np.float32)
    starts = list(range(0, samples, hop))
    if starts[-1] != max(0, samples - block_size):
        starts.append(max(0, samples - block_size))

    for start in starts:
        end = min(samples, start + block_size)
        block = audio[:, start:end]
        win = window[: end - start]
        center = min(samples - 1, start + ((end - start) // 2))
        cutoff_hz = float(np.clip(cutoff[center], 20.0, (sample_rate * 0.5) - 100.0))
        sos = signal.butter(2, cutoff_hz, btype=filter_type, fs=sample_rate, output="sos")
        filtered = signal.sosfilt(sos, block, axis=1).astype(np.float32)
        out[:, start:end] += filtered * win[np.newaxis, :]
        weight[start:end] += win

    weight = np.maximum(weight, 1e-6)
    return (out / weight[np.newaxis, :]).astype(np.float32)


def _safe_zero_phase_filter(sos: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """Use zero-phase filtering where the clip is long enough for its padding."""
    if audio.shape[1] < 32:
        return signal.sosfilt(sos, audio, axis=1).astype(np.float32)
    return signal.sosfiltfilt(sos, audio, axis=1).astype(np.float32)


def _apply_isolator_eq(
    audio: np.ndarray,
    *,
    sample_rate: int,
    low_db: np.ndarray,
    mid_db: np.ndarray,
    high_db: np.ndarray,
) -> np.ndarray:
    if all(np.allclose(values, 0.0) for values in (low_db, mid_db, high_db)):
        return audio
    nyquist_safe = max(40.0, (sample_rate * 0.5) - 100.0)
    low_cutoff = min(ISOLATOR_LOW_CUTOFF_HZ, nyquist_safe * 0.45)
    high_cutoff = min(ISOLATOR_HIGH_CUTOFF_HZ, nyquist_safe * 0.9)
    if high_cutoff <= low_cutoff:
        high_cutoff = min(nyquist_safe, low_cutoff * 2.0)
    # The residual mid band gives an exactly flat summed response at 0 dB.
    # Fourth-order Butterworth sections provide the intended 24 dB/octave split.
    low_sos = signal.butter(4, low_cutoff, btype="lowpass", fs=sample_rate, output="sos")
    high_sos = signal.butter(4, high_cutoff, btype="highpass", fs=sample_rate, output="sos")
    low = _safe_zero_phase_filter(low_sos, audio)
    high = _safe_zero_phase_filter(high_sos, audio)
    mid = (audio - low - high).astype(np.float32)
    return (
        (low * _db_to_gain(low_db)[np.newaxis, :])
        + (mid * _db_to_gain(mid_db)[np.newaxis, :])
        + (high * _db_to_gain(high_db)[np.newaxis, :])
    ).astype(np.float32)


def _apply_bipolar_filter(
    audio: np.ndarray,
    *,
    sample_rate: int,
    amount: np.ndarray,
    block_size: int = 4096,
) -> np.ndarray:
    """Apply a dry-centred DJ filter with smooth block overlap."""
    if np.allclose(amount, 0.0) or audio.shape[1] <= 1:
        return audio
    samples = audio.shape[1]
    hop = max(256, block_size // 2)
    window = signal.windows.hann(block_size, sym=False).astype(np.float32)
    out = np.zeros_like(audio, dtype=np.float32)
    weight = np.zeros(samples, dtype=np.float32)
    starts = list(range(0, samples, hop))
    if starts[-1] != max(0, samples - block_size):
        starts.append(max(0, samples - block_size))
    nyquist_safe_max_hz = max(FILTER_MIN_HZ, min(FILTER_MAX_HZ, (sample_rate * 0.5) - 100.0))
    for start in starts:
        end = min(samples, start + block_size)
        block = audio[:, start:end]
        win = window[: end - start]
        center = min(samples - 1, start + ((end - start) // 2))
        signed_amount = float(np.clip(amount[center], -1.0, 1.0))
        wet = abs(signed_amount)
        if wet < 1e-4:
            processed = block
        else:
            if signed_amount < 0:
                cutoff_hz = FILTER_MAX_HZ * ((FILTER_MIN_HZ / FILTER_MAX_HZ) ** wet)
                filter_type = "lowpass"
            else:
                cutoff_hz = FILTER_MIN_HZ * ((nyquist_safe_max_hz / FILTER_MIN_HZ) ** wet)
                filter_type = "highpass"
            cutoff_hz = float(np.clip(cutoff_hz, FILTER_MIN_HZ, nyquist_safe_max_hz))
            sos = signal.butter(2, cutoff_hz, btype=filter_type, fs=sample_rate, output="sos")
            filtered = signal.sosfilt(sos, block, axis=1).astype(np.float32)
            processed = ((1.0 - wet) * block) + (wet * filtered)
        out[:, start:end] += processed * win[np.newaxis, :]
        weight[start:end] += win
    result = (out / np.maximum(weight, 1e-6)[np.newaxis, :]).astype(np.float32)
    # Fully closed DJ filters should be a true kill, not merely the small
    # residual left by a practical IIR filter.  The narrow smooth taper avoids
    # a zipper/artifact at the endpoint during continuous automation.
    magnitude = np.abs(np.asarray(amount, dtype=np.float32))
    edge = np.clip((magnitude - FILTER_EDGE_KILL_START) / (FILTER_EDGE_KILL_END - FILTER_EDGE_KILL_START), 0.0, 1.0)
    edge_gain = 1.0 - _smoothstep(edge)
    edge_gain[magnitude >= FILTER_EDGE_KILL_END] = 0.0
    return (result * edge_gain[np.newaxis, :]).astype(np.float32)


def _apply_mix_headroom(audio: np.ndarray) -> np.ndarray:
    """Apply the fixed preview mix trim after all padded sections are assembled."""
    return (np.asarray(audio, dtype=np.float32) * MIX_HEADROOM_GAIN).astype(np.float32)


def _apply_final_limiter(audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
    """Offline linked soft limiter using a 4x oversampled peak detector."""
    if audio.size == 0:
        return audio.astype(np.float32)
    ceiling = float(10.0 ** (TRUE_PEAK_CEILING_DBTP / 20.0))
    oversampled = signal.resample_poly(audio, up=4, down=1, axis=1).astype(np.float32)
    detector = np.max(np.abs(oversampled), axis=0)
    desired = np.minimum(1.0, ceiling / np.maximum(detector, 1e-9)).astype(np.float32)
    lookahead = max(1, int(round(sample_rate * 4 * 0.003)))
    release = 1.0 - math.exp(-1.0 / max(1.0, sample_rate * 4 * 0.030))
    gain_over = np.ones_like(desired, dtype=np.float32)
    gain = 1.0
    for index in range(desired.size):
        target = float(np.min(desired[index : min(desired.size, index + lookahead + 1)]))
        if target < gain:
            gain = target
        else:
            gain = min(target, gain + ((1.0 - gain) * release))
        gain_over[index] = gain
    gain = signal.resample_poly(gain_over, up=1, down=4).astype(np.float32)
    gain = _fit_length(gain[np.newaxis, :], audio.shape[1])[0]
    limited = (audio * gain[np.newaxis, :]).astype(np.float32)
    knee = ceiling * 0.90
    magnitude = np.abs(limited)
    excess = np.maximum(magnitude - knee, 0.0)
    softened = knee + ((ceiling - knee) * (1.0 - np.exp(-excess / max(ceiling - knee, 1e-9))))
    limited = np.sign(limited) * np.where(magnitude > knee, softened, magnitude)
    return np.clip(limited, -ceiling, ceiling).astype(np.float32)


def _apply_tempo_delay(
    audio: np.ndarray,
    *,
    sample_rate: int,
    depth: np.ndarray,
    beat_seconds: float,
    beats: float,
    tone: float,
) -> np.ndarray:
    """Bounded post-fader echo with a musically quantized delay time."""
    dry = np.asarray(audio, dtype=np.float32)
    amount = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)
    if dry.size == 0 or float(np.max(amount)) <= 1e-5:
        return dry
    delay_samples = max(1, int(round(max(0.001, beat_seconds * beats) * sample_rate)))
    repeats = min(8, max(1, (dry.shape[1] - 1) // delay_samples))
    returned = np.zeros_like(dry)
    nyquist = sample_rate * 0.5
    low_cut = min(180.0, nyquist - 100.0)
    high_cut = float(np.clip(1800.0 * math.pow(6.0, float(tone)), low_cut + 100.0, nyquist - 100.0))
    high_sos = signal.butter(2, high_cut, btype="lowpass", fs=sample_rate, output="sos")
    low_sos = signal.butter(2, low_cut, btype="highpass", fs=sample_rate, output="sos")
    filtered = signal.sosfilt(low_sos, signal.sosfilt(high_sos, dry, axis=1), axis=1).astype(np.float32)
    feedback = 0.18 + (0.38 * amount)
    for repeat in range(1, repeats + 1):
        offset = repeat * delay_samples
        if offset >= dry.shape[1]:
            break
        returned[:, offset:] += filtered[:, :-offset] * np.power(feedback[offset:], repeat).astype(np.float32)[np.newaxis, :]
    return (dry + (returned * (0.52 * amount)[np.newaxis, :])).astype(np.float32)


def _apply_schroeder_reverb(
    audio: np.ndarray,
    *,
    sample_rate: int,
    depth: np.ndarray,
    decay_seconds: float,
    tone: float,
) -> np.ndarray:
    """Small deterministic Schroeder-like reverb return for verified renders."""
    dry = np.asarray(audio, dtype=np.float32)
    amount = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)
    if dry.size == 0 or float(np.max(amount)) <= 1e-5:
        return dry
    decay = float(np.clip(decay_seconds, 0.25, 4.0))
    length = max(1, min(dry.shape[1], int(round(decay * sample_rate))))
    impulse = np.zeros(length, dtype=np.float32)
    for seconds in (0.0297, 0.0371, 0.0411, 0.0437):
        step = max(1, int(round(seconds * sample_rate)))
        positions = np.arange(step, length, step, dtype=np.int64)
        if positions.size:
            impulse[positions] += (0.32 / 4.0) * np.exp((-3.0 * positions) / max(1.0, decay * sample_rate)).astype(np.float32)
    nyquist = sample_rate * 0.5
    cutoff = float(np.clip(1400.0 * math.pow(7.5, float(tone)), 300.0, nyquist - 100.0))
    sos = signal.butter(2, cutoff, btype="lowpass", fs=sample_rate, output="sos")
    impulse = signal.sosfilt(sos, impulse).astype(np.float32)
    returned = np.stack([signal.fftconvolve(channel, impulse, mode="full")[: dry.shape[1]] for channel in dry]).astype(np.float32)
    return (dry + (returned * (0.46 * amount)[np.newaxis, :])).astype(np.float32)


def _render_mix(
    a_audio: np.ndarray,
    b_audio: np.ndarray,
    *,
    sample_rate: int,
    volume_mode: str,
    eq_mode: str,
    filter_mode: str,
    automation_spec: dict[str, dict[str, list[dict[str, float | str]]]] | None = None,
    total_beats: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    channels = max(a_audio.shape[0], b_audio.shape[0])
    samples = min(a_audio.shape[1], b_audio.shape[1])
    a_audio = _fit_length(a_audio, samples)
    b_audio = _fit_length(b_audio, samples)
    if a_audio.shape[0] != channels:
        a_audio = np.repeat(a_audio[:1, :], channels, axis=0)
    if b_audio.shape[0] != channels:
        b_audio = np.repeat(b_audio[:1, :], channels, axis=0)

    if automation_spec is None:
        automation = _automation_curves(
            samples=samples,
            sample_rate=sample_rate,
            volume_mode=volume_mode,
            eq_mode=eq_mode,
            filter_mode=filter_mode,
        )
        a_processed = _apply_low_band_gain(a_audio, sample_rate=sample_rate, cutoff_hz=float(automation["low_cutoff"]), low_gain=np.asarray(automation["a_low_gain"], dtype=np.float32))
        b_processed = _apply_low_band_gain(b_audio, sample_rate=sample_rate, cutoff_hz=float(automation["low_cutoff"]), low_gain=np.asarray(automation["b_low_gain"], dtype=np.float32))
        a_processed = _apply_variable_filter(a_processed, sample_rate=sample_rate, filter_type=automation["filter_a_type"] if isinstance(automation["filter_a_type"], str) else None, cutoff=automation["filter_a_cutoff"] if isinstance(automation["filter_a_cutoff"], np.ndarray) else None)
        b_processed = _apply_variable_filter(b_processed, sample_rate=sample_rate, filter_type=automation["filter_b_type"] if isinstance(automation["filter_b_type"], str) else None, cutoff=automation["filter_b_cutoff"] if isinstance(automation["filter_b_cutoff"], np.ndarray) else None)
    else:
        automation = _explicit_automation_curves(automation_spec, samples=samples, total_beats=total_beats)
        a_processed = _apply_isolator_eq(a_audio, sample_rate=sample_rate, low_db=np.asarray(automation["a_eq_low_db"]), mid_db=np.asarray(automation["a_eq_mid_db"]), high_db=np.asarray(automation["a_eq_high_db"]))
        b_processed = _apply_isolator_eq(b_audio, sample_rate=sample_rate, low_db=np.asarray(automation["b_eq_low_db"]), mid_db=np.asarray(automation["b_eq_mid_db"]), high_db=np.asarray(automation["b_eq_high_db"]))
        a_processed = _apply_bipolar_filter(a_processed, sample_rate=sample_rate, amount=np.asarray(automation["a_filter_amount"]))
        b_processed = _apply_bipolar_filter(b_processed, sample_rate=sample_rate, amount=np.asarray(automation["b_filter_amount"]))

    a_gain = np.asarray(automation["a_gain"], dtype=np.float32)
    b_gain = np.asarray(automation["b_gain"], dtype=np.float32)
    a_stem = (a_processed * a_gain[np.newaxis, :]).astype(np.float32)
    b_stem = (b_processed * b_gain[np.newaxis, :]).astype(np.float32)
    mixed = a_stem + b_stem
    return mixed.astype(np.float32), automation, a_stem, b_stem


def _pad_array(values: np.ndarray | None, *, front_samples: int, back_samples: int, front_value: float, back_value: float) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    front = np.full(front_samples, front_value, dtype=np.float32)
    back = np.full(back_samples, back_value, dtype=np.float32)
    return np.concatenate([front, arr, back]).astype(np.float32)


def _pad_automation(
    automation: dict[str, Any],
    *,
    front_samples: int,
    back_samples: int,
) -> dict[str, np.ndarray | str | float | None]:
    if automation.get("explicit"):
        padded: dict[str, Any] = {**automation}
        for prefix in ("a", "b"):
            gain = np.asarray(automation[f"{prefix}_gain"], dtype=np.float32)
            volume_db = np.asarray(automation[f"{prefix}_volume_db"], dtype=np.float32)
            padded[f"{prefix}_gain"] = _pad_array(gain, front_samples=front_samples, back_samples=back_samples, front_value=float(gain[0]), back_value=float(gain[-1]))
            padded[f"{prefix}_volume_db"] = _pad_array(volume_db, front_samples=front_samples, back_samples=back_samples, front_value=float(volume_db[0]), back_value=float(volume_db[-1]))
            for band in ("low", "mid", "high"):
                padded[f"{prefix}_eq_{band}_db"] = _pad_array(np.asarray(automation[f"{prefix}_eq_{band}_db"], dtype=np.float32), front_samples=front_samples, back_samples=back_samples, front_value=0.0, back_value=0.0)
            padded[f"{prefix}_filter_amount"] = _pad_array(np.asarray(automation[f"{prefix}_filter_amount"], dtype=np.float32), front_samples=front_samples, back_samples=back_samples, front_value=0.0, back_value=0.0)
            padded[f"{prefix}_delay_depth"] = _pad_array(np.asarray(automation[f"{prefix}_delay_depth"], dtype=np.float32), front_samples=front_samples, back_samples=back_samples, front_value=float(automation[f"{prefix}_delay_depth"][0]), back_value=float(automation[f"{prefix}_delay_depth"][-1]))
            padded[f"{prefix}_reverb_depth"] = _pad_array(np.asarray(automation[f"{prefix}_reverb_depth"], dtype=np.float32), front_samples=front_samples, back_samples=back_samples, front_value=float(automation[f"{prefix}_reverb_depth"][0]), back_value=float(automation[f"{prefix}_reverb_depth"][-1]))
        return padded
    filter_a = automation["filter_a_cutoff"] if isinstance(automation["filter_a_cutoff"], np.ndarray) else None
    filter_b = automation["filter_b_cutoff"] if isinstance(automation["filter_b_cutoff"], np.ndarray) else None
    filter_a_front = float(filter_a[0]) if filter_a is not None and filter_a.size else 0.0
    filter_a_back = float(filter_a[-1]) if filter_a is not None and filter_a.size else 0.0
    filter_b_front = float(filter_b[0]) if filter_b is not None and filter_b.size else 0.0
    filter_b_back = float(filter_b[-1]) if filter_b is not None and filter_b.size else 0.0
    a_gain = np.asarray(automation["a_gain"], dtype=np.float32)
    b_gain = np.asarray(automation["b_gain"], dtype=np.float32)
    return {
        **automation,
        "a_gain": _pad_array(a_gain, front_samples=front_samples, back_samples=back_samples, front_value=float(a_gain[0]), back_value=float(a_gain[-1])),
        "b_gain": _pad_array(b_gain, front_samples=front_samples, back_samples=back_samples, front_value=float(b_gain[0]), back_value=float(b_gain[-1])),
        "a_low_gain": _pad_array(np.asarray(automation["a_low_gain"], dtype=np.float32), front_samples=front_samples, back_samples=back_samples, front_value=1.0, back_value=1.0),
        "b_low_gain": _pad_array(np.asarray(automation["b_low_gain"], dtype=np.float32), front_samples=front_samples, back_samples=back_samples, front_value=1.0, back_value=1.0),
        "filter_a_cutoff": _pad_array(filter_a, front_samples=front_samples, back_samples=back_samples, front_value=filter_a_front, back_value=filter_a_back),
        "filter_b_cutoff": _pad_array(filter_b, front_samples=front_samples, back_samples=back_samples, front_value=filter_b_front, back_value=filter_b_back),
        "a_delay_depth": _pad_array(np.asarray(automation["a_delay_depth"], dtype=np.float32), front_samples=front_samples, back_samples=back_samples, front_value=0.0, back_value=0.0),
        "b_delay_depth": _pad_array(np.asarray(automation["b_delay_depth"], dtype=np.float32), front_samples=front_samples, back_samples=back_samples, front_value=0.0, back_value=0.0),
        "a_reverb_depth": _pad_array(np.asarray(automation["a_reverb_depth"], dtype=np.float32), front_samples=front_samples, back_samples=back_samples, front_value=0.0, back_value=0.0),
        "b_reverb_depth": _pad_array(np.asarray(automation["b_reverb_depth"], dtype=np.float32), front_samples=front_samples, back_samples=back_samples, front_value=0.0, back_value=0.0),
    }


def _band_rms_series(audio: np.ndarray, *, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mono = np.mean(audio, axis=0).astype(np.float32)
    if mono.size < 16:
        zeros = np.zeros_like(mono, dtype=np.float32)
        return zeros, zeros, zeros
    nyquist = sample_rate * 0.5
    low_cut = min(250.0, nyquist - 100.0)
    high_cut = min(4000.0, nyquist - 100.0)
    low_sos = signal.butter(4, low_cut, btype="lowpass", fs=sample_rate, output="sos")
    high_sos = signal.butter(4, high_cut, btype="highpass", fs=sample_rate, output="sos")
    low = signal.sosfiltfilt(low_sos, mono).astype(np.float32)
    high = signal.sosfiltfilt(high_sos, mono).astype(np.float32)
    mid = (mono - low - high).astype(np.float32)
    return low, mid, high


def _peaks(audio: np.ndarray, *, bins: int, sample_rate: int) -> list[dict[str, float]]:
    mono = np.mean(audio, axis=0).astype(np.float32)
    low, mid, high = _band_rms_series(audio, sample_rate=sample_rate)
    samples = mono.shape[0]
    bins = max(1, min(int(bins), samples))
    edges = np.linspace(0, samples, bins + 1, dtype=np.int64)
    raw: list[dict[str, float]] = []
    for i in range(bins):
        chunk = mono[edges[i] : edges[i + 1]]
        if chunk.size == 0:
            mn = mx = rms = low_rms = mid_rms = high_rms = 0.0
        else:
            mn = float(np.min(chunk))
            mx = float(np.max(chunk))
            rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
            low_chunk = low[edges[i] : edges[i + 1]]
            mid_chunk = mid[edges[i] : edges[i + 1]]
            high_chunk = high[edges[i] : edges[i + 1]]
            low_rms = float(np.sqrt(np.mean(np.square(low_chunk, dtype=np.float64))))
            mid_rms = float(np.sqrt(np.mean(np.square(mid_chunk, dtype=np.float64))))
            high_rms = float(np.sqrt(np.mean(np.square(high_chunk, dtype=np.float64))))
        total = low_rms + mid_rms + high_rms + 1e-9
        raw.append(
            {
                "t": float(i / max(1, bins - 1)),
                "min": mn,
                "max": mx,
                "rms": rms,
                "low": low_rms,
                "mid": mid_rms,
                "high": high_rms,
                "low_ratio": float(low_rms / total),
                "mid_ratio": float(mid_rms / total),
                "high_ratio": float(high_rms / total),
            }
        )

    def band_scale(key: str) -> float:
        vals = np.asarray([row[key] for row in raw], dtype=np.float32)
        positive = vals[vals > 0]
        if positive.size == 0:
            return 1.0
        return float(max(np.percentile(positive, 97.0), 1e-6))

    low_scale = band_scale("low")
    mid_scale = band_scale("mid")
    high_scale = band_scale("high")
    for row in raw:
        row["low_amp"] = float(np.clip((row["low"] / low_scale) ** 0.55, 0.0, 1.0))
        row["mid_amp"] = float(np.clip((row["mid"] / mid_scale) ** 0.55, 0.0, 1.0))
        row["high_amp"] = float(np.clip((row["high"] / high_scale) ** 0.50, 0.0, 1.0))
    return raw


def _downsample_line(values: np.ndarray | None, *, bins: int, normalize_cutoff: bool = False, sample_rate: int = 44100) -> list[dict[str, float | None]]:
    if values is None:
        return []
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return []
    bins = max(1, min(int(bins), arr.size))
    idxs = np.linspace(0, arr.size - 1, bins, dtype=np.int64)
    out: list[dict[str, float | None]] = []
    min_hz = 20.0
    max_hz = max(min_hz + 1.0, (sample_rate * 0.5) - 100.0)
    log_denom = math.log(max_hz / min_hz)
    for i, idx in enumerate(idxs):
        value = float(arr[idx])
        if normalize_cutoff:
            clamped = float(np.clip(value, min_hz, max_hz))
            normalized = math.log(clamped / min_hz) / log_denom
        else:
            normalized = value
        out.append(
            {
                "t": float(i / max(1, bins - 1)),
                "value": value,
                "normalized": float(np.clip(normalized, 0.0, 1.0)),
            }
        )
    return out


def _write_waveform_json(
    *,
    path: Path,
    a_audio: np.ndarray,
    b_audio: np.ndarray,
    automation: dict[str, Any],
    sample_rate: int,
    duration_seconds: float,
    transition_start_seconds: float,
    transition_duration_seconds: float,
    front_padding_bars: int,
    back_padding_bars: int,
    overlap_bars: int,
    bins: int = 1400,
) -> None:
    transition_start_t = transition_start_seconds / duration_seconds if duration_seconds > 0 else 0.0
    transition_end_t = (transition_start_seconds + transition_duration_seconds) / duration_seconds if duration_seconds > 0 else 1.0
    a_gain = np.asarray(automation["a_gain"], dtype=np.float32)
    b_gain = np.asarray(automation["b_gain"], dtype=np.float32)
    explicit = bool(automation.get("explicit"))
    neutral_cutoff = min(18000.0, (sample_rate * 0.5) - 100.0)
    filter_a_cutoff = automation.get("filter_a_cutoff") if isinstance(automation.get("filter_a_cutoff"), np.ndarray) else np.full(a_gain.shape, neutral_cutoff, dtype=np.float32)
    filter_b_cutoff = automation.get("filter_b_cutoff") if isinstance(automation.get("filter_b_cutoff"), np.ndarray) else np.full(b_gain.shape, neutral_cutoff, dtype=np.float32)
    payload = {
        "version": 2,
        "layout": "stacked-aligned",
        "sample_rate": sample_rate,
        "duration_seconds": duration_seconds,
        "timeline": {
            "transition_start_seconds": transition_start_seconds,
            "transition_end_seconds": transition_start_seconds + transition_duration_seconds,
            "transition_start_t": float(np.clip(transition_start_t, 0.0, 1.0)),
            "transition_end_t": float(np.clip(transition_end_t, 0.0, 1.0)),
            "transition_duration_seconds": transition_duration_seconds,
            "front_padding_bars": front_padding_bars,
            "back_padding_bars": back_padding_bars,
            "overlap_bars": overlap_bars,
            "total_bars": front_padding_bars + overlap_bars + back_padding_bars,
        },
        "waveforms": {
            "outgoing": _peaks(a_audio, bins=bins, sample_rate=sample_rate),
            "incoming": _peaks(b_audio, bins=bins, sample_rate=sample_rate),
        },
        "automation": {},
    }
    if explicit:
        payload["automation"] = {
            "explicit": True,
            "volume": {
                "outgoing": _downsample_line(np.asarray(automation["a_volume_db"], dtype=np.float32), bins=bins),
                "incoming": _downsample_line(np.asarray(automation["b_volume_db"], dtype=np.float32), bins=bins),
            },
            "eq": {
                "type": "isolator",
                "crossovers_hz": [ISOLATOR_LOW_CUTOFF_HZ, ISOLATOR_HIGH_CUTOFF_HZ],
                "outgoing": {band: _downsample_line(np.asarray(automation[f"a_eq_{band}_db"], dtype=np.float32), bins=bins) for band in ("low", "mid", "high")},
                "incoming": {band: _downsample_line(np.asarray(automation[f"b_eq_{band}_db"], dtype=np.float32), bins=bins) for band in ("low", "mid", "high")},
            },
            "filter": {
                "outgoing": _downsample_line(np.asarray(automation["a_filter_amount"], dtype=np.float32), bins=bins),
                "incoming": _downsample_line(np.asarray(automation["b_filter_amount"], dtype=np.float32), bins=bins),
            },
            "delay": {
                "outgoing": _downsample_line(np.asarray(automation["a_delay_depth"], dtype=np.float32), bins=bins),
                "incoming": _downsample_line(np.asarray(automation["b_delay_depth"], dtype=np.float32), bins=bins),
            },
            "reverb": {
                "outgoing": _downsample_line(np.asarray(automation["a_reverb_depth"], dtype=np.float32), bins=bins),
                "incoming": _downsample_line(np.asarray(automation["b_reverb_depth"], dtype=np.float32), bins=bins),
            },
            "points": automation.get("spec"),
        }
    else:
        payload["automation"] = {
            "volume": {"outgoing": _downsample_line(a_gain, bins=bins), "incoming": _downsample_line(b_gain, bins=bins)},
            "eq": {
                "low_band_cutoff_hz": float(automation["low_cutoff"]),
                "outgoing_low_gain": _downsample_line(np.asarray(automation["a_low_gain"], dtype=np.float32), bins=bins),
                "incoming_low_gain": _downsample_line(np.asarray(automation["b_low_gain"], dtype=np.float32), bins=bins),
            },
            "filter": {
                "outgoing_type": automation["filter_a_type"],
                "incoming_type": automation["filter_b_type"],
                "outgoing_cutoff_hz": _downsample_line(filter_a_cutoff, bins=bins, normalize_cutoff=True, sample_rate=sample_rate),
                "incoming_cutoff_hz": _downsample_line(filter_b_cutoff, bins=bins, normalize_cutoff=True, sample_rate=sample_rate),
            },
        }
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def _transition_key(
    *,
    from_track: PlaylistTrack,
    to_track: PlaylistTrack,
    overlap_bars: int,
    front_padding_bars: int,
    back_padding_bars: int,
    from_nudge_beats: float,
    to_nudge_beats: float,
    from_beatgrid_ms: float,
    to_beatgrid_ms: float,
    from_pitch_shift: int,
    to_pitch_shift: int,
    from_bar: float,
    to_bar: float,
    from_cue: str | None,
    to_cue: str | None,
    from_start_seconds: float,
    to_start_seconds: float,
    beats_per_bar: int,
    sample_rate: int,
    preset: str,
    volume_mode: str,
    eq_mode: str,
    filter_mode: str,
    automation: dict[str, dict[str, list[dict[str, float | str]]]] | None,
    effects: dict[str, dict[str, float]],
    loudness_match_mode: str,
    from_loudness_lufs: float | None,
    to_loudness_lufs: float | None,
    from_match_gain_db: float,
    to_match_gain_db: float,
    quality: str,
) -> str:
    automation_key = "" if automation is None else json.dumps(automation, sort_keys=True, separators=(",", ":"))
    raw = "|".join(
        [
            str(from_track.audio_path),
            str(to_track.audio_path),
            str(overlap_bars),
            str(front_padding_bars),
            str(back_padding_bars),
            f"{from_nudge_beats:.6f}",
            f"{to_nudge_beats:.6f}",
            f"{from_beatgrid_ms:.3f}",
            f"{to_beatgrid_ms:.3f}",
            str(from_pitch_shift),
            str(to_pitch_shift),
            str(from_bar),
            str(to_bar),
            str(from_cue or ""),
            str(to_cue or ""),
            f"{from_start_seconds:.9f}",
            f"{to_start_seconds:.9f}",
            str(beats_per_bar),
            str(sample_rate),
            preset,
            volume_mode,
            eq_mode,
            filter_mode,
            automation_key,
            json.dumps(effects, sort_keys=True, separators=(",", ":")),
            loudness_match_mode,
            "" if from_loudness_lufs is None else f"{from_loudness_lufs:.6f}",
            "" if to_loudness_lufs is None else f"{to_loudness_lufs:.6f}",
            f"{from_match_gain_db:.6f}",
            f"{to_match_gain_db:.6f}",
            f"{DEFAULT_TARGET_LUFS:.3f}",
            f"{MAX_COMPENSATION_DB:.3f}",
            quality,
            str(TRANSITION_QUALITY_PROFILES[quality]["stretch_algorithm"]),
            RENDER_MIX_VERSION,
            AUTOMATION_FADER_LAW_VERSION,
            f"{MIX_HEADROOM_DB:.3f}",
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return (
        f"{from_track.track_number:03d}_{_slug(from_track.title)}__"
        f"{to_track.track_number:03d}_{_slug(to_track.title)}__{digest}"
    )


def _source_fingerprint(path: Path) -> str:
    """Small cache identity which changes when a local source file changes."""
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"


def prepare_live_transition(
    *,
    tracklist_csv: Path = DEFAULT_TRACKLIST,
    from_query: str,
    to_query: str,
    overlap_bars: int = 8,
    front_padding_bars: int = 0,
    back_padding_bars: int = 0,
    from_bar: float = 0.0,
    to_bar: float = 0.0,
    from_cue: str | None = None,
    to_cue: str | None = None,
    cue_table: Path | None = None,
    from_nudge_beats: float = 0.0,
    to_nudge_beats: float = 0.0,
    from_beatgrid_ms: float = 0.0,
    to_beatgrid_ms: float = 0.0,
    from_pitch_shift: int = 0,
    to_pitch_shift: int = 0,
    beats_per_bar: int = 4,
    guard_bars: int = 4,
    sample_rate: int | None = None,
    quality: str = "preview",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
) -> PreparedLiveTransition:
    """Prepare a short matched deck pair for live browser-side mixing.

    The source processing deliberately shares the offline renderer's alignment
    and Rubber Band path, but leaves gain/EQ/filter automation to Web Audio.
    Both returned files have the same sample count and a guarded loop region.
    """
    tracks = load_playlist_tracks(tracklist_csv)
    quality, sample_rate = _resolve_quality(quality, sample_rate)
    rows = _read_tracklist_rows(tracklist_csv)
    if cue_table is None:
        cue_table = _default_cue_table(tracklist_csv)
    cue_rows = _read_cue_rows(cue_table)
    from_track = _find_track(tracks, from_query)
    to_track = _find_track(tracks, to_query)
    from_row = _row_for_track(rows, from_track)
    to_row = _row_for_track(rows, to_track)

    from_bpm = _required_float(from_track.bpm, label="bpm", track=from_track)
    to_bpm = _required_float(to_track.bpm, label="bpm", track=to_track)
    from_onset = _onset_time(from_track)
    to_onset = _onset_time(to_track)
    overlap_bars = max(1, int(overlap_bars))
    front_padding_bars = max(0, int(front_padding_bars))
    back_padding_bars = max(0, int(back_padding_bars))
    guard_bars = max(0, int(guard_bars))
    beats_per_bar = max(1, int(beats_per_bar))
    from_pitch_shift = max(-3, min(3, int(from_pitch_shift)))
    to_pitch_shift = max(-3, min(3, int(to_pitch_shift)))
    from_nudge_beats = float(from_nudge_beats)
    to_nudge_beats = float(to_nudge_beats)
    from_beatgrid_ms = float(from_beatgrid_ms)
    to_beatgrid_ms = float(to_beatgrid_ms)

    from_cue_seconds = _cue_seconds(cue_rows=cue_rows, track=from_track, row=from_row, cue_name=from_cue)
    to_cue_seconds = _cue_seconds(cue_rows=cue_rows, track=to_track, row=to_row, cue_name=to_cue)
    if from_cue is not None and from_cue_seconds is None:
        raise ValueError(f"Track A has no imported cue named '{from_cue}': {from_track.title}")
    if to_cue is not None and to_cue_seconds is None:
        raise ValueError(f"Track B has no imported cue named '{to_cue}': {to_track.title}")

    from_start = (
        float(from_cue_seconds)
        if from_cue_seconds is not None
        else from_onset + (float(from_bar) * beats_per_bar * 60.0 / from_bpm)
    )
    from_start += (from_nudge_beats * 60.0 / from_bpm) + (from_beatgrid_ms / 1000.0)
    to_start = (
        float(to_cue_seconds)
        if to_cue_seconds is not None
        else to_onset + (float(to_bar) * beats_per_bar * 60.0 / to_bpm)
    )
    to_start += (to_nudge_beats * 60.0 / to_bpm) + (to_beatgrid_ms / 1000.0)

    beat_seconds = 60.0 / from_bpm
    guard_seconds = guard_bars * beats_per_bar * beat_seconds
    front_padding_seconds = front_padding_bars * beats_per_bar * beat_seconds
    overlap_seconds = overlap_bars * beats_per_bar * beat_seconds
    back_padding_seconds = back_padding_bars * beats_per_bar * beat_seconds
    transport_seconds = front_padding_seconds + overlap_seconds + back_padding_seconds
    total_seconds = guard_seconds + transport_seconds + guard_seconds
    total_samples = max(1, int(round(total_seconds * sample_rate)))
    transport_start_samples = max(0, int(round(guard_seconds * sample_rate)))
    transition_start_samples = transport_start_samples + int(round(front_padding_seconds * sample_rate))
    transition_end_samples = transition_start_samples + int(round(overlap_seconds * sample_rate))
    transport_end_samples = min(total_samples, transport_start_samples + int(round(transport_seconds * sample_rate)))

    key_material = "|".join(
        [
            "live-v2",
            _source_fingerprint(from_track.audio_path),
            _source_fingerprint(to_track.audio_path),
            f"{from_start:.9f}",
            f"{to_start:.9f}",
            str(overlap_bars),
            str(front_padding_bars),
            str(back_padding_bars),
            str(guard_bars),
            str(beats_per_bar),
            str(from_pitch_shift),
            str(to_pitch_shift),
            str(sample_rate),
            quality,
            str(TRANSITION_QUALITY_PROFILES[quality]["stretch_algorithm"]),
        ]
    )
    digest = hashlib.sha1(key_material.encode("utf-8")).hexdigest()[:12]
    transition_id = f"live_{digest}"
    transition_dir = output_dir.expanduser().resolve() / transition_id
    from_path = transition_dir / "from_live.wav"
    to_path = transition_dir / "to_live.wav"
    metadata_path = transition_dir / "live_transition.json"
    if from_path.exists() and to_path.exists() and metadata_path.exists() and not overwrite:
        return PreparedLiveTransition(
            transition_id=transition_id,
            from_path=from_path,
            to_path=to_path,
            sample_rate=sample_rate,
            duration_seconds=total_seconds,
            transport_start_seconds=transport_start_samples / float(sample_rate),
            transport_end_seconds=transport_end_samples / float(sample_rate),
            transition_start_seconds=transition_start_samples / float(sample_rate),
            transition_end_seconds=transition_end_samples / float(sample_rate),
            loop_start_seconds=transport_start_samples / float(sample_rate),
            loop_end_seconds=transport_end_samples / float(sample_rate),
            overlap_bars=overlap_bars,
            front_padding_bars=front_padding_bars,
            back_padding_bars=back_padding_bars,
            guard_bars=guard_bars,
            beats_per_bar=beats_per_bar,
            from_bpm=from_bpm,
            to_bpm=to_bpm,
            b_time_stretch_rate=from_bpm / to_bpm,
        )

    transition_dir.mkdir(parents=True, exist_ok=True)
    from_audio = _load_segment_with_padding(
        from_track.audio_path,
        start_seconds=from_start - front_padding_seconds - guard_seconds,
        duration_seconds=total_seconds,
        sample_rate=sample_rate,
    )
    to_source_seconds = total_seconds * (from_bpm / to_bpm)
    to_audio = _load_segment_with_padding(
        to_track.audio_path,
        start_seconds=to_start - ((guard_bars + front_padding_bars) * beats_per_bar * 60.0 / to_bpm),
        duration_seconds=to_source_seconds,
        sample_rate=sample_rate,
    )
    from_audio = _time_stretch_multichannel(
        from_audio,
        rate=1.0,
        target_samples=total_samples,
        sample_rate=sample_rate,
        quality=quality,
        pitch_shift_semitones=from_pitch_shift,
    )
    to_audio = _time_stretch_multichannel(
        to_audio,
        rate=from_bpm / to_bpm,
        target_samples=total_samples,
        sample_rate=sample_rate,
        quality=quality,
        pitch_shift_semitones=to_pitch_shift,
    )
    sf.write(str(from_path), from_audio.T, sample_rate, format="WAV", subtype="PCM_16")
    sf.write(str(to_path), to_audio.T, sample_rate, format="WAV", subtype="PCM_16")
    metadata_path.write_text(
        json.dumps(
            {
                "version": 2,
                "from_source": str(from_track.audio_path),
                "to_source": str(to_track.audio_path),
                "sample_rate": sample_rate,
                "duration_seconds": total_seconds,
                "transport_start_seconds": transport_start_samples / float(sample_rate),
                "transport_end_seconds": transport_end_samples / float(sample_rate),
                "transition_start_seconds": transition_start_samples / float(sample_rate),
                "transition_end_seconds": transition_end_samples / float(sample_rate),
                "loop_start_seconds": transport_start_samples / float(sample_rate),
                "loop_end_seconds": transport_end_samples / float(sample_rate),
                "overlap_bars": overlap_bars,
                "front_padding_bars": front_padding_bars,
                "back_padding_bars": back_padding_bars,
                "guard_bars": guard_bars,
                "beats_per_bar": beats_per_bar,
                "from_start_seconds": from_start,
                "to_start_seconds": to_start,
                "b_time_stretch_rate": from_bpm / to_bpm,
                "quality": quality,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return PreparedLiveTransition(
        transition_id=transition_id,
        from_path=from_path,
        to_path=to_path,
        sample_rate=sample_rate,
        duration_seconds=total_seconds,
        transport_start_seconds=transport_start_samples / float(sample_rate),
        transport_end_seconds=transport_end_samples / float(sample_rate),
        transition_start_seconds=transition_start_samples / float(sample_rate),
        transition_end_seconds=transition_end_samples / float(sample_rate),
        loop_start_seconds=transport_start_samples / float(sample_rate),
        loop_end_seconds=transport_end_samples / float(sample_rate),
        overlap_bars=overlap_bars,
        front_padding_bars=front_padding_bars,
        back_padding_bars=back_padding_bars,
        guard_bars=guard_bars,
        beats_per_bar=beats_per_bar,
        from_bpm=from_bpm,
        to_bpm=to_bpm,
        b_time_stretch_rate=from_bpm / to_bpm,
    )


def render_transition(
    *,
    tracklist_csv: Path = DEFAULT_TRACKLIST,
    from_query: str,
    to_query: str,
    overlap_bars: int = 8,
    from_bar: float = 0.0,
    to_bar: float = 0.0,
    from_cue: str | None = None,
    to_cue: str | None = None,
    cue_table: Path | None = None,
    front_padding_bars: int = 0,
    back_padding_bars: int = 0,
    from_nudge_beats: float = 0.0,
    to_nudge_beats: float = 0.0,
    from_beatgrid_ms: float = 0.0,
    to_beatgrid_ms: float = 0.0,
    from_pitch_shift: int = 0,
    to_pitch_shift: int = 0,
    preset: str = "auto",
    volume_mode: str | None = None,
    eq_mode: str | None = None,
    filter_mode: str | None = None,
    automation: Mapping[str, Any] | None = None,
    effects: Mapping[str, Any] | None = None,
    loudness_match_mode: str = "off",
    from_loudness_lufs: float | None = None,
    to_loudness_lufs: float | None = None,
    beats_per_bar: int = 4,
    sample_rate: int | None = None,
    quality: str = "preview",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
    write_waveform: bool = True,
) -> RenderedTransition:
    tracks = load_playlist_tracks(tracklist_csv)
    preset, volume_mode, eq_mode, filter_mode = _resolve_modes(
        preset=preset,
        volume_mode=volume_mode,
        eq_mode=eq_mode,
        filter_mode=filter_mode,
    )
    quality, sample_rate = _resolve_quality(quality, sample_rate)
    rows = _read_tracklist_rows(tracklist_csv)
    if cue_table is None:
        cue_table = _default_cue_table(tracklist_csv)
    cue_rows = _read_cue_rows(cue_table)
    from_track = _find_track(tracks, from_query)
    to_track = _find_track(tracks, to_query)
    from_row = _row_for_track(rows, from_track)
    to_row = _row_for_track(rows, to_track)

    from_bpm = _required_float(from_track.bpm, label="bpm", track=from_track)
    to_bpm = _required_float(to_track.bpm, label="bpm", track=to_track)
    from_onset = _onset_time(from_track)
    to_onset = _onset_time(to_track)

    overlap_bars = max(1, int(overlap_bars))
    front_padding_bars = max(0, int(front_padding_bars))
    back_padding_bars = max(0, int(back_padding_bars))
    from_nudge_beats = float(from_nudge_beats)
    to_nudge_beats = float(to_nudge_beats)
    from_beatgrid_ms = float(from_beatgrid_ms)
    to_beatgrid_ms = float(to_beatgrid_ms)
    from_pitch_shift = max(-3, min(3, int(from_pitch_shift)))
    to_pitch_shift = max(-3, min(3, int(to_pitch_shift)))
    beats_per_bar = max(1, int(beats_per_bar))
    total_beats = float(overlap_bars * beats_per_bar)
    normalized_automation = _normalize_automation(automation, total_beats=total_beats)
    normalized_effects = _normalize_fx_settings(effects)
    loudness_match_mode = normalize_mode(loudness_match_mode)
    from_match_gain_db = LoudnessMeasurement(integrated_lufs=from_loudness_lufs).gain_db(loudness_match_mode)
    to_match_gain_db = LoudnessMeasurement(integrated_lufs=to_loudness_lufs).gain_db(loudness_match_mode)

    target_duration = overlap_bars * beats_per_bar * 60.0 / from_bpm
    front_padding_duration = front_padding_bars * beats_per_bar * 60.0 / from_bpm
    back_padding_duration = back_padding_bars * beats_per_bar * 60.0 / from_bpm
    render_duration = front_padding_duration + target_duration + back_padding_duration
    from_cue_seconds = _cue_seconds(cue_rows=cue_rows, track=from_track, row=from_row, cue_name=from_cue)
    to_cue_seconds = _cue_seconds(cue_rows=cue_rows, track=to_track, row=to_row, cue_name=to_cue)
    if from_cue is not None and from_cue_seconds is None:
        cue_hint = f" in {cue_table}" if cue_table is not None else ""
        raise ValueError(f"Track A has no imported cue named '{from_cue}'{cue_hint}: {from_track.title}")
    if to_cue is not None and to_cue_seconds is None:
        cue_hint = f" in {cue_table}" if cue_table is not None else ""
        raise ValueError(f"Track B has no imported cue named '{to_cue}'{cue_hint}: {to_track.title}")

    from_start = (
        float(from_cue_seconds)
        if from_cue_seconds is not None
        else from_onset + (float(from_bar) * beats_per_bar * 60.0 / from_bpm)
    )
    from_start += (from_nudge_beats * 60.0 / from_bpm) + (from_beatgrid_ms / 1000.0)
    to_start = (
        float(to_cue_seconds)
        if to_cue_seconds is not None
        else to_onset + (float(to_bar) * beats_per_bar * 60.0 / to_bpm)
    )
    to_start += (to_nudge_beats * 60.0 / to_bpm) + (to_beatgrid_ms / 1000.0)
    target_samples = max(1, int(round(target_duration * sample_rate)))

    key = _transition_key(
        from_track=from_track,
        to_track=to_track,
        overlap_bars=overlap_bars,
        front_padding_bars=front_padding_bars,
        back_padding_bars=back_padding_bars,
        from_nudge_beats=from_nudge_beats,
        to_nudge_beats=to_nudge_beats,
        from_beatgrid_ms=from_beatgrid_ms,
        to_beatgrid_ms=to_beatgrid_ms,
        from_pitch_shift=from_pitch_shift,
        to_pitch_shift=to_pitch_shift,
        from_bar=from_bar,
        to_bar=to_bar,
        from_cue=from_cue,
        to_cue=to_cue,
        from_start_seconds=from_start,
        to_start_seconds=to_start,
        beats_per_bar=beats_per_bar,
        sample_rate=sample_rate,
        preset=preset,
        volume_mode=volume_mode,
        eq_mode=eq_mode,
        filter_mode=filter_mode,
        automation=normalized_automation,
        effects=normalized_effects,
        loudness_match_mode=loudness_match_mode,
        from_loudness_lufs=from_loudness_lufs,
        to_loudness_lufs=to_loudness_lufs,
        from_match_gain_db=from_match_gain_db,
        to_match_gain_db=to_match_gain_db,
        quality=quality,
    )
    transition_dir = output_dir.expanduser().resolve() / key
    preview_path = transition_dir / "preview.wav"
    metadata_path = transition_dir / "transition.json"
    waveform_path = transition_dir / "transition_waveforms.json"
    if preview_path.exists() and metadata_path.exists() and not overwrite and (not write_waveform or waveform_path.exists()):
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        return RenderedTransition(
            preview_path=preview_path,
            metadata_path=metadata_path,
            waveform_path=Path(data.get("files", {}).get("waveform_json", waveform_path)),
            from_track_number=int(data["from_track"]["track_number"]),
            to_track_number=int(data["to_track"]["track_number"]),
            from_title=str(data["from_track"]["title"]),
            to_title=str(data["to_track"]["title"]),
            from_bpm=float(data["from_track"]["bpm"]),
            to_bpm=float(data["to_track"]["bpm"]),
            from_onset_time=float(data["from_track"]["onset_time"]),
            to_onset_time=float(data["to_track"]["onset_time"]),
            from_start_seconds=float(data["render"]["from_start_seconds"]),
            to_start_seconds=float(data["render"]["to_start_seconds"]),
            duration_seconds=float(data["render"]["duration_seconds"]),
            overlap_bars=int(data["render"]["overlap_bars"]),
            front_padding_bars=int(data["render"].get("front_padding_bars", 0)),
            back_padding_bars=int(data["render"].get("back_padding_bars", 0)),
            from_nudge_beats=float(data["render"].get("from_nudge_beats", 0.0)),
            to_nudge_beats=float(data["render"].get("to_nudge_beats", 0.0)),
            from_beatgrid_ms=float(data["render"].get("from_beatgrid_ms", 0.0)),
            to_beatgrid_ms=float(data["render"].get("to_beatgrid_ms", 0.0)),
            from_pitch_shift=int(data["render"].get("from_pitch_shift", 0)),
            to_pitch_shift=int(data["render"].get("to_pitch_shift", 0)),
            beats_per_bar=int(data["render"]["beats_per_bar"]),
            target_sample_rate=int(data["render"]["sample_rate"]),
            quality=str(data["render"].get("quality", quality)),
            b_time_stretch_rate=float(data["render"]["b_time_stretch_rate"]),
            preset=str(data["render"].get("preset", preset)),
            volume_mode=str(data["render"].get("volume_mode", volume_mode)),
            eq_mode=str(data["render"].get("eq_mode", eq_mode)),
            filter_mode=str(data["render"].get("filter_mode", filter_mode)),
        )

    transition_dir.mkdir(parents=True, exist_ok=True)
    target_samples = max(1, int(round(target_duration * sample_rate)))
    front_padding_samples = max(0, int(round(front_padding_duration * sample_rate)))
    back_padding_samples = max(0, int(round(back_padding_duration * sample_rate)))
    total_samples = front_padding_samples + target_samples + back_padding_samples

    a_source_start = from_start - front_padding_duration
    a_context = _load_segment_with_padding(
        from_track.audio_path,
        start_seconds=a_source_start,
        duration_seconds=render_duration,
        sample_rate=sample_rate,
    )
    b_source_start = to_start - (front_padding_bars * beats_per_bar * 60.0 / to_bpm)
    b_source_duration = (front_padding_bars + overlap_bars + back_padding_bars) * beats_per_bar * 60.0 / to_bpm
    b_context = _load_segment_with_padding(
        to_track.audio_path,
        start_seconds=b_source_start,
        duration_seconds=b_source_duration,
        sample_rate=sample_rate,
    )

    b_rate = from_bpm / to_bpm
    b_context = _time_stretch_multichannel(
        b_context,
        rate=b_rate,
        target_samples=total_samples,
        sample_rate=sample_rate,
        quality=quality,
        pitch_shift_semitones=to_pitch_shift,
    )
    a_context = _time_stretch_multichannel(
        a_context,
        rate=1.0,
        target_samples=total_samples,
        sample_rate=sample_rate,
        quality=quality,
        pitch_shift_semitones=from_pitch_shift,
    )
    # Loudness matching is a fixed pre-fader deck gain.  It applies to both
    # audible padding and overlap material before any deck automation or FX.
    a_context = (a_context * float(10.0 ** (from_match_gain_db / 20.0))).astype(np.float32)
    b_context = (b_context * float(10.0 ** (to_match_gain_db / 20.0))).astype(np.float32)
    a_overlap = a_context[:, front_padding_samples : front_padding_samples + target_samples]
    b_overlap = b_context[:, front_padding_samples : front_padding_samples + target_samples]
    mixed, automation, a_overlap_stem, b_overlap_stem = _render_mix(
        a_overlap,
        b_overlap,
        sample_rate=sample_rate,
        volume_mode=volume_mode,
        eq_mode=eq_mode,
        filter_mode=filter_mode,
        automation_spec=normalized_automation,
        total_beats=total_beats,
    )

    channels = max(a_context.shape[0], b_context.shape[0], mixed.shape[0])
    if a_context.shape[0] != channels:
        a_context = np.repeat(a_context[:1, :], channels, axis=0)
    if b_context.shape[0] != channels:
        b_context = np.repeat(b_context[:1, :], channels, axis=0)
    if mixed.shape[0] != channels:
        mixed = np.repeat(mixed[:1, :], channels, axis=0)

    a_visual = _fit_length(a_context, total_samples)
    b_visual = _fit_length(b_context, total_samples)
    full_mix = np.zeros((channels, total_samples), dtype=np.float32)

    transition_start = front_padding_samples
    transition_end = transition_start + target_samples
    a_stem = np.zeros((channels, total_samples), dtype=np.float32)
    b_stem = np.zeros((channels, total_samples), dtype=np.float32)
    if front_padding_samples:
        a_stem[:, :transition_start] = a_visual[:, :transition_start]
    a_stem[:, transition_start:transition_end] = a_overlap_stem
    b_stem[:, transition_start:transition_end] = b_overlap_stem
    if back_padding_samples:
        b_stem[:, transition_end:] = b_visual[:, transition_end:]

    full_automation = _pad_automation(
        automation,
        front_samples=front_padding_samples,
        back_samples=back_padding_samples,
    )
    for stem, prefix, deck in ((a_stem, "a", "from"), (b_stem, "b", "to")):
        settings = normalized_effects[deck]
        stem[:] = _apply_tempo_delay(
            stem,
            sample_rate=sample_rate,
            depth=np.asarray(full_automation[f"{prefix}_delay_depth"], dtype=np.float32),
            beat_seconds=60.0 / from_bpm,
            beats=settings["delay_beats"],
            tone=settings["delay_tone"],
        )
        stem[:] = _apply_schroeder_reverb(
            stem,
            sample_rate=sample_rate,
            depth=np.asarray(full_automation[f"{prefix}_reverb_depth"], dtype=np.float32),
            decay_seconds=settings["reverb_decay"],
            tone=settings["reverb_tone"],
        )
    full_mix = a_stem + b_stem

    full_mix = _apply_mix_headroom(full_mix)
    full_mix = _apply_final_limiter(full_mix, sample_rate=sample_rate)
    sf.write(str(preview_path), full_mix.T, sample_rate, format="WAV", subtype="PCM_16")
    if write_waveform:
        _write_waveform_json(
            path=waveform_path,
            a_audio=a_visual,
            b_audio=b_visual,
            automation=full_automation,
            sample_rate=sample_rate,
            duration_seconds=render_duration,
            transition_start_seconds=front_padding_duration,
            transition_duration_seconds=target_duration,
            front_padding_bars=front_padding_bars,
            back_padding_bars=back_padding_bars,
            overlap_bars=overlap_bars,
        )

    metadata = {
        "from_track": {
            "track_number": from_track.track_number,
            "title": from_track.title,
            "artists": from_track.artists,
            "filename": from_track.mp3_name,
            "audio_path": str(from_track.audio_path),
            "key": from_track.key,
            "shifted_key": _shift_camelot_key(from_track.key, from_pitch_shift),
            "bpm": from_bpm,
            "onset_time": from_onset,
            "start_bar": float(from_bar),
            "cue_name": from_cue,
            "cue_seconds": from_cue_seconds,
            "nudge_beats": from_nudge_beats,
            "beatgrid_ms": from_beatgrid_ms,
            "pitch_shift": from_pitch_shift,
        },
        "to_track": {
            "track_number": to_track.track_number,
            "title": to_track.title,
            "artists": to_track.artists,
            "filename": to_track.mp3_name,
            "audio_path": str(to_track.audio_path),
            "key": to_track.key,
            "shifted_key": _shift_camelot_key(to_track.key, to_pitch_shift),
            "bpm": to_bpm,
            "onset_time": to_onset,
            "start_bar": float(to_bar),
            "cue_name": to_cue,
            "cue_seconds": to_cue_seconds,
            "nudge_beats": to_nudge_beats,
            "beatgrid_ms": to_beatgrid_ms,
            "pitch_shift": to_pitch_shift,
        },
        "render": {
            "preset": preset,
            "volume_mode": volume_mode,
            "eq_mode": eq_mode,
            "filter_mode": filter_mode,
            "automation": normalized_automation,
            "effects": normalized_effects,
            "loudness_matching": {
                "mode": loudness_match_mode,
                "target_lufs": DEFAULT_TARGET_LUFS,
                "maximum_compensation_db": MAX_COMPENSATION_DB,
                "from_integrated_lufs": from_loudness_lufs,
                "to_integrated_lufs": to_loudness_lufs,
                "from_match_gain_db": from_match_gain_db,
                "to_match_gain_db": to_match_gain_db,
            },
            "mix_headroom_db": MIX_HEADROOM_DB,
            "mix_revision": RENDER_MIX_VERSION,
            "limiter": {
                "type": "oversampled-soft-true-peak",
                "oversample": 4,
                "ceiling_dbtp": TRUE_PEAK_CEILING_DBTP,
            },
            "overlap_bars": overlap_bars,
            "front_padding_bars": front_padding_bars,
            "back_padding_bars": back_padding_bars,
            "from_nudge_beats": from_nudge_beats,
            "to_nudge_beats": to_nudge_beats,
            "from_beatgrid_ms": from_beatgrid_ms,
            "to_beatgrid_ms": to_beatgrid_ms,
            "from_pitch_shift": from_pitch_shift,
            "to_pitch_shift": to_pitch_shift,
            "timeline_bars": front_padding_bars + overlap_bars + back_padding_bars,
            "beats_per_bar": beats_per_bar,
            "sample_rate": sample_rate,
            "quality": quality,
            "duration_seconds": render_duration,
            "transition_duration_seconds": target_duration,
            "front_padding_seconds": front_padding_duration,
            "back_padding_seconds": back_padding_duration,
            "transition_start_seconds": front_padding_duration,
            "transition_end_seconds": front_padding_duration + target_duration,
            "from_start_seconds": from_start,
            "to_start_seconds": to_start,
            "outgoing_context_start_seconds": a_source_start,
            "incoming_context_start_seconds": b_source_start,
            "b_time_stretch_rate": b_rate,
            "stretch_algorithm": str(TRANSITION_QUALITY_PROFILES[quality]["stretch_algorithm"]),
            "uses_track_a_bpm": True,
            "cue_table": "" if cue_table is None else str(cue_table),
        },
        "files": {
            "preview_wav": str(preview_path),
            "metadata_json": str(metadata_path),
            "waveform_json": str(waveform_path) if write_waveform else "",
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return RenderedTransition(
        preview_path=preview_path,
        metadata_path=metadata_path,
        waveform_path=waveform_path,
        from_track_number=from_track.track_number,
        to_track_number=to_track.track_number,
        from_title=from_track.title,
        to_title=to_track.title,
        from_bpm=from_bpm,
        to_bpm=to_bpm,
        from_onset_time=from_onset,
        to_onset_time=to_onset,
        from_start_seconds=from_start,
        to_start_seconds=to_start,
        duration_seconds=render_duration,
        overlap_bars=overlap_bars,
        front_padding_bars=front_padding_bars,
        back_padding_bars=back_padding_bars,
        from_nudge_beats=from_nudge_beats,
        to_nudge_beats=to_nudge_beats,
        from_beatgrid_ms=from_beatgrid_ms,
        to_beatgrid_ms=to_beatgrid_ms,
        from_pitch_shift=from_pitch_shift,
        to_pitch_shift=to_pitch_shift,
        beats_per_bar=beats_per_bar,
        target_sample_rate=sample_rate,
        quality=quality,
        b_time_stretch_rate=b_rate,
        preset=preset,
        volume_mode=volume_mode,
        eq_mode=eq_mode,
        filter_mode=filter_mode,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a prototype equal-power transition between two tracks in a playlist CSV.",
    )
    parser.add_argument("from_track", help="Source track number, title, filename, or filename stem.")
    parser.add_argument("to_track", help="Destination track number, title, filename, or filename stem.")
    parser.add_argument("--tracklist", type=Path, default=DEFAULT_TRACKLIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overlap-bars", type=int, default=8)
    parser.add_argument("--padding-bars", type=int, default=0, help="Apply this many context bars before and after the transition.")
    parser.add_argument("--front-padding-bars", type=int, default=None, help="Outgoing-song bars to include before the transition cue.")
    parser.add_argument("--back-padding-bars", type=int, default=None, help="Incoming-song bars to include after the transition.")
    parser.add_argument("--from-bar", type=float, default=0.0, help="Bar offset from track A onset-time.")
    parser.add_argument("--to-bar", type=float, default=0.0, help="Bar offset from track B onset-time.")
    parser.add_argument("--from-cue", default=None, help="Track A cue label, e.g. OUT 1 or OUT 2.")
    parser.add_argument("--to-cue", default=None, help="Track B cue label, e.g. IN 1 or IN 2.")
    parser.add_argument("--from-nudge-beats", type=float, default=0.0, help="Temporary beat offset from track A cue/start.")
    parser.add_argument("--to-nudge-beats", type=float, default=0.0, help="Temporary beat offset from track B cue/start.")
    parser.add_argument("--from-beatgrid-ms", type=float, default=0.0, help="Fine beatgrid offset for track A, in milliseconds.")
    parser.add_argument("--to-beatgrid-ms", type=float, default=0.0, help="Fine beatgrid offset for track B, in milliseconds.")
    parser.add_argument("--from-pitch-shift", type=int, default=0, choices=range(-3, 4), metavar="-3..3")
    parser.add_argument("--to-pitch-shift", type=int, default=0, choices=range(-3, 4), metavar="-3..3")
    parser.add_argument("--cue-table", type=Path, default=None, help="Long-format cue CSV. Defaults to <playlist>_cues.csv when present.")
    parser.add_argument(
        "--preset",
        default="auto",
        choices=sorted(TRANSITION_PRESETS),
        help="Combined transition preset. Explicit volume/EQ/filter flags override this.",
    )
    parser.add_argument("--volume-mode", default=None, choices=sorted(VOLUME_MODES))
    parser.add_argument("--eq-mode", default=None, choices=sorted(EQ_MODES))
    parser.add_argument("--filter-mode", default=None, choices=sorted(FILTER_MODES))
    parser.add_argument("--beats-per-bar", type=int, default=4)
    parser.add_argument("--quality", choices=TRANSITION_QUALITY_CHOICES, default="preview", help="Preview uses faster Rubber Band R2 at 44.1 kHz; final uses higher-quality R3.")
    parser.add_argument("--sample-rate", type=int, default=None, help="Override the sample rate chosen by --quality.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable render metadata.")
    args = parser.parse_args(argv)
    front_padding_bars = args.padding_bars if args.front_padding_bars is None else args.front_padding_bars
    back_padding_bars = args.padding_bars if args.back_padding_bars is None else args.back_padding_bars

    result = render_transition(
        tracklist_csv=args.tracklist,
        from_query=args.from_track,
        to_query=args.to_track,
        overlap_bars=args.overlap_bars,
        from_bar=args.from_bar,
        to_bar=args.to_bar,
        from_cue=args.from_cue,
        to_cue=args.to_cue,
        cue_table=args.cue_table,
        front_padding_bars=front_padding_bars,
        back_padding_bars=back_padding_bars,
        from_nudge_beats=args.from_nudge_beats,
        to_nudge_beats=args.to_nudge_beats,
        from_beatgrid_ms=args.from_beatgrid_ms,
        to_beatgrid_ms=args.to_beatgrid_ms,
        from_pitch_shift=args.from_pitch_shift,
        to_pitch_shift=args.to_pitch_shift,
        preset=args.preset,
        volume_mode=args.volume_mode,
        eq_mode=args.eq_mode,
        filter_mode=args.filter_mode,
        beats_per_bar=args.beats_per_bar,
        sample_rate=args.sample_rate,
        quality=args.quality,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
    )
    if args.json:
        print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in asdict(result).items()}, indent=2))
    else:
        print(f"Rendered transition: {result.from_track_number} {result.from_title} -> {result.to_track_number} {result.to_title}")
        print(f"Preview WAV: {result.preview_path}")
        print(f"Metadata: {result.metadata_path}")
        print(f"Waveform JSON: {result.waveform_path}")
        print(
            "Timing: "
            f"{result.overlap_bars} transition bars"
            f" (+{result.front_padding_bars}/+{result.back_padding_bars} padding), "
            f"{result.duration_seconds:.3f}s, "
            f"A BPM {result.from_bpm:.3f}, B BPM {result.to_bpm:.3f}, "
            f"B stretch rate {result.b_time_stretch_rate:.6f}"
            f" / {result.quality} quality"
        )
        print(
            "Automation: "
            f"preset={result.preset}, volume={result.volume_mode}, "
            f"eq={result.eq_mode}, filter={result.filter_mode}"
        )
        print(f"Pitch shift: A {result.from_pitch_shift:+d} st, B {result.to_pitch_shift:+d} st")
        print(f"Cue nudge: A {result.from_nudge_beats:+.2f} beats, B {result.to_nudge_beats:+.2f} beats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
