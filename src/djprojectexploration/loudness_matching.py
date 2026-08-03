"""Shared loudness-matching metadata and gain policy for served auditions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from djprojectexploration.tracklists import PROJECT_ROOT, read_csv_rows


LOUDNESS_MATCH_OFF = "off"
LOUDNESS_MATCH_AUDITIONS = "auditions"
LOUDNESS_MATCH_AUDITIONS_TRANSITIONS = "auditions_transitions"
LOUDNESS_MATCH_MODES = (
    LOUDNESS_MATCH_OFF,
    LOUDNESS_MATCH_AUDITIONS,
    LOUDNESS_MATCH_AUDITIONS_TRANSITIONS,
)
DEFAULT_LOUDNESS_MATCH_MODE = LOUDNESS_MATCH_AUDITIONS_TRANSITIONS
DEFAULT_TARGET_LUFS = -12.0
MAX_COMPENSATION_DB = 6.0


def _token(value: Any) -> str:
    return Path(str(value or "").strip()).name.casefold()


def _finite(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


@dataclass(frozen=True)
class LoudnessMeasurement:
    integrated_lufs: float | None = None
    short_term_mean_lufs: float | None = None
    short_term_std_lu: float | None = None
    short_term_range_lu: float | None = None

    def gain_db(self, mode: str = DEFAULT_LOUDNESS_MATCH_MODE) -> float:
        if (
            mode != LOUDNESS_MATCH_AUDITIONS_TRANSITIONS
            or self.integrated_lufs is None
            or not math.isfinite(self.integrated_lufs)
        ):
            return 0.0
        return max(-MAX_COMPENSATION_DB, min(MAX_COMPENSATION_DB, DEFAULT_TARGET_LUFS - self.integrated_lufs))

    def payload(self, mode: str = DEFAULT_LOUDNESS_MATCH_MODE) -> dict[str, float | None]:
        return {
            "integrated_lufs": self.integrated_lufs,
            "short_term_mean_lufs": self.short_term_mean_lufs,
            "short_term_std_lu": self.short_term_std_lu,
            "short_term_range_lu": self.short_term_range_lu,
            "match_gain_db": self.gain_db(mode),
        }


def normalize_mode(value: Any) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in LOUDNESS_MATCH_MODES else DEFAULT_LOUDNESS_MATCH_MODE


def feature_csv_for_tracklist(tracklist_csv: str | Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    tracklist = Path(tracklist_csv).expanduser().resolve()
    return project_root / "data" / "energy_features" / f"{tracklist.stem}_energy_features.csv"


def load_loudness_catalog(tracklist_csv: str | Path, *, project_root: Path = PROJECT_ROOT) -> tuple[dict[str, LoudnessMeasurement], dict[str, LoudnessMeasurement]]:
    """Return filename and track-number lookups from an optional feature CSV."""
    filename_lookup: dict[str, LoudnessMeasurement] = {}
    number_lookup: dict[str, LoudnessMeasurement] = {}
    for row in read_csv_rows(feature_csv_for_tracklist(tracklist_csv, project_root=project_root), missing_ok=True):
        measurement = LoudnessMeasurement(
            integrated_lufs=_finite(row.get("full_integrated_loudness_lufs")),
            short_term_mean_lufs=_finite(row.get("full_short_term_loudness_mean_lufs")),
            short_term_std_lu=_finite(row.get("full_short_term_loudness_std_lu")),
            short_term_range_lu=_finite(row.get("full_short_term_loudness_range_lu")),
        )
        filename = _token(row.get("filename"))
        number = str(row.get("track_number") or "").strip()
        if filename and filename not in filename_lookup:
            filename_lookup[filename] = measurement
        if number and number not in number_lookup:
            number_lookup[number] = measurement
    return filename_lookup, number_lookup


def lookup_loudness(
    filename_lookup: dict[str, LoudnessMeasurement],
    number_lookup: dict[str, LoudnessMeasurement],
    *,
    filename: Any,
    track_number: Any,
) -> LoudnessMeasurement:
    return filename_lookup.get(_token(filename)) or number_lookup.get(str(track_number or "").strip()) or LoudnessMeasurement()
