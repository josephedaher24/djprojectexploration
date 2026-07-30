"""Shared PaCMAP/reducer settings used by app and standalone exports."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PACMAP_PAIR_SOURCE_CHOICES = ("neighbors-only", "combined-all")
DISTANCE_COMBINE_CHOICES = ("l2", "l1")
LAYOUT_INIT_CHOICES = ("neighbor", "pca", "random")
LAYOUT_SELECTION_MODE_CHOICES = ("interpolated", "discrete")
SEQUENCE_BUILDER_PRESET_SECTION_KEYS = {"sequence_builder"}
SEQUENCE_BUILDER_PRESET_KEYS = {
    "mix_slugs",
    "tracklists",
    "energy_npz_path",
    "output_file",
    "output_dir",
    "sequence_length",
    "control_mode",
}
SEQUENCE_BUILDER_PRESET_ALIASES = {
    "mix": "mix_slugs",
    "mixes": "mix_slugs",
    "mix_slugs": "mix_slugs",
    "tracklist": "tracklists",
    "tracklists": "tracklists",
    "tracklist_paths": "tracklists",
    "energy_npz": "energy_npz_path",
    "energy_npz_path": "energy_npz_path",
    "output_file": "output_file",
    "output_dir": "output_dir",
    "sequence_length": "sequence_length",
    "control_mode": "control_mode",
}


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    preset_path = Path(path).expanduser().resolve()
    with preset_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{label} preset must be a JSON object: {preset_path}")
    return raw


def _string_list(value: Any, *, key: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError(f"{key} must be a string or list of strings.")


def _path_list(value: Any, *, key: str) -> list[Path]:
    return [Path(item) for item in _string_list(value, key=key)]


def ensure_numba_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Set a writable Numba cache directory before importing PaCMAP."""
    raw = (
        cache_dir
        or os.environ.get("DJPROJECTEXPLORATION_NUMBA_CACHE_DIR")
        or os.environ.get("NUMBA_CACHE_DIR")
    )
    path = Path(raw).expanduser() if raw else Path(tempfile.gettempdir()) / "djprojectexploration_numba_cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(path))
    if "numba" in sys.modules:
        try:
            import numba

            numba.config.CACHE_DIR = os.environ["NUMBA_CACHE_DIR"]
        except Exception:
            pass
    return path


@dataclass(frozen=True)
class PacmapSettings:
    n_neighbors: int = 10
    mn_ratio: float = 0.5
    fp_ratio: float = 1.5
    pair_source: str = "combined-all"
    distance_combine: str = "l1"
    layout_init: str = "neighbor"
    layout_selection_mode: str = "interpolated"
    step: float = 0.1
    static_layout: bool = False
    distance: str = "angular"
    random_state: int = 7777
    align_layouts: bool = True

    def validate(self) -> "PacmapSettings":
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1.")
        if self.mn_ratio < 0:
            raise ValueError("mn_ratio must be >= 0.")
        if self.fp_ratio < 0:
            raise ValueError("fp_ratio must be >= 0.")
        if self.step <= 0:
            raise ValueError("step must be > 0.")
        if self.pair_source not in PACMAP_PAIR_SOURCE_CHOICES:
            raise ValueError(f"pair_source must be one of {PACMAP_PAIR_SOURCE_CHOICES}, got {self.pair_source!r}.")
        if self.distance_combine not in DISTANCE_COMBINE_CHOICES:
            raise ValueError(
                f"distance_combine must be one of {DISTANCE_COMBINE_CHOICES}, got {self.distance_combine!r}."
            )
        if self.layout_init not in LAYOUT_INIT_CHOICES:
            raise ValueError(f"layout_init must be one of {LAYOUT_INIT_CHOICES}, got {self.layout_init!r}.")
        if self.layout_selection_mode not in LAYOUT_SELECTION_MODE_CHOICES:
            raise ValueError(
                "layout_selection_mode must be one of "
                f"{LAYOUT_SELECTION_MODE_CHOICES}, got {self.layout_selection_mode!r}."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "PacmapSettings":
        preset_path = Path(path).expanduser().resolve()
        raw = _load_json_object(preset_path, label="PaCMAP")
        aliases = {
            "MN_ratio": "mn_ratio",
            "mn_ratio": "mn_ratio",
            "FP_ratio": "fp_ratio",
            "fp_ratio": "fp_ratio",
            "n_neighbors": "n_neighbors",
        }
        allowed = set(cls.__dataclass_fields__)
        values: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "ui" or key in SEQUENCE_BUILDER_PRESET_SECTION_KEYS:
                continue
            normalized = aliases.get(str(key), SEQUENCE_BUILDER_PRESET_ALIASES.get(str(key), str(key)))
            if normalized in SEQUENCE_BUILDER_PRESET_KEYS:
                continue
            if normalized not in allowed:
                raise ValueError(f"Unknown PaCMAP preset key {key!r} in {preset_path}")
            values[normalized] = value
        return cls(**values).validate()


@dataclass(frozen=True)
class SequenceBuilderUiSettings:
    latent_links_per_track: int = 3
    recommended_links_highlight: int = 25
    point_color: str = "genre"
    map_renderer: str = "plotly"
    map_fx: bool = True

    def validate(self) -> "SequenceBuilderUiSettings":
        if self.latent_links_per_track < 0:
            raise ValueError("ui.latent_links_per_track must be >= 0.")
        if self.recommended_links_highlight < 1:
            raise ValueError("ui.recommended_links_highlight must be >= 1.")
        if self.point_color not in {"genre", "energy", "tempo", "key", "target", "energy_residual"}:
            raise ValueError(
                "ui.point_color must be one of {'genre', 'energy', 'tempo', 'key', 'target', 'energy_residual'}."
            )
        if self.map_renderer not in {"plotly", "webgl"}:
            raise ValueError("ui.map_renderer must be one of {'plotly', 'webgl'}.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "SequenceBuilderUiSettings":
        preset_path = Path(path).expanduser().resolve()
        raw = _load_json_object(preset_path, label="PaCMAP")
        raw_ui = raw.get("ui", {})
        if raw_ui is None:
            raw_ui = {}
        if not isinstance(raw_ui, dict):
            raise ValueError(f"PaCMAP preset ui section must be a JSON object: {preset_path}")
        aliases = {
            "latentLinksPerTrack": "latent_links_per_track",
            "recommendedLinksHighlight": "recommended_links_highlight",
            "pointColor": "point_color",
            "mapRenderer": "map_renderer",
            "mapFx": "map_fx",
        }
        allowed = set(cls.__dataclass_fields__)
        values: dict[str, Any] = {}
        for key, value in raw_ui.items():
            normalized = aliases.get(str(key), str(key))
            if normalized not in allowed:
                raise ValueError(f"Unknown PaCMAP preset ui key {key!r} in {preset_path}")
            values[normalized] = value
        return cls(**values).validate()


@dataclass(frozen=True)
class SequenceBuilderRunSettings:
    mix_slugs: list[str] | None = None
    tracklists: list[Path] | None = None
    energy_npz_path: Path | None = None
    output_file: Path | None = None
    output_dir: Path | None = None
    sequence_length: int | None = None
    control_mode: str | None = None

    def validate(self) -> "SequenceBuilderRunSettings":
        if self.sequence_length is not None and self.sequence_length < 2:
            raise ValueError("sequence_length must be >= 2.")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> "SequenceBuilderRunSettings":
        preset_path = Path(path).expanduser().resolve()
        raw = _load_json_object(preset_path, label="Sequence builder")
        values: dict[str, Any] = {}

        section = raw.get("sequence_builder", {})
        if section is None:
            section = {}
        if not isinstance(section, dict):
            raise ValueError(f"sequence_builder section must be a JSON object: {preset_path}")

        for source in (raw, section):
            for key, value in source.items():
                normalized = SEQUENCE_BUILDER_PRESET_ALIASES.get(str(key), str(key))
                if normalized not in SEQUENCE_BUILDER_PRESET_KEYS:
                    continue
                if normalized == "mix_slugs":
                    values[normalized] = _string_list(value, key=str(key))
                elif normalized == "tracklists":
                    values[normalized] = _path_list(value, key=str(key))
                elif normalized in {"energy_npz_path", "output_file", "output_dir"}:
                    values[normalized] = Path(str(value))
                elif normalized == "sequence_length":
                    values[normalized] = int(value)
                else:
                    values[normalized] = str(value)
        return cls(**values).validate()


def add_pacmap_args(parser: argparse.ArgumentParser, *, include_static_layout: bool = True) -> None:
    parser.add_argument("--pacmap-preset", type=Path, default=None, help="JSON file with reusable PaCMAP settings.")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)
    parser.add_argument("--mn-ratio", type=float, default=None)
    parser.add_argument("--fp-ratio", type=float, default=None)
    parser.add_argument("--distance", default=None)
    parser.add_argument("--step", type=float, default=None)
    parser.add_argument("--no-align-layouts", action="store_true", default=None)
    parser.add_argument("--pair-source", choices=PACMAP_PAIR_SOURCE_CHOICES, default=None)
    parser.add_argument("--distance-combine", choices=DISTANCE_COMBINE_CHOICES, default=None)
    parser.add_argument("--layout-init", choices=LAYOUT_INIT_CHOICES, default=None)
    parser.add_argument(
        "--layout-selection-mode",
        choices=LAYOUT_SELECTION_MODE_CHOICES,
        default=None,
        help="Use interpolated coordinates between layouts or snap to the nearest precomputed layout.",
    )
    if include_static_layout:
        parser.add_argument("--static-layout", action="store_true", default=None, help="Generate one fixed/default layout.")


def pacmap_settings_from_args(args: argparse.Namespace) -> PacmapSettings:
    base = (
        PacmapSettings.from_json(args.pacmap_preset)
        if getattr(args, "pacmap_preset", None) is not None
        else PacmapSettings()
    )
    values = base.to_dict()
    cli_fields = {
        "n_neighbors": getattr(args, "n_neighbors", None),
        "mn_ratio": getattr(args, "mn_ratio", None),
        "fp_ratio": getattr(args, "fp_ratio", None),
        "pair_source": getattr(args, "pair_source", None),
        "distance_combine": getattr(args, "distance_combine", None),
        "layout_init": getattr(args, "layout_init", None),
        "layout_selection_mode": getattr(args, "layout_selection_mode", None),
        "step": getattr(args, "step", None),
        "static_layout": getattr(args, "static_layout", None),
        "distance": getattr(args, "distance", None),
        "random_state": getattr(args, "random_state", None),
    }
    values.update({key: value for key, value in cli_fields.items() if value is not None})
    if getattr(args, "no_align_layouts", None):
        values["align_layouts"] = False
    return PacmapSettings(**values).validate()


def sequence_builder_ui_settings_from_args(args: argparse.Namespace) -> SequenceBuilderUiSettings:
    if getattr(args, "pacmap_preset", None) is None:
        return SequenceBuilderUiSettings()
    return SequenceBuilderUiSettings.from_json(args.pacmap_preset)


def sequence_builder_run_settings_from_args(args: argparse.Namespace) -> SequenceBuilderRunSettings:
    if getattr(args, "pacmap_preset", None) is None:
        return SequenceBuilderRunSettings()
    return SequenceBuilderRunSettings.from_json(args.pacmap_preset)
