"""Shared PaCMAP/reducer settings used by app and standalone exports."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PACMAP_PAIR_SOURCE_CHOICES = ("neighbors-only", "combined-all")
DISTANCE_COMBINE_CHOICES = ("l2", "l1")
LAYOUT_INIT_CHOICES = ("neighbor", "pca", "random")
LAYOUT_SELECTION_MODE_CHOICES = ("interpolated", "discrete")


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
        with preset_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"PaCMAP preset must be a JSON object: {preset_path}")
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
            if key == "ui":
                continue
            normalized = aliases.get(str(key), str(key))
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
        with preset_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"PaCMAP preset must be a JSON object: {preset_path}")
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
