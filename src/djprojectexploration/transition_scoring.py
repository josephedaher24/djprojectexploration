"""Canonical three-component transition scoring for DJ sequence tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULT_STYLE_WEIGHT = 0.50
DEFAULT_RHYTHM_WEIGHT = 0.30
DEFAULT_HARMONY_WEIGHT = 0.20
DEFAULT_RHYTHM_TEMPO_WEIGHT = 0.70


def _nonnegative(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return max(0.0, number)


def normalize_top_level_weights(
    style_weight: float,
    rhythm_weight: float,
    harmony_weight: float,
) -> tuple[float, float, float]:
    """Normalize non-negative Style/Rhythm/Harmony weights to sum to one."""
    weights = np.asarray(
        [
            _nonnegative(style_weight),
            _nonnegative(rhythm_weight),
            _nonnegative(harmony_weight),
        ],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= 1e-12:
        weights = np.asarray(
            [DEFAULT_STYLE_WEIGHT, DEFAULT_RHYTHM_WEIGHT, DEFAULT_HARMONY_WEIGHT],
            dtype=np.float64,
        )
        total = float(weights.sum())
    weights /= total
    return float(weights[0]), float(weights[1]), float(weights[2])


def canonical_scoring_config(values: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Return canonical weights, converting supported legacy representations."""
    raw = dict(values or {})
    nested = raw.get("transition_scoring")
    if isinstance(nested, Mapping):
        raw = dict(nested)
    nested_weights = raw.get("weights")
    if isinstance(nested_weights, Mapping):
        raw = {**raw, **dict(nested_weights)}

    has_legacy_rhythm_parts = any(
        key in raw
        for key in (
            "tempo_weight",
            "groove_weight",
            "tempo",
            "groove",
            "mix_tempo",
            "mix_groove",
        )
    )
    has_canonical = "rhythm_weight" in raw or (
        any(key in raw for key in ("style_weight", "harmony_weight"))
        and not has_legacy_rhythm_parts
    )
    if has_canonical:
        style = _nonnegative(
            raw.get("style_weight", raw.get("maest_weight", raw.get("maest", DEFAULT_STYLE_WEIGHT)))
        )
        rhythm = _nonnegative(raw.get("rhythm_weight", DEFAULT_RHYTHM_WEIGHT))
        harmony = _nonnegative(
            raw.get(
                "harmony_weight",
                raw.get("chroma_weight", raw.get("chroma", DEFAULT_HARMONY_WEIGHT)),
            )
        )
        rhythm_tempo = float(
            np.clip(
                _nonnegative(
                    raw.get("rhythm_tempo_weight", DEFAULT_RHYTHM_TEMPO_WEIGHT),
                    DEFAULT_RHYTHM_TEMPO_WEIGHT,
                ),
                0.0,
                1.0,
            )
        )
    else:
        style = _nonnegative(
            raw.get("style_weight", raw.get("maest_weight", raw.get("maest", DEFAULT_STYLE_WEIGHT)))
        )
        tempo = _nonnegative(raw.get("tempo_weight", raw.get("tempo", 0.0)))
        groove = _nonnegative(raw.get("groove_weight", raw.get("groove", 0.0)))
        harmony = _nonnegative(
            raw.get(
                "harmony_weight",
                raw.get("chroma_weight", raw.get("chroma", DEFAULT_HARMONY_WEIGHT)),
            )
        )

        # Older sequence-builder exports stored a Style weight plus a normalized
        # tempo/groove/chroma split for the remaining mass.
        if any(key in raw for key in ("mix_tempo", "mix_groove", "mix_chroma")):
            style = float(np.clip(style, 0.0, 1.0))
            remainder = 1.0 - style
            tempo = remainder * _nonnegative(raw.get("mix_tempo", 0.0))
            groove = remainder * _nonnegative(raw.get("mix_groove", 0.0))
            harmony = remainder * _nonnegative(raw.get("mix_chroma", 0.0))

        rhythm = tempo + groove
        rhythm_tempo = (
            tempo / rhythm if rhythm > 1e-12 else DEFAULT_RHYTHM_TEMPO_WEIGHT
        )

    style, rhythm, harmony = normalize_top_level_weights(style, rhythm, harmony)
    return {
        "style_weight": style,
        "rhythm_weight": rhythm,
        "harmony_weight": harmony,
        "rhythm_tempo_weight": rhythm_tempo,
    }


def rhythm_component_score(
    tempo_score: np.ndarray | float,
    groove_score: np.ndarray | float,
    *,
    rhythm_tempo_weight: float = DEFAULT_RHYTHM_TEMPO_WEIGHT,
) -> np.ndarray:
    """Combine already-normalized tempo and groove scores without re-normalizing."""
    tempo_weight = float(np.clip(rhythm_tempo_weight, 0.0, 1.0))
    return (
        tempo_weight * np.asarray(tempo_score, dtype=np.float64)
        + (1.0 - tempo_weight) * np.asarray(groove_score, dtype=np.float64)
    )


def baseline_transition_score(
    style_score: np.ndarray | float,
    rhythm_score: np.ndarray | float,
    harmony_score: np.ndarray | float,
    *,
    style_weight: float = DEFAULT_STYLE_WEIGHT,
    rhythm_weight: float = DEFAULT_RHYTHM_WEIGHT,
    harmony_weight: float = DEFAULT_HARMONY_WEIGHT,
) -> np.ndarray:
    """Compute the normalized Style/Rhythm/Harmony baseline score."""
    style_weight, rhythm_weight, harmony_weight = normalize_top_level_weights(
        style_weight,
        rhythm_weight,
        harmony_weight,
    )
    return (
        style_weight * np.asarray(style_score, dtype=np.float64)
        + rhythm_weight * np.asarray(rhythm_score, dtype=np.float64)
        + harmony_weight * np.asarray(harmony_score, dtype=np.float64)
    )


def blend_component_distances(
    style_distance: np.ndarray,
    rhythm_distance: np.ndarray,
    harmony_distance: np.ndarray,
    *,
    style_weight: float = DEFAULT_STYLE_WEIGHT,
    rhythm_weight: float = DEFAULT_RHYTHM_WEIGHT,
    harmony_weight: float = DEFAULT_HARMONY_WEIGHT,
) -> np.ndarray:
    """Linearly blend Style/Rhythm/Harmony distances for layout generation."""
    style_weight, rhythm_weight, harmony_weight = normalize_top_level_weights(
        style_weight,
        rhythm_weight,
        harmony_weight,
    )
    distance = (
        style_weight * np.asarray(style_distance, dtype=np.float64)
        + rhythm_weight * np.asarray(rhythm_distance, dtype=np.float64)
        + harmony_weight * np.asarray(harmony_distance, dtype=np.float64)
    )
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.0)
    return distance.astype(np.float32)


def energy_fit_score(
    candidate_energy: np.ndarray | float,
    target_energy: np.ndarray | float,
    *,
    energy_penalty: float,
) -> np.ndarray:
    """Return the multiplicative post-hoc energy fit."""
    error = np.asarray(candidate_energy, dtype=np.float64) - np.asarray(
        target_energy, dtype=np.float64
    )
    return 1.0 / (1.0 + _nonnegative(energy_penalty) * error**2)


def final_transition_score(
    baseline_score: np.ndarray | float,
    energy_fit: np.ndarray | float,
) -> np.ndarray:
    """Apply the energy fit and keep normalized inputs inside score bounds."""
    baseline = np.clip(np.asarray(baseline_score, dtype=np.float64), 0.0, 1.0)
    fit = np.clip(np.asarray(energy_fit, dtype=np.float64), 0.0, 1.0)
    return baseline * fit


def weight_shorthand(values: Mapping[str, Any] | None = None) -> str:
    """Format canonical weights for plot titles and exports."""
    config = canonical_scoring_config(values)
    return (
        f"{config['style_weight']:.2f}S/"
        f"{config['rhythm_weight']:.2f}R/"
        f"{config['harmony_weight']:.2f}H "
        f"(rhythm={config['rhythm_tempo_weight']:.2f}T+"
        f"{1.0 - config['rhythm_tempo_weight']:.2f}G)"
    )
