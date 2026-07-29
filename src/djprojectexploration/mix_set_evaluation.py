"""Batch evaluation of ordered DJ-set manifests over style/rhythm/harmony weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from djprojectexploration.multimodal_compatibility import (
    SongFeatureSet,
    SongMetadata,
    _build_harmonic_kernel,
    _pairwise_cosine_similarity_matrix,
    _pairwise_fifth_aware_similarity_matrix,
    _pairwise_tempo_similarity_matrix,
)
from djprojectexploration.transition_scoring import (
    DEFAULT_RHYTHM_TEMPO_WEIGHT,
    rhythm_component_score as _rhythm_component_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PERMUTATIONS = 500_000
DEFAULT_RANDOM_SEED = 20240722
DEFAULT_SIMPLEX_STEP = 0.10
DEFAULT_NORMALIZATION_DISTANCE_PERCENTILE = 95.0


def slug_from_manifest_path(manifest_path: str | Path) -> str:
    """Return a dataset slug from a rendered-set or canonical build manifest."""
    path = Path(manifest_path)
    name = path.name
    for suffix in ("_rendered_set_manifest.json", "_build_manifest.json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if name.endswith(".json"):
        return path.stem
    return path.name


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Mix manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected JSON object in manifest: {path}")
    return manifest


def _resolve_project_path(path: str | Path, *, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _resolve_feature_npz_path(
    *,
    manifest: dict[str, Any],
    key: str,
    fallback_subdir: str,
    slug: str,
    project_root: Path,
    override: str | Path | None = None,
) -> Path:
    if override is not None:
        return _resolve_project_path(override, project_root=project_root)

    outputs = manifest.get("outputs")
    if isinstance(outputs, dict) and outputs.get(key):
        return _resolve_project_path(outputs[key], project_root=project_root)

    return (project_root / "data" / fallback_subdir / f"{slug}_tracks.npz").resolve()


def _as_string_array(npz: np.lib.npyio.NpzFile, key: str, default: str, count: int) -> np.ndarray:
    if key in npz.files:
        return np.asarray(npz[key]).astype(str)
    return np.asarray([default] * count, dtype=str)


def _track_number_index(npz: np.lib.npyio.NpzFile, *, path: Path) -> dict[int, int]:
    if "track_numbers" not in npz.files:
        raise KeyError(f"Missing 'track_numbers' in NPZ: {path}")
    numbers = np.asarray(npz["track_numbers"], dtype=np.int64).reshape(-1)
    lookup: dict[int, int] = {}
    for index, track_number in enumerate(numbers):
        key = int(track_number)
        if key in lookup:
            raise ValueError(f"Duplicate track_number={key} in NPZ: {path}")
        lookup[key] = index
    return lookup


def _manifest_track_order(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("tracks")
    if isinstance(rows, list) and rows:
        return [row for row in rows if isinstance(row, dict)]
    return []


def load_rendered_set_feature_set(
    *,
    manifest: dict[str, Any],
    slug: str,
    project_root: Path,
    maest_npz_path: str | Path | None = None,
    chroma_npz_path: str | Path | None = None,
    tempo_npz_path: str | Path | None = None,
    groove_npz_path: str | Path | None = None,
    allow_missing_features: bool = False,
) -> tuple[SongFeatureSet, np.ndarray, dict[str, Path]]:
    """Load MAEST, chroma, tempo, and groove NPZs aligned to manifest order."""
    paths = {
        "maest": _resolve_feature_npz_path(
            manifest=manifest,
            key="maest",
            fallback_subdir="maest_embeddings",
            slug=slug,
            project_root=project_root,
            override=maest_npz_path,
        ),
        "chroma": _resolve_feature_npz_path(
            manifest=manifest,
            key="chroma",
            fallback_subdir="chroma_embeddings",
            slug=slug,
            project_root=project_root,
            override=chroma_npz_path,
        ),
        "tempo": _resolve_feature_npz_path(
            manifest=manifest,
            key="tempo",
            fallback_subdir="tempo_embeddings",
            slug=slug,
            project_root=project_root,
            override=tempo_npz_path,
        ),
        "groove": _resolve_feature_npz_path(
            manifest=manifest,
            key="groove",
            fallback_subdir="groove_embeddings",
            slug=slug,
            project_root=project_root,
            override=groove_npz_path,
        ),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} NPZ not found: {path}")

    with (
        np.load(paths["maest"], allow_pickle=False) as maest_npz,
        np.load(paths["chroma"], allow_pickle=False) as chroma_npz,
        np.load(paths["tempo"], allow_pickle=False) as tempo_npz,
        np.load(paths["groove"], allow_pickle=False) as groove_npz,
    ):
        maest_embeddings_all = np.asarray(maest_npz["embeddings"], dtype=np.float32)
        chroma_embeddings_all = np.asarray(chroma_npz["embeddings"], dtype=np.float32)
        groove_embeddings_all = np.asarray(groove_npz["embeddings"], dtype=np.float32)

        if "tempo_bpm" in tempo_npz.files:
            tempo_bpm_all = np.asarray(tempo_npz["tempo_bpm"], dtype=np.float32)
        else:
            tempo_embeddings = np.asarray(tempo_npz["embeddings"], dtype=np.float32)
            tempo_bpm_all = tempo_embeddings[:, 0].astype(np.float32)

        if "tempo_confidence" in tempo_npz.files:
            tempo_confidence_all = np.asarray(tempo_npz["tempo_confidence"], dtype=np.float32)
        else:
            tempo_embeddings = np.asarray(tempo_npz["embeddings"], dtype=np.float32)
            if tempo_embeddings.ndim == 2 and tempo_embeddings.shape[1] >= 2:
                tempo_confidence_all = tempo_embeddings[:, 1].astype(np.float32)
            else:
                tempo_confidence_all = np.ones_like(tempo_bpm_all, dtype=np.float32)

        maest_index = _track_number_index(maest_npz, path=paths["maest"])
        chroma_index = _track_number_index(chroma_npz, path=paths["chroma"])
        tempo_index = _track_number_index(tempo_npz, path=paths["tempo"])
        groove_index = _track_number_index(groove_npz, path=paths["groove"])

        manifest_rows = _manifest_track_order(manifest)
        if manifest_rows:
            ordered_numbers = [
                int(row["track_number"])
                for row in manifest_rows
                if row.get("track_number") is not None
            ]
            manifest_by_number = {int(row["track_number"]): row for row in manifest_rows}
        else:
            ordered_numbers = sorted(maest_index)
            manifest_by_number = {}

        titles = _as_string_array(maest_npz, "titles", "", maest_embeddings_all.shape[0])
        artists = _as_string_array(maest_npz, "artists", "", maest_embeddings_all.shape[0])
        filenames = _as_string_array(maest_npz, "filenames", "", maest_embeddings_all.shape[0])
        genres = _as_string_array(maest_npz, "genres", "", maest_embeddings_all.shape[0])

        chroma_pitch_means = None
        if "chroma_pitch_class_mean" in chroma_npz.files:
            pitch_means = np.asarray(chroma_npz["chroma_pitch_class_mean"], dtype=np.float32)
            if pitch_means.ndim == 2 and pitch_means.shape[1] >= 12:
                chroma_pitch_means = pitch_means[:, :12]

        chroma_center_baseline: float | None = None
        if "config_center_baseline" in chroma_npz.files:
            baseline = np.asarray(chroma_npz["config_center_baseline"], dtype=np.float32).reshape(-1)
            if baseline.size and np.isfinite(float(baseline[0])):
                chroma_center_baseline = float(baseline[0])
        if chroma_pitch_means is not None and chroma_center_baseline is None:
            mean_sum = float(np.nanmean(np.sum(chroma_pitch_means, axis=1)))
            if np.isfinite(mean_sum) and abs(mean_sum) < 1e-3:
                chroma_center_baseline = 1.0 / 12.0

        missing: list[int] = []
        metadata: list[SongMetadata] = []
        maest_vectors: list[np.ndarray] = []
        chroma_vectors: list[np.ndarray] = []
        chroma_pitch_vectors: list[np.ndarray] = []
        tempo_bpms: list[float] = []
        tempo_confidences: list[float] = []
        groove_vectors: list[np.ndarray] = []

        for track_number in ordered_numbers:
            if (
                track_number not in maest_index
                or track_number not in chroma_index
                or track_number not in tempo_index
                or track_number not in groove_index
            ):
                missing.append(track_number)
                continue

            mi = maest_index[track_number]
            ci = chroma_index[track_number]
            ti = tempo_index[track_number]
            gi = groove_index[track_number]

            maest_vec = np.asarray(maest_embeddings_all[mi], dtype=np.float32).reshape(-1)
            chroma_full = np.asarray(chroma_embeddings_all[ci], dtype=np.float32).reshape(-1)
            groove_vec = np.asarray(groove_embeddings_all[gi], dtype=np.float32).reshape(-1)
            if not maest_vec.size or not chroma_full.size or not groove_vec.size:
                missing.append(track_number)
                continue

            if chroma_pitch_means is not None and ci < chroma_pitch_means.shape[0]:
                chroma_pitch = np.asarray(chroma_pitch_means[ci], dtype=np.float32).reshape(-1)
                if chroma_center_baseline is not None:
                    chroma_pitch = chroma_pitch + chroma_center_baseline
            elif chroma_full.size >= 12:
                chroma_pitch = chroma_full[:12].astype(np.float32)
            else:
                missing.append(track_number)
                continue

            chroma_pitch = np.clip(chroma_pitch, 0.0, None).astype(np.float32)
            if chroma_pitch.size != 12 or float(np.sum(chroma_pitch)) <= 0.0:
                missing.append(track_number)
                continue

            row = manifest_by_number.get(track_number, {})
            title = str(row.get("track_name") or titles[mi] or Path(str(filenames[mi])).stem)
            artist = str(row.get("artist") or artists[mi] or "")
            filename = str(filenames[mi] or f"{track_number:04d}")
            genre = str(row.get("genre") or genres[mi] or "")

            metadata.append(
                SongMetadata(
                    track_number=int(track_number),
                    title=title,
                    artist=artist,
                    filename=filename,
                    genre=genre,
                )
            )
            maest_vectors.append(maest_vec)
            # The app's harmonic matrix uses chroma_pitch; chroma is kept for API completeness.
            chroma_vectors.append(chroma_pitch)
            chroma_pitch_vectors.append(chroma_pitch)
            tempo_bpms.append(float(tempo_bpm_all[ti]) if ti < tempo_bpm_all.size else float("nan"))
            tempo_confidences.append(
                float(tempo_confidence_all[ti]) if ti < tempo_confidence_all.size else 1.0
            )
            groove_vectors.append(groove_vec)

    if missing and not allow_missing_features:
        preview = ", ".join(str(value) for value in missing[:12])
        suffix = "" if len(missing) <= 12 else f", ... ({len(missing)} total)"
        raise ValueError(
            f"Missing or invalid feature rows for track_number(s): {preview}{suffix}. "
            "Use --allow-missing-features to evaluate the remaining aligned tracks."
        )

    if len(metadata) < 3:
        raise ValueError(f"Need at least 3 aligned tracks to evaluate a mix; found {len(metadata)}.")

    feature_set = SongFeatureSet(
        metadata=metadata,
        maest=np.vstack(maest_vectors).astype(np.float32),
        chroma=np.vstack(chroma_vectors).astype(np.float32),
        chroma_pitch=np.vstack(chroma_pitch_vectors).astype(np.float32),
        tempo_bpm=np.asarray(tempo_bpms, dtype=np.float32),
        tempo_confidence=np.asarray(tempo_confidences, dtype=np.float32),
    )
    groove_embeddings = np.vstack(groove_vectors).astype(np.float32)
    return feature_set, groove_embeddings, paths


def normalize_distance_matrix_for_evaluation(
    distance: np.ndarray,
    *,
    percentile: float = DEFAULT_NORMALIZATION_DISTANCE_PERCENTILE,
    top_nonself_score_anchor: bool = True,
) -> np.ndarray:
    normalized = np.asarray(distance, dtype=np.float32).copy()
    if normalized.ndim != 2 or normalized.shape[0] != normalized.shape[1]:
        raise ValueError(f"Expected square distance matrix, got shape={normalized.shape}.")

    normalized = 0.5 * (normalized + normalized.T)
    np.fill_diagonal(normalized, 0.0)

    mask = ~np.eye(normalized.shape[0], dtype=bool)
    finite_values = normalized[mask]
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Distance matrix has no finite off-diagonal values.")

    lower = float(np.min(finite_values)) if top_nonself_score_anchor else 0.0
    upper = float(np.percentile(finite_values, percentile))
    if not np.isfinite(upper) or upper <= lower:
        upper = float(np.max(finite_values))
    if not np.isfinite(upper) or upper <= lower:
        if np.allclose(finite_values, finite_values[0]):
            np.fill_diagonal(normalized, 0.0)
            return np.zeros_like(normalized, dtype=np.float32)
        raise ValueError(f"Distance matrix scale is invalid: lower={lower:.6g}, upper={upper:.6g}.")

    normalized = np.clip((normalized - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(normalized, 0.0)
    return normalized


def normalized_component_scores(
    matrices: dict[str, np.ndarray],
    *,
    percentile: float,
    top_nonself_score_anchor: bool,
) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for name in ("maest", "tempo", "groove", "chroma"):
        distance = normalize_distance_matrix_for_evaluation(
            1.0 - np.asarray(matrices[f"{name}_similarity"], dtype=np.float32),
            percentile=percentile,
            top_nonself_score_anchor=top_nonself_score_anchor,
        )
        scores[name] = (1.0 - distance).astype(np.float64)
    return scores


def rhythm_component_score(
    tempo_score: np.ndarray,
    groove_score: np.ndarray,
    *,
    tempo_weight: float = DEFAULT_RHYTHM_TEMPO_WEIGHT,
) -> np.ndarray:
    """Backward-compatible argument name for the canonical rhythm blend."""
    return _rhythm_component_score(
        tempo_score,
        groove_score,
        rhythm_tempo_weight=tempo_weight,
    )


def simplex_weight_grid(step: float = DEFAULT_SIMPLEX_STEP) -> pd.DataFrame:
    units = int(round(1.0 / float(step)))
    if units <= 0 or not np.isclose(units * float(step), 1.0):
        raise ValueError("simplex step must evenly divide 1.0, e.g. 0.1, 0.05, or 0.025.")

    rows = []
    for style_units in range(units + 1):
        for rhythm_units in range(units - style_units + 1):
            harmony_units = units - style_units - rhythm_units
            rows.append(
                {
                    "style_weight": style_units / units,
                    "rhythm_weight": rhythm_units / units,
                    "harmony_weight": harmony_units / units,
                }
            )
    return pd.DataFrame(rows)


def simplex_xy(
    style_weight: np.ndarray,
    rhythm_weight: np.ndarray,
    harmony_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    style_weight = np.asarray(style_weight, dtype=np.float64)
    rhythm_weight = np.asarray(rhythm_weight, dtype=np.float64)
    harmony_weight = np.asarray(harmony_weight, dtype=np.float64)
    del rhythm_weight
    x = harmony_weight + 0.5 * style_weight
    y = (np.sqrt(3.0) / 2.0) * style_weight
    return x, y


def rank_percentiles_from_ranks(ranks: np.ndarray, candidate_count: int) -> np.ndarray:
    if candidate_count <= 1:
        return np.ones_like(np.asarray(ranks, dtype=np.float64))
    return 1.0 - (np.asarray(ranks, dtype=np.float64) - 1.0) / (candidate_count - 1.0)


def rank_metrics_for_similarity(
    similarity: np.ndarray,
    order: np.ndarray,
    *,
    candidate_mode: str = "all",
) -> dict[str, float]:
    if candidate_mode not in {"all", "future"}:
        raise ValueError("candidate_mode must be 'all' or 'future'.")

    n = int(similarity.shape[0])
    ranks: list[int] = []
    edge_scores: list[float] = []
    candidate_counts: list[int] = []

    for position, (source, target) in enumerate(zip(order[:-1], order[1:], strict=True)):
        if candidate_mode == "future":
            candidates = np.asarray(order[position + 1 :], dtype=np.int64)
            candidates = candidates[candidates != source]
        else:
            candidates = np.asarray([index for index in range(n) if index != source], dtype=np.int64)

        if target not in set(candidates.tolist()):
            continue

        ranked = candidates[np.argsort(similarity[source, candidates])[::-1]]
        ranks.append(int(np.flatnonzero(ranked == target)[0]) + 1)
        edge_scores.append(float(similarity[source, target]))
        candidate_counts.append(int(candidates.size))

    if not ranks:
        return {
            "mrr": float("nan"),
            "median_rank": float("nan"),
            "mean_rank_percentile": float("nan"),
            "median_rank_percentile": float("nan"),
            "mean_actual_transition_score": float("nan"),
            "recall_at_1": float("nan"),
            "recall_at_3": float("nan"),
            "recall_at_5": float("nan"),
            "top_10_percent_hit": float("nan"),
            "top_20_percent_hit": float("nan"),
            "top_25_percent_hit": float("nan"),
        }

    rank_array = np.asarray(ranks, dtype=np.float64)
    edge_score_array = np.asarray(edge_scores, dtype=np.float64)
    percentiles = np.asarray(
        [
            rank_percentiles_from_ranks(np.asarray([rank]), candidate_count)[0]
            for rank, candidate_count in zip(rank_array, candidate_counts, strict=True)
        ],
        dtype=np.float64,
    )
    return {
        "mrr": float(np.mean(1.0 / rank_array)),
        "median_rank": float(np.median(rank_array)),
        "mean_rank_percentile": float(percentiles.mean()),
        "median_rank_percentile": float(np.median(percentiles)),
        "mean_actual_transition_score": float(edge_score_array.mean()),
        "recall_at_1": float((rank_array <= 1).mean()),
        "recall_at_3": float((rank_array <= 3).mean()),
        "recall_at_5": float((rank_array <= 5).mean()),
        "top_10_percent_hit": float((percentiles >= 0.90).mean()),
        "top_20_percent_hit": float((percentiles >= 0.80).mean()),
        "top_25_percent_hit": float((percentiles >= 0.75).mean()),
    }


def _with_prefix(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def adjacent_component_means(component_scores: list[np.ndarray], order: np.ndarray) -> np.ndarray:
    sources = order[:-1]
    targets = order[1:]
    return np.asarray([float(np.mean(score[sources, targets])) for score in component_scores], dtype=np.float64)


def random_component_mean_bank(
    component_scores: list[np.ndarray],
    order: np.ndarray,
    *,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
    progress_every: int = 50_000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_permutations = int(n_permutations)
    out = np.empty((n_permutations, len(component_scores)), dtype=np.float64)
    for index in range(n_permutations):
        shuffled = rng.permutation(order)
        sources = shuffled[:-1]
        targets = shuffled[1:]
        out[index] = [float(np.mean(score[sources, targets])) for score in component_scores]
        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(f"permutations: {index + 1:,}/{n_permutations:,}", file=sys.stderr, flush=True)
    return out


def simplex_permutation_metrics(
    weights: np.ndarray,
    real_component_means: np.ndarray,
    random_component_means: np.ndarray,
) -> dict[str, float]:
    observed = float(real_component_means @ weights)
    random_means = random_component_means @ weights
    random_mean = float(np.mean(random_means))
    random_std = float(np.std(random_means, ddof=1))
    lift = observed - random_mean
    return {
        "observed_mean_score": observed,
        "random_mean_score": random_mean,
        "random_score_std": random_std,
        "lift": float(lift),
        "relative_lift": float(lift / random_mean) if random_mean else float("nan"),
        "z_score": float(lift / random_std) if random_std > 0.0 else float("nan"),
        "permutation_p": float((np.sum(random_means >= observed) + 1) / (len(random_means) + 1)),
    }


def evaluate_simplex(
    *,
    style_score: np.ndarray,
    rhythm_score: np.ndarray,
    harmony_score: np.ndarray,
    order: np.ndarray,
    step: float,
    n_permutations: int,
    seed: int,
    progress_every: int,
) -> pd.DataFrame:
    component_scores = [style_score, rhythm_score, harmony_score]
    real_component_means = adjacent_component_means(component_scores, order)
    random_component_means = random_component_mean_bank(
        component_scores,
        order,
        n_permutations=n_permutations,
        seed=seed,
        progress_every=progress_every,
    )

    grid = simplex_weight_grid(step)
    metric_rows: list[dict[str, float]] = []
    for row in grid.itertuples(index=False):
        weights = np.asarray([row.style_weight, row.rhythm_weight, row.harmony_weight], dtype=np.float64)
        similarity = (
            weights[0] * style_score
            + weights[1] * rhythm_score
            + weights[2] * harmony_score
        )
        metric_rows.append(
            {
                **rank_metrics_for_similarity(similarity, order, candidate_mode="all"),
                **_with_prefix(
                    rank_metrics_for_similarity(similarity, order, candidate_mode="future"),
                    "future_",
                ),
                **simplex_permutation_metrics(weights, real_component_means, random_component_means),
            }
        )

    result = pd.concat([grid, pd.DataFrame(metric_rows)], axis=1)
    result["x"], result["y"] = simplex_xy(
        result["style_weight"],
        result["rhythm_weight"],
        result["harmony_weight"],
    )
    return result


def _component_matrices_for_features(args: argparse.Namespace, features: SongFeatureSet, groove: np.ndarray) -> dict[str, np.ndarray]:
    maest_cos = _pairwise_cosine_similarity_matrix(features.maest)
    maest_similarity = np.clip(0.5 * (maest_cos + 1.0), 0.0, 1.0).astype(np.float32)

    tempo_similarity = _pairwise_tempo_similarity_matrix(
        features.tempo_bpm,
        features.tempo_confidence,
        bandwidth=float(args.tempo_bandwidth),
        decay=float(args.tempo_decay),
        allow_octave=not bool(args.no_tempo_octave),
        octave_penalty=float(args.tempo_octave_penalty),
        tempo_similarity_shape=str(args.tempo_similarity_shape),
        softflat_sharpness=float(args.tempo_softflat_sharpness),
        use_confidence=bool(args.tempo_use_confidence),
    )

    harmonic_kernel = _build_harmonic_kernel(
        exact_weight=float(args.harmonic_exact_weight),
        first_fifth_weight=float(args.harmonic_first_fifth_weight),
        second_fifth_weight=float(args.harmonic_second_fifth_weight),
        other_weight=float(args.harmonic_other_weight),
    )
    chroma_similarity = _pairwise_fifth_aware_similarity_matrix(
        features.chroma_pitch,
        kernel=harmonic_kernel,
        normalize_by_self=not bool(args.no_harmonic_self_normalize),
    )
    chroma_similarity = np.clip(chroma_similarity, 0.0, 1.0).astype(np.float32)

    groove_arr = np.asarray(groove, dtype=np.float32)
    if groove_arr.ndim != 2 or groove_arr.shape[0] != features.maest.shape[0]:
        raise ValueError(
            f"Groove embeddings must have shape [tracks, dims] aligned with features. "
            f"Got {groove_arr.shape}, expected first dimension {features.maest.shape[0]}."
        )
    groove_cos = _pairwise_cosine_similarity_matrix(groove_arr)
    groove_similarity = np.clip(0.5 * (groove_cos + 1.0), 0.0, 1.0).astype(np.float32)

    return {
        "maest_similarity": maest_similarity,
        "tempo_similarity": tempo_similarity,
        "chroma_similarity": chroma_similarity,
        "groove_similarity": groove_similarity,
    }


def build_summary(metrics: pd.DataFrame, *, slug: str, output_csv: Path) -> dict[str, Any]:
    best_mrr = metrics.loc[metrics["mrr"].idxmax()].to_dict()
    best_z = metrics.loc[metrics["z_score"].idxmax()].to_dict()
    return {
        "mix_slug": slug,
        "output_csv": str(output_csv),
        "rows": int(metrics.shape[0]),
        "best_mrr": best_mrr,
        "best_z_score": best_z,
    }


def write_outputs(
    metrics: pd.DataFrame,
    *,
    output_csv: Path,
    summary_json: Path | None,
    slug: str,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_csv, index=False)

    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary = build_summary(metrics, slug=slug, output_csv=output_csv)
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a rendered-set or canonical tracklist-build manifest over a "
            "style/rhythm/harmony simplex. "
            "Rhythm is tempo/groove combined after score normalization."
        )
    )
    parser.add_argument("manifests", nargs="+", type=Path, help="Rendered-set manifest JSON(s).")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project root.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "mix_evaluation",
        help="Default directory for <slug>_simplex_metrics.csv.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary JSON path. Defaults to <slug>_simplex_summary.json next to the CSV.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=None,
        help="Optional CSV path containing concatenated rows from all evaluated manifests.",
    )
    parser.add_argument("--no-summary-json", action="store_true", help="Do not write the summary JSON.")

    parser.add_argument("--maest-npz", type=Path, default=None, help="Override MAEST NPZ path.")
    parser.add_argument("--chroma-npz", type=Path, default=None, help="Override chroma NPZ path.")
    parser.add_argument("--tempo-npz", type=Path, default=None, help="Override tempo NPZ path.")
    parser.add_argument("--groove-npz", type=Path, default=None, help="Override groove NPZ path.")
    parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="Evaluate only aligned tracks if some manifest rows are missing feature rows.",
    )

    parser.add_argument("--simplex-step", type=float, default=DEFAULT_SIMPLEX_STEP)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--rhythm-tempo-weight", type=float, default=DEFAULT_RHYTHM_TEMPO_WEIGHT)
    parser.add_argument(
        "--normalization-distance-percentile",
        type=float,
        default=DEFAULT_NORMALIZATION_DISTANCE_PERCENTILE,
    )
    parser.add_argument(
        "--app-style-normalization",
        action="store_true",
        help="Use zero-distance anchor instead of the top-nonself score anchor.",
    )

    parser.add_argument("--tempo-bandwidth", type=float, default=0.06)
    parser.add_argument("--tempo-decay", type=float, default=0.5)
    parser.add_argument("--no-tempo-octave", action="store_true")
    parser.add_argument("--tempo-octave-penalty", type=float, default=0.5)
    parser.add_argument("--tempo-similarity-shape", default="gaussian")
    parser.add_argument("--tempo-softflat-sharpness", type=float, default=8.0)
    parser.add_argument("--tempo-use-confidence", action="store_true")

    parser.add_argument("--harmonic-exact-weight", type=float, default=1.0)
    parser.add_argument("--harmonic-first-fifth-weight", type=float, default=0.2)
    parser.add_argument("--harmonic-second-fifth-weight", type=float, default=0.0)
    parser.add_argument("--harmonic-other-weight", type=float, default=0.0)
    parser.add_argument("--no-harmonic-self-normalize", action="store_true")
    args = parser.parse_args(argv)
    if len(args.manifests) > 1 and args.output is not None:
        parser.error("--output can only be used with a single manifest; use --output-dir for batches.")
    if len(args.manifests) > 1 and args.summary_json is not None:
        parser.error("--summary-json can only be used with a single manifest.")
    return args


def evaluate_manifest(args: argparse.Namespace, *, manifest_path: Path, project_root: Path) -> pd.DataFrame:
    manifest = load_manifest(manifest_path)
    slug = slug_from_manifest_path(manifest_path)

    output_csv = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (args.output_dir.expanduser().resolve() / f"{slug}_simplex_metrics.csv")
    )
    summary_json = None
    if not args.no_summary_json:
        summary_json = (
            args.summary_json.expanduser().resolve()
            if args.summary_json is not None
            else output_csv.with_name(f"{slug}_simplex_summary.json")
        )

    features, groove_embeddings, paths = load_rendered_set_feature_set(
        manifest=manifest,
        slug=slug,
        project_root=project_root,
        maest_npz_path=args.maest_npz,
        chroma_npz_path=args.chroma_npz,
        tempo_npz_path=args.tempo_npz,
        groove_npz_path=args.groove_npz,
        allow_missing_features=bool(args.allow_missing_features),
    )
    print(f"mix: {slug}", file=sys.stderr)
    print(f"tracks: {len(features.metadata)}", file=sys.stderr)
    for name, path in paths.items():
        print(f"{name}: {path}", file=sys.stderr)

    matrices = _component_matrices_for_features(args, features, groove_embeddings)
    top_nonself = not bool(args.app_style_normalization)
    scores = normalized_component_scores(
        matrices,
        percentile=float(args.normalization_distance_percentile),
        top_nonself_score_anchor=top_nonself,
    )
    rhythm_score = rhythm_component_score(
        scores["tempo"],
        scores["groove"],
        tempo_weight=float(args.rhythm_tempo_weight),
    )
    order = np.arange(len(features.metadata), dtype=np.int64)

    metrics = evaluate_simplex(
        style_score=scores["maest"],
        rhythm_score=rhythm_score,
        harmony_score=scores["chroma"],
        order=order,
        step=float(args.simplex_step),
        n_permutations=int(args.permutations),
        seed=int(args.random_seed),
        progress_every=int(args.progress_every),
    )
    metrics.insert(0, "mix_slug", slug)
    metrics.insert(1, "n_tracks", len(features.metadata))
    metrics.insert(2, "n_transitions", len(features.metadata) - 1)
    metrics.insert(3, "permutations", int(args.permutations))
    metrics.insert(4, "random_seed", int(args.random_seed))
    metrics.insert(5, "simplex_step", float(args.simplex_step))
    metrics.insert(6, "rhythm_tempo_weight", float(args.rhythm_tempo_weight))
    metrics.insert(7, "rhythm_groove_weight", 1.0 - float(args.rhythm_tempo_weight))
    metrics.insert(8, "normalization", "top_nonself_anchor" if top_nonself else "app_style")
    metrics.insert(9, "normalization_distance_percentile", float(args.normalization_distance_percentile))

    write_outputs(metrics, output_csv=output_csv, summary_json=summary_json, slug=slug)
    print(f"wrote: {output_csv}", file=sys.stderr)
    if summary_json is not None:
        print(f"wrote: {summary_json}", file=sys.stderr)

    best_mrr = metrics.loc[metrics["mrr"].idxmax()]
    best_z = metrics.loc[metrics["z_score"].idxmax()]
    print(
        "best MRR: "
        f"{best_mrr['mrr']:.4f} at "
        f"S={best_mrr['style_weight']:.2f}, R={best_mrr['rhythm_weight']:.2f}, "
        f"H={best_mrr['harmony_weight']:.2f}",
        file=sys.stderr,
    )
    print(
        "best z-score: "
        f"{best_z['z_score']:.3f} at "
        f"S={best_z['style_weight']:.2f}, R={best_z['rhythm_weight']:.2f}, "
        f"H={best_z['harmony_weight']:.2f}",
        file=sys.stderr,
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.expanduser().resolve()

    all_metrics: list[pd.DataFrame] = []
    for manifest_path in args.manifests:
        metrics = evaluate_manifest(
            args,
            manifest_path=manifest_path.expanduser().resolve(),
            project_root=project_root,
        )
        all_metrics.append(metrics)

    if args.combined_output is not None:
        combined_output = args.combined_output.expanduser().resolve()
        combined_output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(all_metrics, ignore_index=True).to_csv(combined_output, index=False)
        print(f"wrote: {combined_output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
