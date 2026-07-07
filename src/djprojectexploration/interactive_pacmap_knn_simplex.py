"""Export the canonical interactive DJ embedding visualization."""

from __future__ import annotations

import argparse
import html
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from djprojectexploration.interactive_visualization_common import (
    PROJECT_ROOT,
    _align_to_reference,
    _build_plot,
    _json_script_payload,
    _load_combined_records_and_features,
    _neighbor_pairs_from_distance,
    _normalize_distance_matrix,
)
from djprojectexploration.multimodal_compatibility import (
    _build_harmonic_kernel,
    _pairwise_cosine_similarity_matrix,
    _pairwise_fifth_aware_similarity_matrix,
    _pairwise_tempo_similarity_matrix,
)

PACMAP_PAIR_SOURCE_CHOICES = ("neighbors-only", "combined-all")
DISTANCE_COMBINE_CHOICES = ("l2", "l1")
LAYOUT_INIT_CHOICES = ("neighbor", "pca", "random")
CONTROL_MODE_CHOICES = ("genre-mixability", "legacy-simplex", "legacy-discrete-simplex")
REDUCER_CHOICES = ("pacmap", "umap")


def _format_setting_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _build_settings_panel(settings: dict[str, Any]) -> str:
    rows = "\n".join(
        (
            "      <dt>"
            f"{html.escape(label)}"
            "</dt><dd>"
            f"{html.escape(_format_setting_value(value))}"
            "</dd>"
        )
        for label, value in settings.items()
    )
    return f"""
  <div class="box settings-panel">
    <h2>Generation Settings</h2>
    <dl>
{rows}
    </dl>
  </div>
"""


def _simplex_grid(step: float) -> list[tuple[float, float, float]]:
    scale = int(round(1.0 / float(step)))
    if scale <= 0:
        raise ValueError("step must be > 0.")
    return [
        (i / scale, j / scale, (scale - i - j) / scale)
        for i in range(scale + 1)
        for j in range(scale + 1 - i)
    ]


def _simplex_grid_4way(step: float) -> list[tuple[float, float, float, float]]:
    scale = int(round(1.0 / float(step)))
    if scale <= 0:
        raise ValueError("step must be > 0.")
    return [
        (i / scale, j / scale, k / scale, (scale - i - j - k) / scale)
        for i in range(scale + 1)
        for j in range(scale + 1 - i)
        for k in range(scale + 1 - i - j)
    ]


def _simplex_key(weights: tuple[float, float, float]) -> str:
    return ",".join(f"{w:.1f}" for w in weights)


def _simplex_key_4way(weights: tuple[float, float, float, float]) -> str:
    return ",".join(f"{w:.1f}" for w in weights)


def _component_matrices_3way(
    features,
    *,
    tempo_bandwidth: float,
    tempo_decay: float,
    tempo_allow_octave: bool,
    tempo_octave_penalty: float,
    tempo_similarity_shape: str,
    tempo_softflat_sharpness: float,
    tempo_use_confidence: bool,
    harmonic_exact_weight: float,
    harmonic_first_fifth_weight: float,
    harmonic_second_fifth_weight: float,
    harmonic_other_weight: float,
    harmonic_self_normalize: bool,
) -> dict[str, np.ndarray]:
    maest_cos = _pairwise_cosine_similarity_matrix(features.maest)
    maest_similarity = np.clip(0.5 * (maest_cos + 1.0), 0.0, 1.0).astype(np.float32)

    tempo_similarity = _pairwise_tempo_similarity_matrix(
        features.tempo_bpm,
        features.tempo_confidence,
        bandwidth=tempo_bandwidth,
        decay=tempo_decay,
        allow_octave=tempo_allow_octave,
        octave_penalty=tempo_octave_penalty,
        tempo_similarity_shape=tempo_similarity_shape,
        softflat_sharpness=tempo_softflat_sharpness,
        use_confidence=tempo_use_confidence,
    )
    tempo_similarity = np.clip(tempo_similarity, 0.0, 1.0).astype(np.float32)

    harmonic_kernel = _build_harmonic_kernel(
        exact_weight=harmonic_exact_weight,
        first_fifth_weight=harmonic_first_fifth_weight,
        second_fifth_weight=harmonic_second_fifth_weight,
        other_weight=harmonic_other_weight,
    )
    chroma_similarity = _pairwise_fifth_aware_similarity_matrix(
        features.chroma_pitch,
        kernel=harmonic_kernel,
        normalize_by_self=harmonic_self_normalize,
    )
    chroma_similarity = np.clip(chroma_similarity, 0.0, 1.0).astype(np.float32)

    return {
        "maest_similarity": maest_similarity,
        "tempo_similarity": tempo_similarity,
        "chroma_similarity": chroma_similarity,
        "maest_distance": _normalize_distance_matrix(1.0 - maest_similarity),
        "tempo_distance": _normalize_distance_matrix(1.0 - tempo_similarity),
        "chroma_distance": _normalize_distance_matrix(1.0 - chroma_similarity),
    }


def _load_combined_groove_embeddings(
    *,
    project_root: Path,
    mix_slugs: list[str],
    groove_dir: Path,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for mix_slug in mix_slugs:
        csv_stem = f"{mix_slug.replace('-', '_')}_tracks"
        groove_file = groove_dir / f"{csv_stem}.npz"
        if not groove_file.exists():
            raise FileNotFoundError(
                f"Groove embedding collection not found: {groove_file}. "
                "Generate it with `uv run djprojectexploration-groove-playlist "
                f"music/{mix_slug}/{csv_stem}.csv`."
            )
        with np.load(groove_file) as data:
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(f"Invalid groove embeddings in {groove_file}: shape={embeddings.shape}")
        chunks.append(embeddings)
    del project_root
    return np.vstack(chunks).astype(np.float32)


def _component_matrices_4way(
    features,
    *,
    groove_embeddings: np.ndarray,
    tempo_bandwidth: float,
    tempo_decay: float,
    tempo_allow_octave: bool,
    tempo_octave_penalty: float,
    tempo_similarity_shape: str,
    tempo_softflat_sharpness: float,
    tempo_use_confidence: bool,
    harmonic_exact_weight: float,
    harmonic_first_fifth_weight: float,
    harmonic_second_fifth_weight: float,
    harmonic_other_weight: float,
    harmonic_self_normalize: bool,
) -> dict[str, np.ndarray]:
    matrices = _component_matrices_3way(
        features,
        tempo_bandwidth=tempo_bandwidth,
        tempo_decay=tempo_decay,
        tempo_allow_octave=tempo_allow_octave,
        tempo_octave_penalty=tempo_octave_penalty,
        tempo_similarity_shape=tempo_similarity_shape,
        tempo_softflat_sharpness=tempo_softflat_sharpness,
        tempo_use_confidence=tempo_use_confidence,
        harmonic_exact_weight=harmonic_exact_weight,
        harmonic_first_fifth_weight=harmonic_first_fifth_weight,
        harmonic_second_fifth_weight=harmonic_second_fifth_weight,
        harmonic_other_weight=harmonic_other_weight,
        harmonic_self_normalize=harmonic_self_normalize,
    )
    groove = np.asarray(groove_embeddings, dtype=np.float32)
    if groove.ndim != 2 or groove.shape[0] != features.maest.shape[0]:
        raise ValueError(
            f"Groove embeddings must have shape [tracks, dims] aligned with features. "
            f"Got {groove.shape}, expected first dimension {features.maest.shape[0]}."
        )
    groove_cos = _pairwise_cosine_similarity_matrix(groove)
    groove_similarity = np.clip(0.5 * (groove_cos + 1.0), 0.0, 1.0).astype(np.float32)
    matrices["groove_similarity"] = groove_similarity
    matrices["groove_distance"] = _normalize_distance_matrix(1.0 - groove_similarity)
    return matrices


def _combined_distance_3way(
    D_maest: np.ndarray,
    D_tempo: np.ndarray,
    D_chroma: np.ndarray,
    weights: tuple[float, float, float],
    *,
    combine_mode: str = "l2",
) -> np.ndarray:
    if combine_mode not in DISTANCE_COMBINE_CHOICES:
        raise ValueError(f"combine_mode must be one of {DISTANCE_COMBINE_CHOICES}, got {combine_mode!r}.")

    wm, wt, wc = [float(np.clip(v, 0.0, 1.0)) for v in weights]
    total = max(wm + wt + wc, 1e-12)
    wm, wt, wc = wm / total, wt / total, wc / total
    if combine_mode == "l1":
        D = ((wm * D_maest) + (wt * D_tempo) + (wc * D_chroma)).astype(np.float32)
    else:
        D = np.sqrt((wm * D_maest**2) + (wt * D_tempo**2) + (wc * D_chroma**2)).astype(np.float32)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def _combined_distance_4way(
    D_maest: np.ndarray,
    D_tempo: np.ndarray,
    D_groove: np.ndarray,
    D_chroma: np.ndarray,
    weights: tuple[float, float, float, float],
    *,
    combine_mode: str = "l2",
) -> np.ndarray:
    if combine_mode not in DISTANCE_COMBINE_CHOICES:
        raise ValueError(f"combine_mode must be one of {DISTANCE_COMBINE_CHOICES}, got {combine_mode!r}.")

    wm, wt, wg, wc = [float(np.clip(v, 0.0, 1.0)) for v in weights]
    total = max(wm + wt + wg + wc, 1e-12)
    wm, wt, wg, wc = wm / total, wt / total, wg / total, wc / total
    if combine_mode == "l1":
        D = ((wm * D_maest) + (wt * D_tempo) + (wg * D_groove) + (wc * D_chroma)).astype(np.float32)
    else:
        D = np.sqrt(
            (wm * D_maest**2) + (wt * D_tempo**2) + (wg * D_groove**2) + (wc * D_chroma**2)
        ).astype(np.float32)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def _simplex_int_weights(weights: tuple[float, float, float], *, scale: int) -> tuple[int, int, int]:
    values = [int(round(float(w) * scale)) for w in weights]
    diff = scale - sum(values)
    if diff:
        fractional = [float(w) * scale - int(round(float(w) * scale)) for w in weights]
        order = np.argsort(fractional)
        if diff > 0:
            order = order[::-1]
        for idx in order[: abs(diff)]:
            values[int(idx)] += 1 if diff > 0 else -1
    return int(values[0]), int(values[1]), int(values[2])


def _simplex_float_weights(int_weights: tuple[int, int, int], *, scale: int) -> tuple[float, float, float]:
    return tuple(v / scale for v in int_weights)  # type: ignore[return-value]


def _simplex_int_neighbors(
    int_weights: tuple[int, int, int],
    *,
    scale: int,
) -> list[tuple[int, int, int]]:
    deltas = [
        (1, -1, 0),
        (-1, 1, 0),
        (1, 0, -1),
        (-1, 0, 1),
        (0, 1, -1),
        (0, -1, 1),
    ]
    out: list[tuple[int, int, int]] = []
    for da, db, dc in deltas:
        candidate = (int_weights[0] + da, int_weights[1] + db, int_weights[2] + dc)
        if all(v >= 0 for v in candidate) and sum(candidate) == scale:
            out.append(candidate)
    return out


def _central_simplex_anchor(grid_ints: set[tuple[int, int, int]], *, scale: int) -> tuple[int, int, int]:
    target = np.array([scale / 3.0, scale / 3.0, scale / 3.0], dtype=np.float64)
    return min(
        grid_ints,
        key=lambda w: (float(np.sum((np.asarray(w, dtype=np.float64) - target) ** 2)), -w[0], -w[1], -w[2]),
    )


def _neighbor_aware_simplex_order(
    grid: list[tuple[float, float, float]],
    *,
    scale: int,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int] | None]]:
    grid_ints = {_simplex_int_weights(weights, scale=scale) for weights in grid}
    anchor = _central_simplex_anchor(grid_ints, scale=scale)
    target = np.asarray(anchor, dtype=np.float64)
    order: list[tuple[tuple[int, int, int], tuple[int, int, int] | None]] = []
    visited = {anchor}
    queue: list[tuple[int, int, int]] = [anchor]
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {anchor: None}

    while queue:
        current = queue.pop(0)
        order.append((current, parent[current]))
        neighbors = [n for n in _simplex_int_neighbors(current, scale=scale) if n in grid_ints and n not in visited]
        neighbors.sort(
            key=lambda w: (
                float(np.sum((np.asarray(w, dtype=np.float64) - target) ** 2)),
                -w[0],
                -w[1],
                -w[2],
            )
        )
        for neighbor in neighbors:
            visited.add(neighbor)
            parent[neighbor] = current
            queue.append(neighbor)

    if len(order) != len(grid_ints):
        missing = sorted(grid_ints - visited, key=lambda w: (-w[0], -w[1], -w[2]))
        order.extend((weights, None) for weights in missing)
    return order


def _alignment_residual(reference: np.ndarray, coords: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(coords, dtype=np.float64)
    if ref.shape != cur.shape or ref.size == 0:
        return float("inf")
    return float(np.linalg.norm(ref - cur) / np.sqrt(ref.shape[0]))


def _align_to_best_neighbor(
    coords: np.ndarray,
    *,
    int_weights: tuple[int, int, int],
    aligned_arrays: dict[tuple[int, int, int], np.ndarray],
    scale: int,
) -> np.ndarray:
    candidates = [
        neighbor
        for neighbor in _simplex_int_neighbors(int_weights, scale=scale)
        if neighbor in aligned_arrays
    ]
    if not candidates:
        return coords

    best_coords: np.ndarray | None = None
    best_residual = float("inf")
    for candidate in candidates:
        aligned = _align_to_reference(aligned_arrays[candidate], coords)
        residual = _alignment_residual(aligned_arrays[candidate], aligned)
        if residual < best_residual:
            best_coords = aligned
            best_residual = residual
    return coords if best_coords is None else best_coords


def _pairs_from_distance(D: np.ndarray, *, count: int, mode: str, skip: int = 0) -> np.ndarray:
    distances = np.asarray(D, dtype=np.float32)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError(f"Expected square distance matrix, got shape {distances.shape}.")

    n = distances.shape[0]
    k = min(max(1, int(count)), max(1, n - 1))
    pairs = np.empty((n * k, 2), dtype=np.int32)
    row = 0
    for i in range(n):
        order = np.argsort(distances[i], kind="stable")
        order = order[order != i]
        if mode == "nearest":
            chosen = order[skip : skip + k]
        elif mode == "mid":
            start = min(max(int(skip), 0), max(0, len(order) - k))
            chosen = order[start : start + k]
        elif mode == "farthest":
            chosen = order[::-1][:k]
        else:
            raise ValueError(f"Unsupported pair mode: {mode}")
        if len(chosen) < k:
            chosen = np.resize(chosen, k)
        for j in chosen[:k]:
            pairs[row] = (i, int(j))
            row += 1
    return pairs


def _custom_pacmap_pairs_from_distance(
    D: np.ndarray,
    *,
    n_neighbors: int,
    mn_ratio: float,
    fp_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = D.shape[0]
    k_neighbors = min(max(1, int(n_neighbors)), n - 1)
    k_mn = min(max(1, int(round(k_neighbors * float(mn_ratio)))), n - 1)
    k_fp = min(max(1, int(round(k_neighbors * float(fp_ratio)))), n - 1)

    pair_neighbors = _pairs_from_distance(D, count=k_neighbors, mode="nearest")
    mid_start = min(k_neighbors, max(0, n - 1 - k_mn))
    pair_mn = _pairs_from_distance(D, count=k_mn, mode="mid", skip=mid_start)
    pair_fp = _pairs_from_distance(D, count=k_fp, mode="farthest")
    return pair_neighbors, pair_mn, pair_fp


def _compute_pacmap_knn_simplex_layouts(
    *,
    X_reference: np.ndarray,
    D_maest: np.ndarray,
    D_tempo: np.ndarray,
    D_chroma: np.ndarray,
    grid: list[tuple[float, float, float]],
    n_neighbors: int,
    mn_ratio: float,
    fp_ratio: float,
    distance: str,
    random_state: int,
    align: bool,
    pair_source: str,
    distance_combine: str,
    layout_init: str,
) -> dict[str, list[list[float]]]:
    try:
        import pacmap
    except ImportError as exc:
        raise ImportError("PaCMAP is not installed. Install with: uv add pacmap") from exc

    if pair_source not in PACMAP_PAIR_SOURCE_CHOICES:
        raise ValueError(f"pair_source must be one of {PACMAP_PAIR_SOURCE_CHOICES}, got {pair_source!r}.")
    if distance_combine not in DISTANCE_COMBINE_CHOICES:
        raise ValueError(
            f"distance_combine must be one of {DISTANCE_COMBINE_CHOICES}, got {distance_combine!r}."
        )
    if layout_init not in LAYOUT_INIT_CHOICES:
        raise ValueError(f"layout_init must be one of {LAYOUT_INIT_CHOICES}, got {layout_init!r}.")

    X = np.asarray(X_reference, dtype=np.float32)
    layouts: dict[str, list[list[float]]] = {}
    aligned_arrays: dict[tuple[int, int, int], np.ndarray] = {}
    n = D_maest.shape[0]
    effective_neighbors = min(max(1, int(n_neighbors)), n - 1)
    scale = int(round(1.0 / min(w for weights in grid for w in weights if w > 0.0)))
    traversal = _neighbor_aware_simplex_order(grid, scale=scale)

    for int_weights, parent_weights in traversal:
        weights = _simplex_float_weights(int_weights, scale=scale)
        D = _combined_distance_3way(
            D_maest,
            D_tempo,
            D_chroma,
            weights,
            combine_mode=distance_combine,
        )
        pair_mn = None
        pair_fp = None
        if pair_source == "combined-all":
            pair_neighbors, pair_mn, pair_fp = _custom_pacmap_pairs_from_distance(
                D,
                n_neighbors=effective_neighbors,
                mn_ratio=mn_ratio,
                fp_ratio=fp_ratio,
            )
        else:
            pair_neighbors = _neighbor_pairs_from_distance(D, n_neighbors=effective_neighbors)
        init: np.ndarray | str = layout_init if layout_init != "neighbor" else "pca"
        if layout_init == "neighbor" and parent_weights is not None and parent_weights in aligned_arrays:
            init = aligned_arrays[parent_weights]
        reducer = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            MN_ratio=float(mn_ratio),
            FP_ratio=float(fp_ratio),
            pair_neighbors=pair_neighbors,
            pair_MN=pair_mn,
            pair_FP=pair_fp,
            distance=str(distance),
            random_state=int(random_state),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Warning: random state is set to.*")
            logging.getLogger("pacmap").setLevel(logging.ERROR)
            coords = np.asarray(reducer.fit_transform(X, init=init), dtype=np.float32)
        if align:
            coords = _align_to_best_neighbor(
                coords,
                int_weights=int_weights,
                aligned_arrays=aligned_arrays,
                scale=scale,
            )
        aligned_arrays[int_weights] = coords
        layouts[_simplex_key(weights)] = [[float(x), float(y)] for x, y in coords]
    return layouts


def _compute_pacmap_knn_4way_layouts(
    *,
    X_reference: np.ndarray,
    D_maest: np.ndarray,
    D_tempo: np.ndarray,
    D_groove: np.ndarray,
    D_chroma: np.ndarray,
    grid: list[tuple[float, float, float, float]],
    n_neighbors: int,
    mn_ratio: float,
    fp_ratio: float,
    distance: str,
    random_state: int,
    align: bool,
    pair_source: str,
    distance_combine: str,
    layout_init: str,
) -> dict[str, list[list[float]]]:
    try:
        import pacmap
    except ImportError as exc:
        raise ImportError("PaCMAP is not installed. Install with: uv add pacmap") from exc

    if pair_source not in PACMAP_PAIR_SOURCE_CHOICES:
        raise ValueError(f"pair_source must be one of {PACMAP_PAIR_SOURCE_CHOICES}, got {pair_source!r}.")
    if distance_combine not in DISTANCE_COMBINE_CHOICES:
        raise ValueError(
            f"distance_combine must be one of {DISTANCE_COMBINE_CHOICES}, got {distance_combine!r}."
        )
    if layout_init not in LAYOUT_INIT_CHOICES:
        raise ValueError(f"layout_init must be one of {LAYOUT_INIT_CHOICES}, got {layout_init!r}.")

    X = np.asarray(X_reference, dtype=np.float32)
    layouts: dict[str, list[list[float]]] = {}
    n = D_maest.shape[0]
    effective_neighbors = min(max(1, int(n_neighbors)), n - 1)

    # Start from the balanced point, then move outward. This gives alignment and
    # neighbor initialization a stable path without needing full tetrahedral graph traversal.
    ordered_grid = sorted(
        grid,
        key=lambda w: (
            float(np.sum((np.asarray(w, dtype=np.float64) - 0.25) ** 2)),
            -w[0],
            -w[1],
            -w[2],
            -w[3],
        ),
    )
    reference_coords: np.ndarray | None = None
    previous_coords: np.ndarray | None = None

    for weights in ordered_grid:
        D = _combined_distance_4way(
            D_maest,
            D_tempo,
            D_groove,
            D_chroma,
            weights,
            combine_mode=distance_combine,
        )
        pair_mn = None
        pair_fp = None
        if pair_source == "combined-all":
            pair_neighbors, pair_mn, pair_fp = _custom_pacmap_pairs_from_distance(
                D,
                n_neighbors=effective_neighbors,
                mn_ratio=mn_ratio,
                fp_ratio=fp_ratio,
            )
        else:
            pair_neighbors = _neighbor_pairs_from_distance(D, n_neighbors=effective_neighbors)

        init: np.ndarray | str = layout_init if layout_init != "neighbor" else "pca"
        if layout_init == "neighbor" and previous_coords is not None:
            init = previous_coords

        reducer = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            MN_ratio=float(mn_ratio),
            FP_ratio=float(fp_ratio),
            pair_neighbors=pair_neighbors,
            pair_MN=pair_mn,
            pair_FP=pair_fp,
            distance=str(distance),
            random_state=int(random_state),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Warning: random state is set to.*")
            logging.getLogger("pacmap").setLevel(logging.ERROR)
            coords = np.asarray(reducer.fit_transform(X, init=init), dtype=np.float32)
        if align and reference_coords is not None:
            coords = _align_to_reference(reference_coords, coords)
        if reference_coords is None:
            reference_coords = coords
        previous_coords = coords
        layouts[_simplex_key_4way(weights)] = [[float(x), float(y)] for x, y in coords]
    return layouts


def _compute_umap_simplex_layouts(
    *,
    D_maest: np.ndarray,
    D_tempo: np.ndarray,
    D_chroma: np.ndarray,
    grid: list[tuple[float, float, float]],
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    align: bool,
    distance_combine: str,
) -> dict[str, list[list[float]]]:
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError("UMAP is not installed. Install with: uv add umap-learn") from exc

    if distance_combine not in DISTANCE_COMBINE_CHOICES:
        raise ValueError(
            f"distance_combine must be one of {DISTANCE_COMBINE_CHOICES}, got {distance_combine!r}."
        )

    layouts: dict[str, list[list[float]]] = {}
    reference: np.ndarray | None = None
    n = D_maest.shape[0]
    effective_neighbors = min(max(2, int(n_neighbors)), max(2, n - 1))

    # Compute high-style layouts first so the aligned UMAP sequence has a stable anchor.
    ordered_grid = sorted(grid, key=lambda w: (-w[0], -w[1], -w[2]))
    for weights in ordered_grid:
        D = _combined_distance_3way(
            D_maest,
            D_tempo,
            D_chroma,
            weights,
            combine_mode=distance_combine,
        )
        reducer = UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=float(min_dist),
            metric="precomputed",
            random_state=int(random_state),
        )
        coords = np.asarray(reducer.fit_transform(D), dtype=np.float32)
        if align and reference is not None:
            coords = _align_to_reference(reference, coords)
        if reference is None:
            reference = coords
        layouts[_simplex_key(weights)] = [[float(x), float(y)] for x, y in coords]
    return layouts


def _compute_umap_4way_layouts(
    *,
    D_maest: np.ndarray,
    D_tempo: np.ndarray,
    D_groove: np.ndarray,
    D_chroma: np.ndarray,
    grid: list[tuple[float, float, float, float]],
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    align: bool,
    distance_combine: str,
) -> dict[str, list[list[float]]]:
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError("UMAP is not installed. Install with: uv add umap-learn") from exc

    if distance_combine not in DISTANCE_COMBINE_CHOICES:
        raise ValueError(
            f"distance_combine must be one of {DISTANCE_COMBINE_CHOICES}, got {distance_combine!r}."
        )

    layouts: dict[str, list[list[float]]] = {}
    reference: np.ndarray | None = None
    n = D_maest.shape[0]
    effective_neighbors = min(max(2, int(n_neighbors)), max(2, n - 1))

    ordered_grid = sorted(
        grid,
        key=lambda w: (
            float(np.sum((np.asarray(w, dtype=np.float64) - 0.25) ** 2)),
            -w[0],
            -w[1],
            -w[2],
            -w[3],
        ),
    )
    for weights in ordered_grid:
        D = _combined_distance_4way(
            D_maest,
            D_tempo,
            D_groove,
            D_chroma,
            weights,
            combine_mode=distance_combine,
        )
        reducer = UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=float(min_dist),
            metric="precomputed",
            random_state=int(random_state),
        )
        coords = np.asarray(reducer.fit_transform(D), dtype=np.float32)
        if align and reference is not None:
            coords = _align_to_reference(reference, coords)
        if reference is None:
            reference = coords
        layouts[_simplex_key_4way(weights)] = [[float(x), float(y)] for x, y in coords]
    return layouts


def _build_similarity_payload_3way(
    *,
    records: list[dict[str, Any]],
    maest_similarity: np.ndarray,
    tempo_similarity: np.ndarray,
    chroma_similarity: np.ndarray,
    maest_distance: np.ndarray,
    tempo_distance: np.ndarray,
    chroma_distance: np.ndarray,
    temperature: float,
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    n = len(records)
    for src_idx in range(n):
        rows: list[dict[str, Any]] = []
        for cand_idx in range(n):
            if cand_idx == src_idx:
                continue
            cand = records[cand_idx]
            src_bpm = float(records[src_idx]["est_bpm"])
            cand_bpm = float(cand["est_bpm"])
            bpm_delta_frac = 0.0
            if src_bpm > 0.0 and cand_bpm > 0.0:
                bpm_delta_frac = float((cand_bpm - src_bpm) / src_bpm)
            rows.append(
                {
                    "idx": int(cand_idx),
                    "track_number": str(cand["track_number"]),
                    "mix_slug": str(cand["mix_slug"]),
                    "title": str(cand["title"]),
                    "artists": str(cand["artists"]),
                    "genre": str(cand["genre"]),
                    "key": str(cand["key"]),
                    "csv_bpm": str(cand["csv_bpm"]),
                    "est_bpm": cand_bpm,
                    "est_conf": float(cand["est_conf"]),
                    "maest_similarity": float(maest_similarity[src_idx, cand_idx]),
                    "tempo_similarity": float(tempo_similarity[src_idx, cand_idx]),
                    "chroma_similarity": float(chroma_similarity[src_idx, cand_idx]),
                    "maest_score_norm": float(1.0 - maest_distance[src_idx, cand_idx]),
                    "tempo_score_norm": float(1.0 - tempo_distance[src_idx, cand_idx]),
                    "chroma_score_norm": float(1.0 - chroma_distance[src_idx, cand_idx]),
                    "bpm_delta_frac": bpm_delta_frac,
                }
            )
        payload[str(src_idx)] = {"candidates": rows, "temperature": float(temperature)}
    return payload


def _build_similarity_payload_4way(
    *,
    records: list[dict[str, Any]],
    maest_similarity: np.ndarray,
    tempo_similarity: np.ndarray,
    groove_similarity: np.ndarray,
    chroma_similarity: np.ndarray,
    maest_distance: np.ndarray,
    tempo_distance: np.ndarray,
    groove_distance: np.ndarray,
    chroma_distance: np.ndarray,
    temperature: float,
) -> dict[str, dict[str, Any]]:
    payload = _build_similarity_payload_3way(
        records=records,
        maest_similarity=maest_similarity,
        tempo_similarity=tempo_similarity,
        chroma_similarity=chroma_similarity,
        maest_distance=maest_distance,
        tempo_distance=tempo_distance,
        chroma_distance=chroma_distance,
        temperature=temperature,
    )
    for src_key, group in payload.items():
        src_idx = int(src_key)
        for row in group["candidates"]:
            cand_idx = int(row["idx"])
            row["groove_similarity"] = float(groove_similarity[src_idx, cand_idx])
            row["groove_score_norm"] = float(1.0 - groove_distance[src_idx, cand_idx])
    return payload


def _build_simplex_html(
    *,
    plot_html: str,
    records: list[dict[str, Any]],
    layouts: dict[str, list[list[float]]],
    similarity_payload: dict[str, dict[str, Any]],
    plot_div_id: str,
    title: str,
    step: float,
    top_k_rows: int,
    temperature: float,
    background_links_per_song: int,
    click_links_per_song: int,
    bpm_color_scale_pct: float,
    generation_settings: dict[str, Any],
    layout_mode: str,
) -> str:
    record_payload = [
        {
            "idx": int(r["idx"]),
            "title": str(r["title"]),
            "artists": str(r["artists"]),
            "genre": str(r["genre"]),
            "raw_genre": str(r.get("raw_genre", r["genre"])),
            "key": str(r["key"]),
            "csv_bpm": str(r["csv_bpm"]),
            "est_bpm": float(r["est_bpm"]),
            "est_conf": float(r["est_conf"]),
            "track_number": str(r["track_number"]),
            "filename": str(r["filename"]),
            "snippet_uri": str(r["snippet_uri"]),
            "snippet_start": float(r["snippet_start"]),
            "snippet_end": float(r["snippet_end"]),
            "snippet_rms": float(r["snippet_rms"]),
            "mix_slug": str(r["mix_slug"]),
        }
        for r in records
    ]
    data_scripts = "\n".join(
        [
            f'<script id="simplex-records-json" type="application/json">{_json_script_payload(record_payload)}</script>',
            f'<script id="simplex-layouts-json" type="application/json">{_json_script_payload(layouts)}</script>',
            f'<script id="simplex-sim-json" type="application/json">{_json_script_payload(similarity_payload)}</script>',
            f'<script id="simplex-config-json" type="application/json">{_json_script_payload({"step": step, "top_k_rows": top_k_rows, "temperature": temperature, "background_links_per_song": background_links_per_song, "click_links_per_song": click_links_per_song, "bpm_color_scale_pct": bpm_color_scale_pct, "layout_mode": layout_mode})}</script>',
        ]
    )
    style = """
<style>
  :root { color-scheme: light; --line:#d9dee8; --ink:#17202a; --muted:#5b6678; }
  body { font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 18px; color: var(--ink); background: #f6f8fb; }
  h1 { font-size: 22px; margin: 0 0 14px; }
  .workspace { display:grid; grid-template-columns:minmax(0, 1fr) 390px; gap:12px; align-items:start; }
  .plot-pane { min-width:0; }
  .side-panel { display:grid; gap:10px; position:sticky; top:12px; }
  .detail-panel { display:grid; gap:10px; margin-top:12px; }
  .box, .weights { border:1px solid var(--line); background:#fff; padding:10px; }
  .control-grid { display:grid; gap:10px; align-items:center; }
  .simplex-control { width:100%; display:block; touch-action:none; user-select:none; }
  .simplex-area { fill:#fbfcff; stroke:#9aa8bc; stroke-width:1.5; }
  .simplex-grid-line { stroke:#d8deea; stroke-width:0.8; }
  .simplex-label { fill:#2f3a4c; font-size:13px; font-weight:600; text-anchor:middle; }
  .simplex-handle { fill:#111827; stroke:#fff; stroke-width:4; cursor:grab; }
  .simplex-handle:active { cursor:grabbing; }
  .weights { display:grid; grid-template-columns: 72px 1fr 56px; gap:8px; align-items:center; }
  .settings-panel h2 { font-size:14px; margin:0 0 8px; }
  .settings-panel dl { display:grid; grid-template-columns: minmax(120px, auto) minmax(0, 1fr); gap:5px 10px; margin:0; font-size:12px; }
  .settings-panel dt { color:var(--muted); }
  .settings-panel dd { margin:0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap:anywhere; }
  .muted { color:var(--muted); font-size:12px; }
  audio { width:100%; display:block; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th, td { border-bottom:1px solid #edf0f5; padding:5px 6px; vertical-align:top; }
  th { position:sticky; top:0; background:#f9fafc; z-index:1; text-align:left; color:#3c4758; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .table-wrap { max-height:380px; overflow:auto; border:1px solid #edf0f5; }
  @media (max-width: 980px) {
    .workspace { grid-template-columns:1fr; }
    .side-panel { position:static; }
  }
</style>
"""
    settings_panel = _build_settings_panel(generation_settings)
    controls = f"""
<aside class="side-panel">
  <div class="box control-grid">
    <svg id="simplex-control" class="simplex-control" viewBox="0 0 360 320" role="img" aria-label="Genre tempo key blend triangle">
      <polygon id="simplex-area" class="simplex-area" points="180,34 44,270 316,270"></polygon>
      <line class="simplex-grid-line" x1="166.4" y1="57.6" x2="71.2" y2="270"></line>
      <line class="simplex-grid-line" x1="193.6" y1="57.6" x2="288.8" y2="270"></line>
      <line class="simplex-grid-line" x1="153.0" y1="80.8" x2="98.4" y2="270"></line>
      <line class="simplex-grid-line" x1="207.0" y1="80.8" x2="261.6" y2="270"></line>
      <line class="simplex-grid-line" x1="139.2" y1="104.4" x2="125.6" y2="270"></line>
      <line class="simplex-grid-line" x1="220.8" y1="104.4" x2="234.4" y2="270"></line>
      <line class="simplex-grid-line" x1="125.6" y1="128.0" x2="152.8" y2="270"></line>
      <line class="simplex-grid-line" x1="234.4" y1="128.0" x2="207.2" y2="270"></line>
      <line class="simplex-grid-line" x1="112.0" y1="151.6" x2="180.0" y2="270"></line>
      <line class="simplex-grid-line" x1="248.0" y1="151.6" x2="180.0" y2="270"></line>
      <line class="simplex-grid-line" x1="112.0" y1="151.6" x2="248.0" y2="151.6"></line>
      <line class="simplex-grid-line" x1="98.4" y1="175.2" x2="261.6" y2="175.2"></line>
      <line class="simplex-grid-line" x1="84.8" y1="198.8" x2="275.2" y2="198.8"></line>
      <line class="simplex-grid-line" x1="71.2" y1="222.4" x2="288.8" y2="222.4"></line>
      <line class="simplex-grid-line" x1="57.6" y1="246.0" x2="302.4" y2="246.0"></line>
      <text class="simplex-label" x="180" y="22">Genre</text>
      <text class="simplex-label" x="44" y="294">Tempo</text>
      <text class="simplex-label" x="316" y="294">Key</text>
      <circle id="simplex-handle" class="simplex-handle" cx="180" cy="128.4" r="10"></circle>
    </svg>
    <div class="weights">
      <div>Genre</div><input id="weight-maest" type="range" min="0" max="1" step="0.01" value="0.6"><output id="weight-maest-val">0.600</output>
      <div>Tempo</div><input id="weight-tempo" type="range" min="0" max="1" step="0.01" value="0.2"><output id="weight-tempo-val">0.200</output>
      <div>Key</div><input id="weight-chroma" type="range" min="0" max="1" step="0.01" value="0.2"><output id="weight-chroma-val">0.200</output>
      <div></div><div id="weight-summary" class="muted"></div><div></div>
    </div>
  </div>
{settings_panel}
</aside>
"""
    detail_panel = """
<section class="detail-panel">
  <div id="track-meta" class="box">Click a point to play its snippet and show Genre/tempo/key recommendations.</div>
  <audio id="track-audio" controls></audio>
  <div id="similarity-panel" class="box">Recommendations will appear here after you click a point.</div>
</section>
"""
    script = f"""
<script>
(function() {{
  const records = JSON.parse(document.getElementById('simplex-records-json').textContent);
  const layouts = JSON.parse(document.getElementById('simplex-layouts-json').textContent);
  const simMap = JSON.parse(document.getElementById('simplex-sim-json').textContent);
  const config = JSON.parse(document.getElementById('simplex-config-json').textContent);
  const plot = document.getElementById('{plot_div_id}');
  const els = {{
    ma: document.getElementById('weight-maest'),
    te: document.getElementById('weight-tempo'),
    ch: document.getElementById('weight-chroma'),
    maVal: document.getElementById('weight-maest-val'),
    teVal: document.getElementById('weight-tempo-val'),
    chVal: document.getElementById('weight-chroma-val'),
    simplex: document.getElementById('simplex-control'),
    simplexHandle: document.getElementById('simplex-handle'),
    summary: document.getElementById('weight-summary'),
    meta: document.getElementById('track-meta'),
    audio: document.getElementById('track-audio'),
    panel: document.getElementById('similarity-panel'),
  }};
  let currentPoints = [];
  let selectedIdx = null;
  let backgroundTraceIndices = [];
  let selectedTraceIndices = [];

  function esc(v) {{ return String(v == null ? '' : v).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch])); }}
  function fmt(v,d) {{ return Number(v || 0).toFixed(d); }}
  function pct(v) {{ return (Number(v || 0) * 100).toFixed(1) + '%'; }}
  function signedPct(v) {{ const n = Number(v || 0) * 100; return (n >= 0 ? '+' : '') + n.toFixed(1) + '%'; }}
  function clamp(v, lo, hi) {{ return Math.min(hi, Math.max(lo, v)); }}
  const simplexVertices = {{
    maest: {{x: 180, y: 34}},
    tempo: {{x: 44, y: 270}},
    chroma: {{x: 316, y: 270}},
  }};
  function weights() {{
    const ma = Math.max(0, Number(els.ma.value || 0));
    const te = Math.max(0, Number(els.te.value || 0));
    const ch = Math.max(0, Number(els.ch.value || 0));
    const s = ma + te + ch;
    if (s <= 1e-12) return {{maest: 1/3, tempo: 1/3, chroma: 1/3}};
    return {{maest: ma/s, tempo: te/s, chroma: ch/s}};
  }}
  function setSliderWeights(w) {{
    els.ma.value = String(clamp(w.maest, 0, 1));
    els.te.value = String(clamp(w.tempo, 0, 1));
    els.ch.value = String(clamp(w.chroma, 0, 1));
  }}
  function normalizeSliderWeights() {{ setSliderWeights(weights()); }}
  function simplexPoint(w) {{
    return {{
      x: w.maest * simplexVertices.maest.x + w.tempo * simplexVertices.tempo.x + w.chroma * simplexVertices.chroma.x,
      y: w.maest * simplexVertices.maest.y + w.tempo * simplexVertices.tempo.y + w.chroma * simplexVertices.chroma.y,
    }};
  }}
  function updateSimplexHandle(w) {{
    const p = simplexPoint(w);
    els.simplexHandle.setAttribute('cx', String(p.x));
    els.simplexHandle.setAttribute('cy', String(p.y));
  }}
  function simplexWeightsFromPoint(px, py) {{
    const a = simplexVertices.maest;
    const b = simplexVertices.tempo;
    const c = simplexVertices.chroma;
    const denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    let ma = ((b.y - c.y) * (px - c.x) + (c.x - b.x) * (py - c.y)) / denom;
    let te = ((c.y - a.y) * (px - c.x) + (a.x - c.x) * (py - c.y)) / denom;
    let ch = 1 - ma - te;
    ma = clamp(ma, 0, 1);
    te = clamp(te, 0, 1);
    ch = clamp(ch, 0, 1);
    const s = Math.max(1e-12, ma + te + ch);
    return {{maest: ma / s, tempo: te / s, chroma: ch / s}};
  }}
  function eventToSvgPoint(ev) {{
    const rect = els.simplex.getBoundingClientRect();
    return {{
      x: (ev.clientX - rect.left) * 360 / Math.max(1, rect.width),
      y: (ev.clientY - rect.top) * 320 / Math.max(1, rect.height),
    }};
  }}
  function setWeightsFromSimplexEvent(ev) {{
    const p = eventToSvgPoint(ev);
    const w = simplexWeightsFromPoint(p.x, p.y);
    setSliderWeights(w);
    renderAll();
  }}
  function gridScale() {{ return Math.round(1 / Number(config.step || 0.1)); }}
  function key(i,j,k) {{
    const scale = gridScale();
    return (i/scale).toFixed(1)+','+(j/scale).toFixed(1)+','+(k/scale).toFixed(1);
  }}
  function layoutAtGrid(i,j,k) {{
    const scale = gridScale();
    if (i < 0 || j < 0 || k < 0 || i + j + k !== scale) return null;
    return layouts[key(i,j,k)];
  }}
  function nearestGridCoords(w) {{
    const scale = gridScale();
    let i = Math.round(clamp(w.maest, 0, 1) * scale);
    let j = Math.round(clamp(w.tempo, 0, 1) * scale);
    if (i + j > scale) {{
      const excess = i + j - scale;
      if (j >= excess) j -= excess;
      else i = Math.max(0, i - (excess - j));
    }}
    const k = scale - i - j;
    return {{i, j, k, scale}};
  }}
  function nearestGridWeights(w) {{
    const g = nearestGridCoords(w);
    return {{maest: g.i / g.scale, tempo: g.j / g.scale, chroma: g.k / g.scale}};
  }}
  function nearestGridLayout(w) {{
    const g = nearestGridCoords(w);
    const i = g.i, j = g.j, k = g.k;
    return layoutAtGrid(i, j, k) || layouts['1.0,0.0,0.0'] || [];
  }}
  function simplexLayoutPoints(w) {{
    if (config.layout_mode === 'discrete') return nearestGridLayout(w);
    return simplexInterpolatedPoints(w);
  }}
  function simplexInterpolatedPoints(w) {{
    const scale = gridScale();
    const eps = 1e-7;
    let x = clamp(w.maest, 0, 1) * scale;
    let y = clamp(w.tempo, 0, 1) * scale;
    if (x + y > scale) {{
      const factor = scale / Math.max(eps, x + y);
      x *= factor;
      y *= factor;
    }}
    x = Math.abs(x - Math.round(x)) < eps ? Math.round(x) : x;
    y = Math.abs(y - Math.round(y)) < eps ? Math.round(y) : y;
    let i = Math.floor(x);
    let j = Math.floor(y);
    if (i + j >= scale) {{
      i = Math.min(scale, i);
      j = Math.min(scale - i, j);
      const A = layoutAtGrid(i, j, scale - i - j) || nearestGridLayout(w);
      return A || [];
    }}
    const rx = x - i;
    const ry = y - j;
    const kBase = scale - i - j;
    let verts, coeffs;
    if (rx + ry <= 1 + eps || kBase <= 1) {{
      verts = [[i,j,scale-i-j], [i+1,j,scale-i-j-1], [i,j+1,scale-i-j-1]];
      coeffs = [Math.max(0, 1-rx-ry), rx, ry];
    }} else {{
      verts = [[i+1,j+1,scale-i-j-2], [i+1,j,scale-i-j-1], [i,j+1,scale-i-j-1]];
      coeffs = [rx+ry-1, 1-ry, 1-rx];
    }}
    const coeffSum = Math.max(eps, coeffs[0] + coeffs[1] + coeffs[2]);
    coeffs = coeffs.map(v => v / coeffSum);
    const L = verts.map(v => layoutAtGrid(v[0], v[1], v[2]));
    if (L.some(v => !v)) return nearestGridLayout(w);
    return L[0].map((_, idx) => [
      coeffs[0]*Number(L[0][idx][0]) + coeffs[1]*Number(L[1][idx][0]) + coeffs[2]*Number(L[2][idx][0]),
      coeffs[0]*Number(L[0][idx][1]) + coeffs[1]*Number(L[1][idx][1]) + coeffs[2]*Number(L[2][idx][1]),
    ]);
  }}
  function baseTraceIndices() {{
    const names = new Set(records.map(r => String(r.genre)));
    const out = [];
    (plot.data || []).forEach((trace, i) => {{ if (names.has(String(trace.name))) out.push(i); }});
    return out;
  }}
  function updatePointCoordinates() {{
    const byGenre = new Map();
    for (const r of records) {{
      const pt = currentPoints[Number(r.idx)];
      if (!pt) continue;
      const g = String(r.genre);
      if (!byGenre.has(g)) byGenre.set(g, {{x: [], y: []}});
      byGenre.get(g).x.push(Number(pt[0]));
      byGenre.get(g).y.push(Number(pt[1]));
    }}
    for (const traceIdx of baseTraceIndices()) {{
      const trace = plot.data[traceIdx] || {{}};
      const vals = byGenre.get(String(trace.name));
      if (vals) Plotly.restyle(plot, {{x: [vals.x], y: [vals.y]}}, [traceIdx]);
    }}
  }}
  function rankedRows(sourceIdx, w) {{
    const candidates = ((simMap[String(sourceIdx)] || {{}}).candidates || []);
    const rows = candidates.map(c => {{
      const score = w.maest*Number(c.maest_score_norm||0) + w.tempo*Number(c.tempo_score_norm||0) + w.chroma*Number(c.chroma_score_norm||0);
      return {{...c, score}};
    }}).sort((a,b) => b.score - a.score);
    const temp = Math.max(1e-6, Number(config.temperature || 0.08));
    const maxScore = rows.length ? Number(rows[0].score || 0) : 0;
    let sum = 0;
    rows.forEach(r => {{ r._exp = Math.exp((Number(r.score || 0) - maxScore) / temp); sum += r._exp; }});
    rows.forEach((r, i) => {{ r.rank = i + 1; r.probability = sum > 0 ? r._exp / sum : 0; }});
    return rows;
  }}
  function clearTraceSet(indices) {{
    if (!indices.length || !window.Plotly) return [];
    try {{ Plotly.deleteTraces(plot, indices.slice().sort((a,b) => b-a)); }} catch (err) {{}}
    return [];
  }}
  function addTraceSet(traces) {{
    if (!traces.length || !window.Plotly) return [];
    const start = (plot.data || []).length;
    try {{ Plotly.addTraces(plot, traces); }} catch (err) {{ return []; }}
    return Array.from({{length: traces.length}}, (_, i) => start + i);
  }}
  function colorForDelta(delta, alpha) {{
    const scale = Math.max(1e-6, Number(config.bpm_color_scale_pct || 0.10));
    const t = clamp(Number(delta || 0) / scale, -1, 1);
    const f = Math.abs(t);
    const base = [155, 155, 155];
    const hot = [218, 65, 45];
    const cold = [45, 98, 210];
    const target = t >= 0 ? hot : cold;
    const rgb = base.map((v, i) => Math.round(v + (target[i] - v) * f));
    return 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + alpha + ')';
  }}
  function linkTraces(sourceIdx, rows, limit, highlighted) {{
    const src = currentPoints[Number(sourceIdx)];
    if (!src) return [];
    const traces = [];
    for (const row of rows.slice(0, limit)) {{
      const dst = currentPoints[Number(row.idx)];
      if (!dst) continue;
      const prob = Number(row.probability || 0);
      traces.push({{
        type: 'scatter',
        mode: 'lines',
        x: [Number(src[0]), Number(dst[0])],
        y: [Number(src[1]), Number(dst[1])],
        line: {{
          color: colorForDelta(row.bpm_delta_frac, highlighted ? clamp(0.40 + prob * 2.2, 0.40, 0.90) : 0.13),
          width: highlighted ? clamp(1.8 + prob * 10.0, 1.8, 5.0) : 0.85,
          dash: highlighted ? 'solid' : 'dot',
        }},
        hoverinfo: 'skip',
        showlegend: false,
      }});
    }}
    return traces;
  }}
  function renderBackgroundLinks(w) {{
    backgroundTraceIndices = clearTraceSet(backgroundTraceIndices);
    const limit = Math.max(0, Number(config.background_links_per_song || 0));
    if (limit <= 0) return;
    const traces = [];
    for (const sourceIdx of Object.keys(simMap)) {{
      traces.push(...linkTraces(Number(sourceIdx), rankedRows(Number(sourceIdx), w), limit, false));
    }}
    backgroundTraceIndices = addTraceSet(traces);
  }}
  function renderSelectedLinks(w) {{
    selectedTraceIndices = clearTraceSet(selectedTraceIndices);
    if (selectedIdx === null) return;
    const limit = Math.max(0, Number(config.click_links_per_song || 0));
    selectedTraceIndices = addTraceSet(linkTraces(selectedIdx, rankedRows(selectedIdx, w), limit, true));
  }}
  function renderPanel(w) {{
    if (selectedIdx === null) return;
    const rows = rankedRows(selectedIdx, w).slice(0, Number(config.top_k_rows || 25));
    const body = rows.map(row => '<tr>' +
      '<td class="num">' + row.rank + '</td><td class="num">' + esc(row.track_number) + '</td><td>' + esc(row.mix_slug) + '</td>' +
      '<td>' + esc(row.title) + '</td><td>' + esc(row.artists) + '</td><td>' + esc(row.genre) + '</td><td>' + esc(row.key) + '</td>' +
      '<td class="num">' + signedPct(row.bpm_delta_frac) + '</td><td class="num">' + fmt(row.score,4) + '</td>' +
      '<td class="num">' + fmt(row.maest_score_norm,4) + '</td><td class="num">' + fmt(row.tempo_score_norm,4) + '</td><td class="num">' + fmt(row.chroma_score_norm,4) + '</td>' +
      '<td class="num">' + fmt(row.maest_similarity,4) + '</td><td class="num">' + fmt(row.tempo_similarity,4) + '</td><td class="num">' + fmt(row.chroma_similarity,4) + '</td>' +
      '<td class="num">' + pct(row.probability) + '</td></tr>').join('');
    els.panel.innerHTML = '<b>Top ' + Number(config.top_k_rows || 25) + ' Genre/tempo/key matches</b>' +
      '<div class="muted">Ranking uses normalized component scores derived from the same normalized distances as the layout. Raw similarities are shown for diagnostics.</div>' +
      '<div class="table-wrap"><table><thead><tr><th class="num">#</th><th class="num">Track</th><th>Mix</th><th>Title</th><th>Artists</th><th>Genre</th><th>Key</th>' +
      '<th class="num">dBPM%</th><th class="num">Score</th><th class="num">Genre norm</th><th class="num">Tempo norm</th><th class="num">Key norm</th>' +
      '<th class="num">Genre raw</th><th class="num">Tempo raw</th><th class="num">Key raw</th><th class="num">Prob</th></tr></thead><tbody>' + body + '</tbody></table></div>';
  }}
  function renderAll() {{
    let w = weights();
    if (config.layout_mode === 'discrete') {{
      w = nearestGridWeights(w);
      setSliderWeights(w);
    }}
    selectedTraceIndices = clearTraceSet(selectedTraceIndices);
    backgroundTraceIndices = clearTraceSet(backgroundTraceIndices);
    els.maVal.textContent = fmt(w.maest, 3);
    els.teVal.textContent = fmt(w.tempo, 3);
    els.chVal.textContent = fmt(w.chroma, 3);
    updateSimplexHandle(w);
    els.summary.innerHTML = 'Normalized weights: Genre <b>' + pct(w.maest) + '</b>, tempo <b>' + pct(w.tempo) + '</b>, key <b>' + pct(w.chroma) + '</b>.' +
      (config.layout_mode === 'discrete' ? ' Showing nearest precomputed layout.' : '');
    currentPoints = simplexLayoutPoints(w);
    updatePointCoordinates();
    renderBackgroundLinks(w);
    renderSelectedLinks(w);
    renderPanel(w);
  }}
  [els.ma, els.te, els.ch].forEach(el => el.addEventListener('input', () => {{
    normalizeSliderWeights();
    renderAll();
  }}));
  if (els.simplex) {{
    let draggingSimplex = false;
    els.simplex.addEventListener('pointerdown', ev => {{
      draggingSimplex = true;
      els.simplex.setPointerCapture(ev.pointerId);
      setWeightsFromSimplexEvent(ev);
    }});
    els.simplex.addEventListener('pointermove', ev => {{
      if (!draggingSimplex) return;
      setWeightsFromSimplexEvent(ev);
    }});
    els.simplex.addEventListener('pointerup', ev => {{
      draggingSimplex = false;
      try {{ els.simplex.releasePointerCapture(ev.pointerId); }} catch (err) {{}}
    }});
    els.simplex.addEventListener('pointercancel', () => {{ draggingSimplex = false; }});
  }}
  if (plot && plot.on) {{
    plot.on('plotly_click', ev => {{
      if (!ev || !ev.points || !ev.points.length) return;
      const c = ev.points[0].customdata || [];
      const idx = Number(c[13]);
      if (!Number.isFinite(idx)) return;
      selectedIdx = idx;
      els.meta.innerHTML = '<b>' + esc(c[0]) + '</b><br>Artists: ' + esc(c[1]) + '<br>Genre: ' + esc(c[2]) +
        '<br>Mix: ' + esc(c[14]) + '<br>Key: ' + esc(c[3]) + '<br>CSV BPM: ' + esc(c[4]) +
        '<br>Estimated BPM: ' + fmt(c[5], 2) + ' (confidence=' + fmt(c[6], 3) + ')<br>Track #: ' + esc(c[7]) + '<br>File: ' + esc(c[8]);
      if (c[9]) {{
        els.audio.src = c[9];
        const p = els.audio.play();
        if (p && p.catch) p.catch(() => {{}});
      }} else {{
        els.audio.removeAttribute('src');
        els.audio.load();
      }}
      renderSelectedLinks(weights());
      renderPanel(weights());
    }});
  }}
  renderAll();
}})();
</script>
"""
    return "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{title}</title>",
            style,
            "</head>",
            "<body>",
            f"<h1>{title}</h1>",
            data_scripts,
            '<main class="workspace">',
            '<section class="plot-pane">',
            plot_html,
            '</section>',
            controls,
            '</main>',
            detail_panel,
            script,
            "</body>",
            "</html>",
        ]
    )


def _build_genre_mixability_html(
    *,
    plot_html: str,
    records: list[dict[str, Any]],
    layouts: dict[str, list[list[float]]],
    similarity_payload: dict[str, dict[str, Any]],
    plot_div_id: str,
    title: str,
    step: float,
    top_k_rows: int,
    temperature: float,
    background_links_per_song: int,
    click_links_per_song: int,
    bpm_color_scale_pct: float,
    generation_settings: dict[str, Any],
) -> str:
    record_payload = [
        {
            "idx": int(r["idx"]),
            "title": str(r["title"]),
            "artists": str(r["artists"]),
            "genre": str(r["genre"]),
            "raw_genre": str(r.get("raw_genre", r["genre"])),
            "key": str(r["key"]),
            "csv_bpm": str(r["csv_bpm"]),
            "est_bpm": float(r["est_bpm"]),
            "est_conf": float(r["est_conf"]),
            "track_number": str(r["track_number"]),
            "filename": str(r["filename"]),
            "snippet_uri": str(r["snippet_uri"]),
            "snippet_start": float(r["snippet_start"]),
            "snippet_end": float(r["snippet_end"]),
            "snippet_rms": float(r["snippet_rms"]),
            "mix_slug": str(r["mix_slug"]),
        }
        for r in records
    ]
    layout_entries = []
    for key, points in layouts.items():
        weights = [float(part) for part in key.split(",")]
        layout_entries.append({"key": key, "weights": weights, "points": points})
    data_scripts = "\n".join(
        [
            f'<script id="simplex-records-json" type="application/json">{_json_script_payload(record_payload)}</script>',
            f'<script id="simplex-layouts-json" type="application/json">{_json_script_payload(layouts)}</script>',
            f'<script id="simplex-layout-entries-json" type="application/json">{_json_script_payload(layout_entries)}</script>',
            f'<script id="simplex-sim-json" type="application/json">{_json_script_payload(similarity_payload)}</script>',
            f'<script id="simplex-config-json" type="application/json">{_json_script_payload({"step": step, "top_k_rows": top_k_rows, "temperature": temperature, "background_links_per_song": background_links_per_song, "click_links_per_song": click_links_per_song, "bpm_color_scale_pct": bpm_color_scale_pct, "control_mode": "genre-mixability"})}</script>',
        ]
    )
    style = """
<style>
  :root { color-scheme: light; --line:#d9dee8; --ink:#17202a; --muted:#5b6678; }
  body { font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 18px; color: var(--ink); background: #f6f8fb; }
  h1 { font-size: 22px; margin: 0 0 14px; }
  .workspace { display:grid; grid-template-columns:minmax(0, 1fr) 390px; gap:12px; align-items:start; }
  .plot-pane { min-width:0; }
  .side-panel { display:grid; gap:10px; position:sticky; top:12px; }
  .detail-panel { display:grid; gap:10px; margin-top:12px; }
  .box, .weights { border:1px solid var(--line); background:#fff; padding:10px; }
  .control-grid { display:grid; gap:10px; align-items:center; }
  .simplex-control { width:100%; display:block; touch-action:none; user-select:none; }
  .simplex-area { fill:#fbfcff; stroke:#9aa8bc; stroke-width:1.5; }
  .simplex-grid-line { stroke:#d8deea; stroke-width:0.8; }
  .simplex-label { fill:#2f3a4c; font-size:13px; font-weight:600; text-anchor:middle; }
  .simplex-handle { fill:#111827; stroke:#fff; stroke-width:4; cursor:grab; }
  .simplex-handle:active { cursor:grabbing; }
  .weights { display:grid; grid-template-columns: 112px 1fr 56px; gap:8px; align-items:center; }
  .settings-panel h2 { font-size:14px; margin:0 0 8px; }
  .settings-panel dl { display:grid; grid-template-columns: minmax(120px, auto) minmax(0, 1fr); gap:5px 10px; margin:0; font-size:12px; }
  .settings-panel dt { color:var(--muted); }
  .settings-panel dd { margin:0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap:anywhere; }
  .muted { color:var(--muted); font-size:12px; }
  audio { width:100%; display:block; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th, td { border-bottom:1px solid #edf0f5; padding:5px 6px; vertical-align:top; }
  th { position:sticky; top:0; background:#f9fafc; z-index:1; text-align:left; color:#3c4758; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .table-wrap { max-height:380px; overflow:auto; border:1px solid #edf0f5; }
  @media (max-width: 980px) {
    .workspace { grid-template-columns:1fr; }
    .side-panel { position:static; }
  }
</style>
"""
    settings_panel = _build_settings_panel(generation_settings)
    controls = f"""
<aside class="side-panel">
  <div class="box control-grid">
    <div class="weights">
      <div>Genre/style</div><input id="weight-style" type="range" min="0" max="1" step="0.01" value="0.45"><output id="weight-style-val">0.450</output>
      <div></div><div class="muted">Left side of the model is MAEST/style; the triangle splits the remaining mixability weight.</div><div></div>
    </div>
    <svg id="simplex-control" class="simplex-control" viewBox="0 0 360 320" role="img" aria-label="Tempo groove key mixability triangle">
      <polygon id="simplex-area" class="simplex-area" points="180,34 44,270 316,270"></polygon>
      <line class="simplex-grid-line" x1="166.4" y1="57.6" x2="71.2" y2="270"></line>
      <line class="simplex-grid-line" x1="193.6" y1="57.6" x2="288.8" y2="270"></line>
      <line class="simplex-grid-line" x1="153.0" y1="80.8" x2="98.4" y2="270"></line>
      <line class="simplex-grid-line" x1="207.0" y1="80.8" x2="261.6" y2="270"></line>
      <line class="simplex-grid-line" x1="139.2" y1="104.4" x2="125.6" y2="270"></line>
      <line class="simplex-grid-line" x1="220.8" y1="104.4" x2="234.4" y2="270"></line>
      <line class="simplex-grid-line" x1="125.6" y1="128.0" x2="152.8" y2="270"></line>
      <line class="simplex-grid-line" x1="234.4" y1="128.0" x2="207.2" y2="270"></line>
      <line class="simplex-grid-line" x1="112.0" y1="151.6" x2="180.0" y2="270"></line>
      <line class="simplex-grid-line" x1="248.0" y1="151.6" x2="180.0" y2="270"></line>
      <line class="simplex-grid-line" x1="112.0" y1="151.6" x2="248.0" y2="151.6"></line>
      <line class="simplex-grid-line" x1="98.4" y1="175.2" x2="261.6" y2="175.2"></line>
      <line class="simplex-grid-line" x1="84.8" y1="198.8" x2="275.2" y2="198.8"></line>
      <line class="simplex-grid-line" x1="71.2" y1="222.4" x2="288.8" y2="222.4"></line>
      <line class="simplex-grid-line" x1="57.6" y1="246.0" x2="302.4" y2="246.0"></line>
      <text class="simplex-label" x="180" y="22">Tempo</text>
      <text class="simplex-label" x="44" y="294">Groove</text>
      <text class="simplex-label" x="316" y="294">Key</text>
      <circle id="simplex-handle" class="simplex-handle" cx="180" cy="128.4" r="10"></circle>
    </svg>
    <div class="weights">
      <div>Tempo</div><input id="weight-tempo" type="range" min="0" max="1" step="0.01" value="0.34"><output id="weight-tempo-val">0.340</output>
      <div>Groove</div><input id="weight-groove" type="range" min="0" max="1" step="0.01" value="0.33"><output id="weight-groove-val">0.330</output>
      <div>Key</div><input id="weight-chroma" type="range" min="0" max="1" step="0.01" value="0.33"><output id="weight-chroma-val">0.330</output>
      <div></div><div id="weight-summary" class="muted"></div><div></div>
    </div>
  </div>
{settings_panel}
</aside>
"""
    detail_panel = """
<section class="detail-panel">
  <div id="track-meta" class="box">Click a point to play its snippet and show genre/mixability recommendations.</div>
  <audio id="track-audio" controls></audio>
  <div id="similarity-panel" class="box">Recommendations will appear here after you click a point.</div>
</section>
"""
    script = f"""
<script>
(function() {{
  const records = JSON.parse(document.getElementById('simplex-records-json').textContent);
  const layouts = JSON.parse(document.getElementById('simplex-layouts-json').textContent);
  const layoutEntries = JSON.parse(document.getElementById('simplex-layout-entries-json').textContent);
  const simMap = JSON.parse(document.getElementById('simplex-sim-json').textContent);
  const config = JSON.parse(document.getElementById('simplex-config-json').textContent);
  const plot = document.getElementById('{plot_div_id}');
  const els = {{
    style: document.getElementById('weight-style'),
    te: document.getElementById('weight-tempo'),
    gr: document.getElementById('weight-groove'),
    ch: document.getElementById('weight-chroma'),
    styleVal: document.getElementById('weight-style-val'),
    teVal: document.getElementById('weight-tempo-val'),
    grVal: document.getElementById('weight-groove-val'),
    chVal: document.getElementById('weight-chroma-val'),
    simplex: document.getElementById('simplex-control'),
    simplexHandle: document.getElementById('simplex-handle'),
    summary: document.getElementById('weight-summary'),
    meta: document.getElementById('track-meta'),
    audio: document.getElementById('track-audio'),
    panel: document.getElementById('similarity-panel'),
  }};
  let currentPoints = [];
  let selectedIdx = null;
  let backgroundTraceIndices = [];
  let selectedTraceIndices = [];

  function esc(v) {{ return String(v == null ? '' : v).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch])); }}
  function fmt(v,d) {{ return Number(v || 0).toFixed(d); }}
  function pct(v) {{ return (Number(v || 0) * 100).toFixed(1) + '%'; }}
  function signedPct(v) {{ const n = Number(v || 0) * 100; return (n >= 0 ? '+' : '') + n.toFixed(1) + '%'; }}
  function clamp(v, lo, hi) {{ return Math.min(hi, Math.max(lo, v)); }}
  const simplexVertices = {{
    tempo: {{x: 180, y: 34}},
    groove: {{x: 44, y: 270}},
    chroma: {{x: 316, y: 270}},
  }};
  function mixWeightsRaw() {{
    const te = Math.max(0, Number(els.te.value || 0));
    const gr = Math.max(0, Number(els.gr.value || 0));
    const ch = Math.max(0, Number(els.ch.value || 0));
    const s = te + gr + ch;
    if (s <= 1e-12) return {{tempo: 1/3, groove: 1/3, chroma: 1/3}};
    return {{tempo: te/s, groove: gr/s, chroma: ch/s}};
  }}
  function weights() {{
    const style = clamp(Number(els.style.value || 0), 0, 1);
    const mix = mixWeightsRaw();
    const m = 1 - style;
    return {{maest: style, tempo: m * mix.tempo, groove: m * mix.groove, chroma: m * mix.chroma, mix}};
  }}
  function setMixSliders(mix) {{
    els.te.value = String(clamp(mix.tempo, 0, 1));
    els.gr.value = String(clamp(mix.groove, 0, 1));
    els.ch.value = String(clamp(mix.chroma, 0, 1));
  }}
  function simplexPoint(mix) {{
    return {{
      x: mix.tempo * simplexVertices.tempo.x + mix.groove * simplexVertices.groove.x + mix.chroma * simplexVertices.chroma.x,
      y: mix.tempo * simplexVertices.tempo.y + mix.groove * simplexVertices.groove.y + mix.chroma * simplexVertices.chroma.y,
    }};
  }}
  function updateSimplexHandle(mix) {{
    const p = simplexPoint(mix);
    els.simplexHandle.setAttribute('cx', String(p.x));
    els.simplexHandle.setAttribute('cy', String(p.y));
  }}
  function simplexWeightsFromPoint(px, py) {{
    const a = simplexVertices.tempo;
    const b = simplexVertices.groove;
    const c = simplexVertices.chroma;
    const denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    let te = ((b.y - c.y) * (px - c.x) + (c.x - b.x) * (py - c.y)) / denom;
    let gr = ((c.y - a.y) * (px - c.x) + (a.x - c.x) * (py - c.y)) / denom;
    let ch = 1 - te - gr;
    te = clamp(te, 0, 1); gr = clamp(gr, 0, 1); ch = clamp(ch, 0, 1);
    const s = Math.max(1e-12, te + gr + ch);
    return {{tempo: te / s, groove: gr / s, chroma: ch / s}};
  }}
  function eventToSvgPoint(ev) {{
    const rect = els.simplex.getBoundingClientRect();
    return {{x: (ev.clientX - rect.left) * 360 / Math.max(1, rect.width), y: (ev.clientY - rect.top) * 320 / Math.max(1, rect.height)}};
  }}
  function setWeightsFromSimplexEvent(ev) {{
    const p = eventToSvgPoint(ev);
    setMixSliders(simplexWeightsFromPoint(p.x, p.y));
    renderAll();
  }}
  function layoutInterpolatedPoints(w) {{
    const target = [w.maest, w.tempo, w.groove, w.chroma];
    const ranked = layoutEntries.map(entry => {{
      const d2 = entry.weights.reduce((acc, val, idx) => acc + Math.pow(Number(val) - target[idx], 2), 0);
      return {{entry, d2}};
    }}).sort((a,b) => a.d2 - b.d2).slice(0, 12);
    if (!ranked.length) return layouts['1.0,0.0,0.0,0.0'] || [];
    if (ranked[0].d2 <= 1e-12) return ranked[0].entry.points;
    const weightsLocal = ranked.map(r => 1 / Math.max(1e-9, r.d2));
    const sum = weightsLocal.reduce((a,b) => a + b, 0);
    return ranked[0].entry.points.map((_, idx) => {{
      let x = 0, y = 0;
      ranked.forEach((r, ridx) => {{
        const c = weightsLocal[ridx] / sum;
        x += c * Number(r.entry.points[idx][0]);
        y += c * Number(r.entry.points[idx][1]);
      }});
      return [x, y];
    }});
  }}
  function baseTraceIndices() {{
    const names = new Set(records.map(r => String(r.genre)));
    const out = [];
    (plot.data || []).forEach((trace, i) => {{ if (names.has(String(trace.name))) out.push(i); }});
    return out;
  }}
  function updatePointCoordinates() {{
    const byGenre = new Map();
    for (const r of records) {{
      const pt = currentPoints[Number(r.idx)];
      if (!pt) continue;
      const g = String(r.genre);
      if (!byGenre.has(g)) byGenre.set(g, {{x: [], y: []}});
      byGenre.get(g).x.push(Number(pt[0]));
      byGenre.get(g).y.push(Number(pt[1]));
    }}
    for (const traceIdx of baseTraceIndices()) {{
      const trace = plot.data[traceIdx] || {{}};
      const vals = byGenre.get(String(trace.name));
      if (vals) Plotly.restyle(plot, {{x: [vals.x], y: [vals.y]}}, [traceIdx]);
    }}
  }}
  function rankedRows(sourceIdx, w) {{
    const candidates = ((simMap[String(sourceIdx)] || {{}}).candidates || []);
    const rows = candidates.map(c => {{
      const score = w.maest*Number(c.maest_score_norm||0) + w.tempo*Number(c.tempo_score_norm||0) + w.groove*Number(c.groove_score_norm||0) + w.chroma*Number(c.chroma_score_norm||0);
      return {{...c, score}};
    }}).sort((a,b) => b.score - a.score);
    const temp = Math.max(1e-6, Number(config.temperature || 0.08));
    const maxScore = rows.length ? Number(rows[0].score || 0) : 0;
    let sum = 0;
    rows.forEach(r => {{ r._exp = Math.exp((Number(r.score || 0) - maxScore) / temp); sum += r._exp; }});
    rows.forEach((r, i) => {{ r.rank = i + 1; r.probability = sum > 0 ? r._exp / sum : 0; }});
    return rows;
  }}
  function clearTraceSet(indices) {{
    if (!indices.length || !window.Plotly) return [];
    try {{ Plotly.deleteTraces(plot, indices.slice().sort((a,b) => b-a)); }} catch (err) {{}}
    return [];
  }}
  function addTraceSet(traces) {{
    if (!traces.length || !window.Plotly) return [];
    const start = (plot.data || []).length;
    try {{ Plotly.addTraces(plot, traces); }} catch (err) {{ return []; }}
    return Array.from({{length: traces.length}}, (_, i) => start + i);
  }}
  function colorForDelta(delta, alpha) {{
    const scale = Math.max(1e-6, Number(config.bpm_color_scale_pct || 0.10));
    const t = clamp(Number(delta || 0) / scale, -1, 1);
    const f = Math.abs(t);
    const base = [155, 155, 155], hot = [218, 65, 45], cold = [45, 98, 210];
    const target = t >= 0 ? hot : cold;
    const rgb = base.map((v, i) => Math.round(v + (target[i] - v) * f));
    return 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + alpha + ')';
  }}
  function linkTraces(sourceIdx, rows, limit, highlighted) {{
    const src = currentPoints[Number(sourceIdx)];
    if (!src) return [];
    const traces = [];
    for (const row of rows.slice(0, limit)) {{
      const dst = currentPoints[Number(row.idx)];
      if (!dst) continue;
      const prob = Number(row.probability || 0);
      traces.push({{
        type: 'scatter', mode: 'lines',
        x: [Number(src[0]), Number(dst[0])], y: [Number(src[1]), Number(dst[1])],
        line: {{color: colorForDelta(row.bpm_delta_frac, highlighted ? clamp(0.40 + prob * 2.2, 0.40, 0.90) : 0.13), width: highlighted ? clamp(1.8 + prob * 10.0, 1.8, 5.0) : 0.85, dash: highlighted ? 'solid' : 'dot'}},
        hoverinfo: 'skip', showlegend: false,
      }});
    }}
    return traces;
  }}
  function renderBackgroundLinks(w) {{
    backgroundTraceIndices = clearTraceSet(backgroundTraceIndices);
    const limit = Math.max(0, Number(config.background_links_per_song || 0));
    if (limit <= 0) return;
    const traces = [];
    for (const sourceIdx of Object.keys(simMap)) traces.push(...linkTraces(Number(sourceIdx), rankedRows(Number(sourceIdx), w), limit, false));
    backgroundTraceIndices = addTraceSet(traces);
  }}
  function renderSelectedLinks(w) {{
    selectedTraceIndices = clearTraceSet(selectedTraceIndices);
    if (selectedIdx === null) return;
    const limit = Math.max(0, Number(config.click_links_per_song || 0));
    selectedTraceIndices = addTraceSet(linkTraces(selectedIdx, rankedRows(selectedIdx, w), limit, true));
  }}
  function renderPanel(w) {{
    if (selectedIdx === null) return;
    const rows = rankedRows(selectedIdx, w).slice(0, Number(config.top_k_rows || 25));
    const body = rows.map(row => '<tr>' +
      '<td class="num">' + row.rank + '</td><td class="num">' + esc(row.track_number) + '</td><td>' + esc(row.mix_slug) + '</td>' +
      '<td>' + esc(row.title) + '</td><td>' + esc(row.artists) + '</td><td>' + esc(row.genre) + '</td><td>' + esc(row.key) + '</td>' +
      '<td class="num">' + signedPct(row.bpm_delta_frac) + '</td><td class="num">' + fmt(row.score,4) + '</td>' +
      '<td class="num">' + fmt(row.maest_score_norm,4) + '</td><td class="num">' + fmt(row.tempo_score_norm,4) + '</td><td class="num">' + fmt(row.groove_score_norm,4) + '</td><td class="num">' + fmt(row.chroma_score_norm,4) + '</td>' +
      '<td class="num">' + fmt(row.maest_similarity,4) + '</td><td class="num">' + fmt(row.tempo_similarity,4) + '</td><td class="num">' + fmt(row.groove_similarity,4) + '</td><td class="num">' + fmt(row.chroma_similarity,4) + '</td>' +
      '<td class="num">' + pct(row.probability) + '</td></tr>').join('');
    els.panel.innerHTML = '<b>Top ' + Number(config.top_k_rows || 25) + ' genre/mixability matches</b>' +
      '<div class="muted">Score = style slider + mixability triangle split across tempo, groove, and key.</div>' +
      '<div class="table-wrap"><table><thead><tr><th class="num">#</th><th class="num">Track</th><th>Mix</th><th>Title</th><th>Artists</th><th>Genre</th><th>Key</th>' +
      '<th class="num">dBPM%</th><th class="num">Score</th><th class="num">Style</th><th class="num">Tempo</th><th class="num">Groove</th><th class="num">Key</th>' +
      '<th class="num">Style raw</th><th class="num">Tempo raw</th><th class="num">Groove raw</th><th class="num">Key raw</th><th class="num">Prob</th></tr></thead><tbody>' + body + '</tbody></table></div>';
  }}
  function renderAll() {{
    const w = weights();
    selectedTraceIndices = clearTraceSet(selectedTraceIndices);
    backgroundTraceIndices = clearTraceSet(backgroundTraceIndices);
    els.styleVal.textContent = fmt(w.maest, 3);
    els.teVal.textContent = fmt(w.mix.tempo, 3);
    els.grVal.textContent = fmt(w.mix.groove, 3);
    els.chVal.textContent = fmt(w.mix.chroma, 3);
    updateSimplexHandle(w.mix);
    els.summary.innerHTML = 'Global weights: Style <b>' + pct(w.maest) + '</b>, tempo <b>' + pct(w.tempo) + '</b>, groove <b>' + pct(w.groove) + '</b>, key <b>' + pct(w.chroma) + '</b>.';
    currentPoints = layoutInterpolatedPoints(w);
    updatePointCoordinates();
    renderBackgroundLinks(w);
    renderSelectedLinks(w);
    renderPanel(w);
  }}
  [els.style, els.te, els.gr, els.ch].forEach(el => el.addEventListener('input', () => {{
    if (el !== els.style) setMixSliders(mixWeightsRaw());
    renderAll();
  }}));
  if (els.simplex) {{
    let draggingSimplex = false;
    els.simplex.addEventListener('pointerdown', ev => {{ draggingSimplex = true; els.simplex.setPointerCapture(ev.pointerId); setWeightsFromSimplexEvent(ev); }});
    els.simplex.addEventListener('pointermove', ev => {{ if (draggingSimplex) setWeightsFromSimplexEvent(ev); }});
    els.simplex.addEventListener('pointerup', ev => {{ draggingSimplex = false; try {{ els.simplex.releasePointerCapture(ev.pointerId); }} catch (err) {{}} }});
    els.simplex.addEventListener('pointercancel', () => {{ draggingSimplex = false; }});
  }}
  if (plot && plot.on) {{
    plot.on('plotly_click', ev => {{
      if (!ev || !ev.points || !ev.points.length) return;
      const c = ev.points[0].customdata || [];
      const idx = Number(c[13]);
      if (!Number.isFinite(idx)) return;
      selectedIdx = idx;
      els.meta.innerHTML = '<b>' + esc(c[0]) + '</b><br>Artists: ' + esc(c[1]) + '<br>Genre: ' + esc(c[2]) +
        '<br>Mix: ' + esc(c[14]) + '<br>Key: ' + esc(c[3]) + '<br>CSV BPM: ' + esc(c[4]) +
        '<br>Estimated BPM: ' + fmt(c[5], 2) + ' (confidence=' + fmt(c[6], 3) + ')<br>Track #: ' + esc(c[7]) + '<br>File: ' + esc(c[8]);
      if (c[9]) {{
        els.audio.src = c[9];
        const p = els.audio.play();
        if (p && p.catch) p.catch(() => {{}});
      }} else {{
        els.audio.removeAttribute('src');
        els.audio.load();
      }}
      renderSelectedLinks(weights());
      renderPanel(weights());
    }});
  }}
  setMixSliders(mixWeightsRaw());
  renderAll();
}})();
</script>
"""
    return "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{title}</title>",
            style,
            "</head>",
            "<body>",
            f"<h1>{title}</h1>",
            data_scripts,
            '<main class="workspace">',
            '<section class="plot-pane">',
            plot_html,
            '</section>',
            controls,
            '</main>',
            detail_panel,
            script,
            "</body>",
            "</html>",
        ]
    )


def export_dj_pacmap(
    *,
    project_root: Path = PROJECT_ROOT,
    mix_slugs: list[str] | None = None,
    output_file: Path | None = None,
    random_state: int = 7777,
    n_neighbors: int = 8,
    mn_ratio: float = 1.0,
    fp_ratio: float = 1.5,
    distance: str = "angular",
    step: float = 0.1,
    align_layouts: bool = True,
    temperature: float = 0.08,
    background_links_per_song: int = 4,
    click_links_per_song: int = 8,
    bpm_color_scale_pct: float = 0.10,
    pair_source: str = "neighbors-only",
    distance_combine: str = "l2",
    layout_init: str = "neighbor",
    control_mode: str = "genre-mixability",
    reducer: str = "pacmap",
    umap_min_dist: float = 0.1,
) -> Path:
    if control_mode not in CONTROL_MODE_CHOICES:
        raise ValueError(f"control_mode must be one of {CONTROL_MODE_CHOICES}, got {control_mode!r}.")
    if reducer not in REDUCER_CHOICES:
        raise ValueError(f"reducer must be one of {REDUCER_CHOICES}, got {reducer!r}.")

    project_root = project_root.expanduser().resolve()
    mix_slugs = mix_slugs or ["aries-mix", "ara-mix"]
    dataset_tag = "__".join(
        str((project_root / "music" / slug / f"{slug.replace('-', '_')}_tracks.csv").stem)
        for slug in mix_slugs
    )
    reducer_label = "PaCMAP" if reducer == "pacmap" else "UMAP"
    reducer_file_tag = "pacmap" if reducer == "pacmap" else "umap"
    output_file = output_file or (
        project_root / "data" / "exports" / f"{dataset_tag}_interactive_dj_{reducer_file_tag}.html"
    )
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records, features = _load_combined_records_and_features(
        project_root=project_root,
        mix_slugs=mix_slugs,
        maest_dir=(project_root / "data" / "maest_embeddings"),
        chroma_dir=(project_root / "data" / "chroma_embeddings"),
        tempo_dir=(project_root / "data" / "tempo_embeddings"),
        html_output_dir=output_file.parent,
        snippet_seconds=8.0,
        snippet_middle_fraction=0.66,
        snippet_hop_seconds=0.25,
        snippet_cache_overwrite=False,
    )
    is_legacy_simplex = control_mode in ("legacy-simplex", "legacy-discrete-simplex")
    if is_legacy_simplex:
        matrices = _component_matrices_3way(
            features,
            tempo_bandwidth=0.06,
            tempo_decay=0.5,
            tempo_allow_octave=True,
            tempo_octave_penalty=0.5,
            tempo_similarity_shape="gaussian",
            tempo_softflat_sharpness=8.0,
            tempo_use_confidence=False,
            harmonic_exact_weight=1.0,
            harmonic_first_fifth_weight=0.0,
            harmonic_second_fifth_weight=0.0,
            harmonic_other_weight=0.0,
            harmonic_self_normalize=True,
        )
        grid = _simplex_grid(step)
        if reducer == "pacmap":
            layouts = _compute_pacmap_knn_simplex_layouts(
                X_reference=features.maest,
                D_maest=matrices["maest_distance"],
                D_tempo=matrices["tempo_distance"],
                D_chroma=matrices["chroma_distance"],
                grid=grid,
                n_neighbors=n_neighbors,
                mn_ratio=mn_ratio,
                fp_ratio=fp_ratio,
                distance=distance,
                random_state=random_state,
                align=align_layouts,
                pair_source=pair_source,
                distance_combine=distance_combine,
                layout_init=layout_init,
            )
        else:
            layouts = _compute_umap_simplex_layouts(
                D_maest=matrices["maest_distance"],
                D_tempo=matrices["tempo_distance"],
                D_chroma=matrices["chroma_distance"],
                grid=grid,
                n_neighbors=n_neighbors,
                min_dist=umap_min_dist,
                random_state=random_state,
                align=align_layouts,
                distance_combine=distance_combine,
            )
        initial_key = _simplex_key((0.6, 0.2, 0.2))
        initial_coords = np.asarray(layouts.get(initial_key) or next(iter(layouts.values())), dtype=np.float32)
        similarity_payload = _build_similarity_payload_3way(
            records=records,
            maest_similarity=matrices["maest_similarity"],
            tempo_similarity=matrices["tempo_similarity"],
            chroma_similarity=matrices["chroma_similarity"],
            maest_distance=matrices["maest_distance"],
            tempo_distance=matrices["tempo_distance"],
            chroma_distance=matrices["chroma_distance"],
            temperature=temperature,
        )
        title = f"Interactive DJ {reducer_label}"
    else:
        groove_embeddings = _load_combined_groove_embeddings(
            project_root=project_root,
            mix_slugs=mix_slugs,
            groove_dir=(project_root / "data" / "groove_embeddings"),
        )
        matrices = _component_matrices_4way(
            features,
            groove_embeddings=groove_embeddings,
            tempo_bandwidth=0.06,
            tempo_decay=0.5,
            tempo_allow_octave=True,
            tempo_octave_penalty=0.5,
            tempo_similarity_shape="gaussian",
            tempo_softflat_sharpness=8.0,
            tempo_use_confidence=False,
            harmonic_exact_weight=1.0,
            harmonic_first_fifth_weight=0.0,
            harmonic_second_fifth_weight=0.0,
            harmonic_other_weight=0.0,
            harmonic_self_normalize=True,
        )
        grid4 = _simplex_grid_4way(step)
        if reducer == "pacmap":
            layouts = _compute_pacmap_knn_4way_layouts(
                X_reference=features.maest,
                D_maest=matrices["maest_distance"],
                D_tempo=matrices["tempo_distance"],
                D_groove=matrices["groove_distance"],
                D_chroma=matrices["chroma_distance"],
                grid=grid4,
                n_neighbors=n_neighbors,
                mn_ratio=mn_ratio,
                fp_ratio=fp_ratio,
                distance=distance,
                random_state=random_state,
                align=align_layouts,
                pair_source=pair_source,
                distance_combine=distance_combine,
                layout_init=layout_init,
            )
        else:
            layouts = _compute_umap_4way_layouts(
                D_maest=matrices["maest_distance"],
                D_tempo=matrices["tempo_distance"],
                D_groove=matrices["groove_distance"],
                D_chroma=matrices["chroma_distance"],
                grid=grid4,
                n_neighbors=n_neighbors,
                min_dist=umap_min_dist,
                random_state=random_state,
                align=align_layouts,
                distance_combine=distance_combine,
            )
        initial_key = _simplex_key_4way((0.5, 0.2, 0.2, 0.1))
        initial_coords = np.asarray(layouts.get(initial_key) or next(iter(layouts.values())), dtype=np.float32)
        similarity_payload = _build_similarity_payload_4way(
            records=records,
            maest_similarity=matrices["maest_similarity"],
            tempo_similarity=matrices["tempo_similarity"],
            groove_similarity=matrices["groove_similarity"],
            chroma_similarity=matrices["chroma_similarity"],
            maest_distance=matrices["maest_distance"],
            tempo_distance=matrices["tempo_distance"],
            groove_distance=matrices["groove_distance"],
            chroma_distance=matrices["chroma_distance"],
            temperature=temperature,
        )
        title = f"Interactive DJ {reducer_label}: Genre vs Mixability"

    plot_div_id = f"{dataset_tag}_{reducer}_simplex_maest_tempo_chroma".replace("-", "_")
    plot_html = _build_plot(
        records,
        initial_coords,
        plot_div_id=plot_div_id,
        title="",
        xaxis_title="",
        yaxis_title="",
        legend_title="Tagged Genre",
        show_axis_ticks=False,
        show_grid=False,
    )
    generation_settings = {
        "mix-slug": mix_slugs,
        "output-file": output_file.relative_to(project_root)
        if output_file.is_relative_to(project_root)
        else output_file,
        "control-mode": control_mode,
        "reducer": reducer,
        "layout-mode": "discrete" if control_mode == "legacy-discrete-simplex" else "interpolated",
        "distance-combine": distance_combine,
        "pair-source": pair_source if reducer == "pacmap" else "n/a",
        "layout-init": layout_init if reducer == "pacmap" else "n/a",
        "distance": distance if reducer == "pacmap" else "precomputed",
        "umap-min-dist": umap_min_dist if reducer == "umap" else "n/a",
        "random-state": random_state,
        "n-neighbors": n_neighbors,
        "mn-ratio": mn_ratio if reducer == "pacmap" else "n/a",
        "fp-ratio": fp_ratio if reducer == "pacmap" else "n/a",
        "step": step,
        "align-layouts": align_layouts,
        "temperature": temperature,
        "background-links-per-song": background_links_per_song,
        "click-links-per-song": click_links_per_song,
        "bpm-color-scale-pct": bpm_color_scale_pct,
        "track-count": len(records),
        "layout-count": len(layouts),
        "neighbor-pairs-per-layout": (
            len(records) * min(max(1, int(n_neighbors)), len(records) - 1)
            if reducer == "pacmap"
            else "n/a"
        ),
    }
    if is_legacy_simplex:
        html = _build_simplex_html(
            plot_html=plot_html,
            records=records,
            layouts=layouts,
            similarity_payload=similarity_payload,
            plot_div_id=plot_div_id,
            title=title,
            step=step,
            top_k_rows=25,
            temperature=temperature,
            background_links_per_song=background_links_per_song,
            click_links_per_song=click_links_per_song,
            bpm_color_scale_pct=bpm_color_scale_pct,
            generation_settings=generation_settings,
            layout_mode="discrete" if control_mode == "legacy-discrete-simplex" else "interpolated",
        )
    else:
        html = _build_genre_mixability_html(
            plot_html=plot_html,
            records=records,
            layouts=layouts,
            similarity_payload=similarity_payload,
            plot_div_id=plot_div_id,
            title=title,
            step=step,
            top_k_rows=25,
            temperature=temperature,
            background_links_per_song=background_links_per_song,
            click_links_per_song=click_links_per_song,
            bpm_color_scale_pct=bpm_color_scale_pct,
            generation_settings=generation_settings,
        )
    output_file.write_text(html, encoding="utf-8")
    print(f"Loaded aligned tracks: {len(records)}")
    print(f"Computed {reducer_label} simplex layouts: {len(layouts)}")
    print(f"Control mode: {control_mode}")
    print(f"Reducer: {reducer}")
    if reducer == "pacmap":
        print(f"PaCMAP pair source: {pair_source}")
        print(f"Layout init mode: {layout_init}")
        print(f"Neighbor pairs per layout: {len(records) * min(max(1, int(n_neighbors)), len(records) - 1)}")
    else:
        print(f"UMAP min_dist: {umap_min_dist}")
    print(f"Distance combine mode: {distance_combine}")
    print(f"Standalone HTML saved to: {output_file}")
    return output_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the interactive DJ embedding HTML.")
    parser.add_argument("--mix-slug", action="append", dest="mix_slugs")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument(
        "--reducer",
        choices=REDUCER_CHOICES,
        default="pacmap",
        help="Embedding reducer used to generate the precomputed layout grid.",
    )
    parser.add_argument("--random-state", type=int, default=7777)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--mn-ratio", type=float, default=0.5)
    parser.add_argument("--fp-ratio", type=float, default=1.5)
    parser.add_argument("--distance", default="angular")
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist value when --reducer umap is selected.",
    )
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--no-align-layouts", action="store_true")
    parser.add_argument("--background-links-per-song", type=int, default=3)
    parser.add_argument("--click-links-per-song", type=int, default=5)
    parser.add_argument("--bpm-color-scale-pct", type=float, default=0.10)
    parser.add_argument(
        "--control-mode",
        choices=CONTROL_MODE_CHOICES,
        default="genre-mixability",
        help=(
            "`genre-mixability` uses a Genre/style slider plus Tempo/Groove/Key simplex. "
            "`legacy-simplex` keeps the older Genre/Tempo/Key simplex with interpolated layouts. "
            "`legacy-discrete-simplex` keeps the older simplex but snaps to exact precomputed layouts."
        ),
    )
    parser.add_argument(
        "--pair-source",
        choices=PACMAP_PAIR_SOURCE_CHOICES,
        default="neighbors-only",
        help=(
            "Which distances define PaCMAP pair constraints. "
            "`neighbors-only` keeps existing behavior: nearest pairs use the weighted simplex distance, "
            "while PaCMAP samples mid/far pairs from the MAEST reference features. "
            "`combined-all` derives nearest, mid-near, and far pairs from the weighted simplex distance."
        ),
    )
    parser.add_argument(
        "--distance-combine",
        choices=DISTANCE_COMBINE_CHOICES,
        default="l2",
        help=(
            "How to combine normalized MAEST/tempo/chroma distances. "
            "`l2` keeps existing root-weighted-squares behavior; "
            "`l1` uses a weighted arithmetic mean for more linear modality balancing."
        ),
    )
    parser.add_argument(
        "--layout-init",
        choices=LAYOUT_INIT_CHOICES,
        default="neighbor",
        help=(
            "PaCMAP initialization for each simplex layout. "
            "`neighbor` keeps existing behavior by initializing from a nearby already-computed simplex layout; "
            "`pca` initializes each layout independently from PCA; "
            "`random` initializes each layout independently at random."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_dj_pacmap(
        mix_slugs=args.mix_slugs,
        output_file=args.output_file,
        random_state=args.random_state,
        n_neighbors=args.n_neighbors,
        mn_ratio=args.mn_ratio,
        fp_ratio=args.fp_ratio,
        distance=args.distance,
        step=args.step,
        align_layouts=not args.no_align_layouts,
        background_links_per_song=args.background_links_per_song,
        click_links_per_song=args.click_links_per_song,
        bpm_color_scale_pct=args.bpm_color_scale_pct,
        pair_source=args.pair_source,
        distance_combine=args.distance_combine,
        layout_init=args.layout_init,
        control_mode=args.control_mode,
        reducer=args.reducer,
        umap_min_dist=args.umap_min_dist,
    )


export_interactive_pacmap_knn_simplex = export_dj_pacmap


if __name__ == "__main__":
    main()
