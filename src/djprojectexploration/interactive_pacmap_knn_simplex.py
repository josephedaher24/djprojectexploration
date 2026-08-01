"""Export the canonical interactive DJ embedding visualization."""

from __future__ import annotations

import argparse
import html
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from djprojectexploration.frontend_assets import frontend_asset_text, render_standalone_document
from djprojectexploration.interactive_visualization_common import (
    PROJECT_ROOT,
    _align_to_reference,
    _build_plot,
    _json_script_payload,
    _load_combined_records_and_features,
    _neighbor_pairs_from_distance,
    _normalize_distance_matrix,
    resolve_tracklist_sources,
)
from djprojectexploration.layout_cache import cached_layouts
from djprojectexploration.multimodal_compatibility import (
    _build_harmonic_kernel,
    _pairwise_cosine_similarity_matrix,
    _pairwise_fifth_aware_similarity_matrix,
    _pairwise_tempo_similarity_matrix,
)
from djprojectexploration.pacmap_settings import (
    DISTANCE_COMBINE_CHOICES,
    LAYOUT_INIT_CHOICES,
    PACMAP_PAIR_SOURCE_CHOICES,
    PacmapSettings,
    add_pacmap_args,
    ensure_numba_cache_dir,
    pacmap_settings_from_args,
)

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
    mix_slugs: list[str] | None = None,
    tracklist_paths: list[Path] | None = None,
    groove_dir: Path,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for mix_slug, tracklist_csv in resolve_tracklist_sources(
        project_root=project_root,
        mix_slugs=mix_slugs,
        tracklist_paths=tracklist_paths,
    ):
        csv_stem = tracklist_csv.stem
        groove_file = groove_dir / f"{csv_stem}.npz"
        if not groove_file.exists():
            raise FileNotFoundError(
                f"Groove embedding collection not found: {groove_file}. "
                "Generate it with `uv run djprojectexploration-groove-playlist "
                f"{tracklist_csv}`."
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
        ensure_numba_cache_dir()
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


# Cached on its keyword arguments + a transitive hash of every djprojectexploration
# function it reaches. Keep this function pure in its kwargs: if it ever reads an
# env var, a module global, or mutable state, the cache goes silently stale.
@cached_layouts
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
        ensure_numba_cache_dir()
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

    settings_panel = _build_settings_panel(generation_settings)
    controls = frontend_asset_text("templates/interactive_pacmap_simplex_controls.html").replace(
        "{{SETTINGS_PANEL}}", settings_panel
    )
    detail_panel = frontend_asset_text("templates/interactive_pacmap_detail_panel.html").replace(
        "{{TRACK_META_TEXT}}", "Click a point to play its snippet and show Genre/tempo/key recommendations."
    )
    body_html = (
        frontend_asset_text("templates/interactive_pacmap_body.html")
        .replace("{{TITLE}}", html.escape(title))
        .replace("{{PLOT_HTML}}", plot_html)
        .replace("{{CONTROLS_HTML}}", controls)
        .replace("{{DETAIL_PANEL}}", detail_panel)
    )
    return render_standalone_document(
        title=title,
        css_asset="static/interactive_pacmap_simplex.css",
        body_html=body_html,
        data_scripts=data_scripts,
        script_asset="static/interactive_pacmap_simplex.js",
        script_replacements={"__PLOT_ID__": plot_div_id},
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
    layout_selection_mode: str,
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
            f'<script id="simplex-config-json" type="application/json">{_json_script_payload({"step": step, "top_k_rows": top_k_rows, "temperature": temperature, "background_links_per_song": background_links_per_song, "click_links_per_song": click_links_per_song, "bpm_color_scale_pct": bpm_color_scale_pct, "control_mode": "genre-mixability", "layout_selection_mode": layout_selection_mode})}</script>',
        ]
    )

    settings_panel = _build_settings_panel(generation_settings)
    controls = frontend_asset_text("templates/interactive_pacmap_genre_mixability_controls.html").replace(
        "{{SETTINGS_PANEL}}", settings_panel
    )
    detail_panel = frontend_asset_text("templates/interactive_pacmap_detail_panel.html").replace(
        "{{TRACK_META_TEXT}}", "Click a point to play its snippet and show genre/mixability recommendations."
    )
    body_html = (
        frontend_asset_text("templates/interactive_pacmap_body.html")
        .replace("{{TITLE}}", html.escape(title))
        .replace("{{PLOT_HTML}}", plot_html)
        .replace("{{CONTROLS_HTML}}", controls)
        .replace("{{DETAIL_PANEL}}", detail_panel)
    )
    return render_standalone_document(
        title=title,
        css_asset="static/interactive_pacmap_genre_mixability.css",
        body_html=body_html,
        data_scripts=data_scripts,
        script_asset="static/interactive_pacmap_genre_mixability.js",
        script_replacements={"__PLOT_ID__": plot_div_id},
    )



def export_dj_pacmap(
    *,
    project_root: Path = PROJECT_ROOT,
    mix_slugs: list[str] | None = None,
    tracklist_paths: list[Path] | None = None,
    dataset_name: str | None = None,
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
    pacmap_settings: PacmapSettings | None = None,
) -> Path:
    if control_mode not in CONTROL_MODE_CHOICES:
        raise ValueError(f"control_mode must be one of {CONTROL_MODE_CHOICES}, got {control_mode!r}.")
    if reducer not in REDUCER_CHOICES:
        raise ValueError(f"reducer must be one of {REDUCER_CHOICES}, got {reducer!r}.")

    if pacmap_settings is not None:
        n_neighbors = pacmap_settings.n_neighbors
        mn_ratio = pacmap_settings.mn_ratio
        fp_ratio = pacmap_settings.fp_ratio
        distance = pacmap_settings.distance
        step = pacmap_settings.step
        align_layouts = pacmap_settings.align_layouts
        pair_source = pacmap_settings.pair_source
        distance_combine = pacmap_settings.distance_combine
        layout_init = pacmap_settings.layout_init
        random_state = pacmap_settings.random_state
        static_layout = pacmap_settings.static_layout
        layout_selection_mode = pacmap_settings.layout_selection_mode
    else:
        static_layout = False
        layout_selection_mode = "discrete" if control_mode == "legacy-discrete-simplex" else "interpolated"

    project_root = project_root.expanduser().resolve()
    sources = resolve_tracklist_sources(
        project_root=project_root,
        mix_slugs=mix_slugs,
        tracklist_paths=tracklist_paths,
    )
    mix_slugs = [slug for slug, _ in sources]
    dataset_tag = dataset_name or "__".join(path.stem for _, path in sources)
    reducer_label = "PaCMAP" if reducer == "pacmap" else "UMAP"
    reducer_file_tag = "pacmap" if reducer == "pacmap" else "umap"
    output_file = output_file or (
        project_root / "data" / "exports" / f"{dataset_tag}_interactive_dj_{reducer_file_tag}.html"
    )
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records, features = _load_combined_records_and_features(
        project_root=project_root,
        mix_slugs=None,
        tracklist_paths=[path for _, path in sources],
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
        grid = [(0.6, 0.2, 0.2)] if static_layout else _simplex_grid(step)
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
            mix_slugs=None,
            tracklist_paths=[path for _, path in sources],
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
        grid4 = [(0.5, 0.2, 0.2, 0.1)] if static_layout else _simplex_grid_4way(step)
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
        "dataset-name": dataset_name or "n/a",
        "source": [str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path) for _, path in sources],
        "output-file": output_file.relative_to(project_root)
        if output_file.is_relative_to(project_root)
        else output_file,
        "control-mode": control_mode,
        "reducer": reducer,
        "layout-mode": "static" if static_layout else layout_selection_mode,
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
            layout_mode="discrete" if layout_selection_mode == "discrete" else "interpolated",
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
            layout_selection_mode="discrete" if layout_selection_mode == "discrete" else "interpolated",
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
    parser.add_argument("--tracklist", action="append", type=Path, default=None, help="Tracklist CSV to include; repeatable.")
    parser.add_argument("--dataset-name", default=None, help="Dataset/output name used when --tracklist is provided.")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument(
        "--reducer",
        choices=REDUCER_CHOICES,
        default="pacmap",
        help="Embedding reducer used to generate the precomputed layout grid.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist value when --reducer umap is selected.",
    )
    add_pacmap_args(parser, include_static_layout=True)
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pacmap_settings = pacmap_settings_from_args(args)
    export_dj_pacmap(
        mix_slugs=args.mix_slugs,
        tracklist_paths=args.tracklist,
        dataset_name=args.dataset_name,
        output_file=args.output_file,
        pacmap_settings=pacmap_settings,
        background_links_per_song=args.background_links_per_song,
        click_links_per_song=args.click_links_per_song,
        bpm_color_scale_pct=args.bpm_color_scale_pct,
        control_mode=args.control_mode,
        reducer=args.reducer,
        umap_min_dist=args.umap_min_dist,
    )


export_interactive_pacmap_knn_simplex = export_dj_pacmap


if __name__ == "__main__":
    main()
