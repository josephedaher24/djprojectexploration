"""Harmonic compatibility metrics for 12-D pitch-class vectors.

Assumes vectors are in chromatic C-order by default:
[C, C#, D, D#, E, F, F#, G, G#, A, A#, B]
"""

from __future__ import annotations

import numpy as np

# Canonical 12-bin chromatic order expected by this module.
PITCH_CLASS_C_ORDER = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _as_length_12(v: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    if arr.size != 12:
        raise ValueError(f"Expected 12-D pitch-class vector, got shape {arr.shape}.")
    return arr


def normalize_unit_sum(v: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Normalize a vector so entries sum to 1; returns zeros if sum is 0."""
    arr = _as_length_12(v).astype(np.float32)
    s = float(np.sum(arr))
    if s == 0.0:
        return np.zeros_like(arr)
    return arr / s


def fifth_distance(i: int, j: int) -> int:
    """Circle-of-fifths distance between pitch-class indices i and j."""
    d = (7 * (int(i) - int(j))) % 12
    return int(min(d, 12 - d))


def build_fifth_kernel(
    exact_weight: float = 1.0,
    first_fifth_weight: float = 0.4,
    second_fifth_weight: float = 0.15,
    other_weight: float = 0.0,
) -> np.ndarray:
    """Build 12x12 fifth-aware harmonic kernel."""
    K = np.full((12, 12), float(other_weight), dtype=np.float32)
    for i in range(12):
        for j in range(12):
            d = fifth_distance(i, j)
            if d == 0:
                K[i, j] = float(exact_weight)
            elif d == 1:
                K[i, j] = float(first_fifth_weight)
            elif d == 2:
                K[i, j] = float(second_fifth_weight)
    return K


def plain_pitch_similarity(
    x: np.ndarray | list[float] | tuple[float, ...],
    y: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    """Plain overlap similarity (dot product after unit-sum normalization)."""
    xn = normalize_unit_sum(x)
    yn = normalize_unit_sum(y)
    return float(np.dot(xn, yn))


def fifth_aware_similarity(
    x: np.ndarray | list[float] | tuple[float, ...],
    y: np.ndarray | list[float] | tuple[float, ...],
    kernel: np.ndarray | None = None,
) -> float:
    """Fifth-aware compatibility score x^T K y after unit-sum normalization."""
    xn = normalize_unit_sum(x)
    yn = normalize_unit_sum(y)

    if kernel is None:
        K = build_fifth_kernel()
    else:
        K = np.asarray(kernel, dtype=np.float32)
        if K.shape != (12, 12):
            raise ValueError(f"Expected kernel shape (12, 12), got {K.shape}.")

    return float(xn @ K @ yn)


def pairwise_fifth_aware_similarity_matrix(
    vectors: np.ndarray,
    kernel: np.ndarray | None = None,
) -> np.ndarray:
    """Compute pairwise fifth-aware similarity for N x 12 pitch-class vectors."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 12:
        raise ValueError(f"Expected shape (N, 12), got {arr.shape}.")

    if kernel is None:
        K = build_fifth_kernel()
    else:
        K = np.asarray(kernel, dtype=np.float32)
        if K.shape != (12, 12):
            raise ValueError(f"Expected kernel shape (12, 12), got {K.shape}.")

    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    arr_n = arr / row_sums

    # S = X K X^T
    return (arr_n @ K @ arr_n.T).astype(np.float32)

