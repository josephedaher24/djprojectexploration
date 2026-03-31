"""Multimodal song-to-song compatibility using MAEST, chroma, and DEAM features."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SongMetadata:
    track_number: int
    title: str
    artist: str
    filename: str
    genre: str


@dataclass(frozen=True)
class SongFeatureSet:
    metadata: list[SongMetadata]
    maest: np.ndarray  # shape: [N, Dm]
    chroma: np.ndarray  # shape: [N, Dc]
    deam: np.ndarray  # shape: [N, 2], columns: [valence, arousal]

    @property
    def filenames(self) -> list[str]:
        return [m.filename for m in self.metadata]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _unit_norm_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return X / norms


def _cosine_similarity_vector_to_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    m = np.asarray(matrix, dtype=np.float32)
    qn = _unit_norm_rows(q)[0]
    mn = _unit_norm_rows(m)
    return (mn @ qn).astype(np.float32)


def _pairwise_cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    m = _unit_norm_rows(np.asarray(matrix, dtype=np.float32))
    return (m @ m.T).astype(np.float32)


def _deam_similarity_vector_to_matrix(
    query: np.ndarray,
    matrix: np.ndarray,
    sigma: float | None = None,
) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    m = np.asarray(matrix, dtype=np.float32)
    diff = m - q
    d2 = np.sum(diff * diff, axis=1)

    if sigma is None:
        distances = np.sqrt(d2)
        nonzero = distances[distances > 0]
        sigma = float(np.median(nonzero)) if nonzero.size else 0.15
    sigma = max(float(sigma), 1e-6)
    return np.exp(-(d2 / (2.0 * sigma * sigma))).astype(np.float32)


def _pairwise_deam_similarity_matrix(
    matrix: np.ndarray,
    sigma: float | None = None,
) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float32)
    diff = m[:, None, :] - m[None, :, :]
    d2 = np.sum(diff * diff, axis=2)

    if sigma is None:
        distances = np.sqrt(d2)
        nonzero = distances[distances > 0]
        sigma = float(np.median(nonzero)) if nonzero.size else 0.15
    sigma = max(float(sigma), 1e-6)
    return np.exp(-(d2 / (2.0 * sigma * sigma))).astype(np.float32)


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-6)
    x = np.asarray(logits, dtype=np.float64) / t
    x = x - np.max(x)
    e = np.exp(x)
    z = np.sum(e)
    if z == 0.0:
        return np.zeros_like(x, dtype=np.float32)
    return (e / z).astype(np.float32)


def load_aries_mix_feature_set(
    mix_csv_path: str | Path = PROJECT_ROOT / "music" / "aries-mix" / "aries_mix_tracks.csv",
    maest_dir: str | Path = PROJECT_ROOT / "data" / "maest_embeddings",
    chroma_dir: str | Path = PROJECT_ROOT / "data" / "chroma_embeddings",
    deam_csv_path: str | Path = PROJECT_ROOT / "data" / "deam_valence_arousal" / "deam_track_predictions.csv",
    chroma_use_base_only: bool = True,
    deam_normalize_to_unit: bool = True,
) -> SongFeatureSet:
    """Load aligned MAEST/chroma/DEAM features for tracks in aries-mix CSV order."""
    mix_csv = Path(mix_csv_path)
    maest_root = Path(maest_dir)
    chroma_root = Path(chroma_dir)
    deam_csv = Path(deam_csv_path)

    if not mix_csv.exists():
        raise FileNotFoundError(f"Mix CSV not found: {mix_csv}")
    if not maest_root.exists():
        raise FileNotFoundError(f"MAEST directory not found: {maest_root}")
    if not chroma_root.exists():
        raise FileNotFoundError(f"Chroma directory not found: {chroma_root}")
    if not deam_csv.exists():
        raise FileNotFoundError(f"DEAM CSV not found: {deam_csv}")

    deam_lookup: dict[str, tuple[float, float]] = {}
    with deam_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("filename") or "").strip().lower()
            if not name:
                continue
            try:
                valence = float(row["valence"])
                arousal = float(row["arousal"])
            except (TypeError, ValueError, KeyError):
                continue
            deam_lookup[name] = (valence, arousal)

    metadata: list[SongMetadata] = []
    maest_vectors: list[np.ndarray] = []
    chroma_vectors: list[np.ndarray] = []
    deam_vectors: list[np.ndarray] = []

    with mix_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tn = (row.get("track_number") or row.get("#") or "").strip()
            title = (row.get("title") or "").strip()
            artist = (row.get("artists") or "").strip()
            filename = (row.get("mp3_name") or "").strip()
            genre = (row.get("genre") or "").strip()

            if not tn or not filename:
                continue
            try:
                track_number = int(tn)
            except ValueError:
                continue

            stem = Path(filename).stem
            maest_file = maest_root / f"{stem}_maest_embedding.json"
            chroma_file = chroma_root / f"{stem}_chroma_embedding.json"
            deam_key = filename.lower()

            if not maest_file.exists() or not chroma_file.exists() or deam_key not in deam_lookup:
                continue

            maest_payload = _read_json(maest_file)
            chroma_payload = _read_json(chroma_file)

            maest_vec = np.asarray(maest_payload.get("embedding", []), dtype=np.float32).reshape(-1)
            chroma_full = np.asarray(chroma_payload.get("embedding", []), dtype=np.float32).reshape(-1)
            if maest_vec.size == 0 or chroma_full.size == 0:
                continue

            if chroma_use_base_only:
                base_dim = int(chroma_payload.get("base_embedding_dimension", 24))
                if chroma_full.size < base_dim:
                    continue
                chroma_vec = chroma_full[:base_dim]
            else:
                chroma_vec = chroma_full

            valence, arousal = deam_lookup[deam_key]
            deam_vec = np.array([valence, arousal], dtype=np.float32)

            metadata.append(
                SongMetadata(
                    track_number=track_number,
                    title=title if title else stem,
                    artist=artist,
                    filename=filename,
                    genre=genre,
                )
            )
            maest_vectors.append(maest_vec)
            chroma_vectors.append(chroma_vec)
            deam_vectors.append(deam_vec)

    if not metadata:
        raise RuntimeError("No tracks had all three embeddings (MAEST/chroma/DEAM) available.")

    order = np.argsort([m.track_number for m in metadata])
    metadata_sorted = [metadata[i] for i in order]
    maest_arr = np.vstack([maest_vectors[i] for i in order]).astype(np.float32)
    chroma_arr = np.vstack([chroma_vectors[i] for i in order]).astype(np.float32)
    deam_arr = np.vstack([deam_vectors[i] for i in order]).astype(np.float32)

    if deam_normalize_to_unit:
        # DEAM targets live in [1, 9]. Map to [0, 1] for stable distance scales.
        deam_arr = (deam_arr - 1.0) / 8.0
        deam_arr = np.clip(deam_arr, 0.0, 1.0).astype(np.float32)

    return SongFeatureSet(
        metadata=metadata_sorted,
        maest=maest_arr,
        chroma=chroma_arr,
        deam=deam_arr,
    )


def compatible_song_distribution(
    features: SongFeatureSet,
    seed_track_number: int,
    maest_weight: float = 0.5,
    chroma_weight: float = 0.3,
    deam_weight: float = 0.2,
    temperature: float = 0.08,
    candidate_top_k: int | None = None,
    deam_sigma: float | None = None,
) -> list[dict]:
    """Return ranked compatible-song distribution from a seed track.

    The compatibility logit is a weighted combination of:
    - MAEST cosine similarity (mapped from [-1,1] to [0,1])
    - Chroma cosine similarity (mapped from [-1,1] to [0,1])
    - DEAM Gaussian similarity in valence/arousal space
    """
    n = len(features.metadata)
    if n < 2:
        raise ValueError("Need at least two tracks to compute compatibility distribution.")

    seed_idx = None
    for i, meta in enumerate(features.metadata):
        if meta.track_number == int(seed_track_number):
            seed_idx = i
            break
    if seed_idx is None:
        raise ValueError(f"Seed track number {seed_track_number} not found in feature set.")

    maest_cos = _cosine_similarity_vector_to_matrix(features.maest[seed_idx], features.maest)
    chroma_cos = _cosine_similarity_vector_to_matrix(features.chroma[seed_idx], features.chroma)
    deam_sim = _deam_similarity_vector_to_matrix(features.deam[seed_idx], features.deam, sigma=deam_sigma)

    maest_sim = 0.5 * (maest_cos + 1.0)
    chroma_sim = 0.5 * (chroma_cos + 1.0)

    logits = (
        float(maest_weight) * maest_sim
        + float(chroma_weight) * chroma_sim
        + float(deam_weight) * deam_sim
    ).astype(np.float32)

    candidate_mask = np.ones(n, dtype=bool)
    candidate_mask[seed_idx] = False

    if candidate_top_k is not None:
        k = max(1, min(int(candidate_top_k), n - 1))
        pool = np.where(candidate_mask)[0]
        order = pool[np.argsort(maest_sim[pool])[::-1]]
        keep = set(order[:k].tolist())
        candidate_mask = np.array([(i in keep) for i in range(n)], dtype=bool)
        candidate_mask[seed_idx] = False

    candidate_indices = np.where(candidate_mask)[0]
    candidate_logits = logits[candidate_indices]
    probs = _softmax(candidate_logits, temperature=temperature)

    rows: list[dict] = []
    for j, idx in enumerate(candidate_indices):
        meta = features.metadata[idx]
        rows.append(
            {
                "track_number": meta.track_number,
                "title": meta.title,
                "artist": meta.artist,
                "filename": meta.filename,
                "genre": meta.genre,
                "maest_similarity": float(maest_sim[idx]),
                "chroma_similarity": float(chroma_sim[idx]),
                "deam_similarity": float(deam_sim[idx]),
                "compatibility_logit": float(candidate_logits[j]),
                "probability": float(probs[j]),
            }
        )

    rows.sort(key=lambda r: r["probability"], reverse=True)
    return rows


def combined_similarity_matrix(
    features: SongFeatureSet,
    maest_weight: float = 0.5,
    chroma_weight: float = 0.3,
    deam_weight: float = 0.2,
    deam_sigma: float | None = None,
) -> np.ndarray:
    """Pairwise weighted compatibility matrix across all songs."""
    maest_cos = _pairwise_cosine_similarity_matrix(features.maest)
    chroma_cos = _pairwise_cosine_similarity_matrix(features.chroma)
    deam_sim = _pairwise_deam_similarity_matrix(features.deam, sigma=deam_sigma)

    maest_sim = 0.5 * (maest_cos + 1.0)
    chroma_sim = 0.5 * (chroma_cos + 1.0)

    return (
        float(maest_weight) * maest_sim
        + float(chroma_weight) * chroma_sim
        + float(deam_weight) * deam_sim
    ).astype(np.float32)

