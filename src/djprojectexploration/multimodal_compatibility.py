"""Multimodal song-to-song compatibility using MAEST, chroma, and tempo NPZ features."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from djprojectexploration.harmonic_compatibility import build_fifth_kernel


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
    chroma_pitch: np.ndarray  # shape: [N, 12], pitch-class profiles for fifth-aware scoring
    tempo_bpm: np.ndarray  # shape: [N]
    tempo_confidence: np.ndarray  # shape: [N], expected in [0,1]

    @property
    def filenames(self) -> list[str]:
        return [m.filename for m in self.metadata]


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


def _unit_sum_rows(X: np.ndarray) -> np.ndarray:
    sums = np.sum(X, axis=1, keepdims=True)
    sums = np.where(sums == 0.0, 1.0, sums)
    return X / sums


_FIFTH_KERNEL_DEFAULT = build_fifth_kernel()


def _build_harmonic_kernel(
    *,
    exact_weight: float = 1.0,
    first_fifth_weight: float = 0.4,
    second_fifth_weight: float = 0.15,
    other_weight: float = 0.0,
) -> np.ndarray:
    return build_fifth_kernel(
        exact_weight=float(exact_weight),
        first_fifth_weight=float(first_fifth_weight),
        second_fifth_weight=float(second_fifth_weight),
        other_weight=float(other_weight),
    )


def _fifth_aware_similarity_vector_to_matrix(
    query: np.ndarray,
    matrix: np.ndarray,
    *,
    kernel: np.ndarray | None = None,
    normalize_by_self: bool = False,
) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    m = np.asarray(matrix, dtype=np.float32)
    if q.shape[1] != 12 or m.ndim != 2 or m.shape[1] != 12:
        raise ValueError(f"Expected query shape (12,) and matrix shape (N, 12), got {q.shape} and {m.shape}.")

    K = _FIFTH_KERNEL_DEFAULT if kernel is None else np.asarray(kernel, dtype=np.float32)
    if K.shape != (12, 12):
        raise ValueError(f"Expected harmonic kernel shape (12, 12), got {K.shape}.")

    qn = _unit_sum_rows(q)[0]
    mn = _unit_sum_rows(m)
    raw = ((mn @ K) @ qn).astype(np.float32)
    if not normalize_by_self:
        return raw

    q_self = float(qn @ K @ qn)
    m_self = np.sum((mn @ K) * mn, axis=1).astype(np.float32)
    den = np.sqrt(max(q_self, 0.0) * np.clip(m_self, 0.0, None)).astype(np.float32)
    return np.divide(raw, den, out=np.zeros_like(raw), where=den > 0).astype(np.float32)


def _pairwise_fifth_aware_similarity_matrix(
    vectors: np.ndarray,
    *,
    kernel: np.ndarray | None = None,
    normalize_by_self: bool = False,
) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 12:
        raise ValueError(f"Expected shape (N, 12), got {arr.shape}.")
    K = _FIFTH_KERNEL_DEFAULT if kernel is None else np.asarray(kernel, dtype=np.float32)
    if K.shape != (12, 12):
        raise ValueError(f"Expected harmonic kernel shape (12, 12), got {K.shape}.")
    arrn = _unit_sum_rows(arr)
    raw = (arrn @ K @ arrn.T).astype(np.float32)
    if not normalize_by_self:
        return raw

    diag = np.diag(raw).astype(np.float32)
    den = np.sqrt(np.outer(np.clip(diag, 0.0, None), np.clip(diag, 0.0, None))).astype(np.float32)
    norm = np.divide(raw, den, out=np.zeros_like(raw), where=den > 0).astype(np.float32)
    valid_diag = diag > 0
    norm[np.diag_indices_from(norm)] = np.where(valid_diag, 1.0, 0.0)
    return norm


def _normalize_tempo_similarity_shape(shape: str) -> str:
    key = str(shape).strip().lower().replace("_", "").replace(" ", "")
    if key in {"gaussian", "gauss"}:
        return "gaussian"
    if key in {"exp", "exponential"}:
        return "exp"
    if key in {"flat+exp", "flatexp"}:
        return "flat+exp"
    if key in {"softflat+exp", "softflatexp"}:
        return "softflat+exp"
    raise ValueError(
        "Unsupported tempo_similarity_shape. "
        "Use one of: 'gaussian', 'exp', 'flat+exp', 'softflat+exp'."
    )


def _tempo_distance_to_similarity_score(
    distance: np.ndarray,
    *,
    sigma: float,
    decay: float,
    shape: str,
    softflat_sharpness: float = 8.0,
) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if softflat_sharpness <= 0:
        raise ValueError("softflat_sharpness must be > 0")

    z = np.asarray(distance, dtype=np.float64) / float(sigma)
    z = np.clip(z, 0.0, None)
    mode = _normalize_tempo_similarity_shape(shape)

    if mode == "gaussian":
        return np.exp(-float(decay) * (z**2))
    if mode == "exp":
        return np.exp(-float(decay) * z)
    if mode == "flat+exp":
        # Hard plateau in-band (z <= 1), exponential decay outside the band.
        over = np.maximum(z - 1.0, 0.0)
        return np.exp(-float(decay) * over)

    # mode == "softflat+exp"
    # Smooth approximation of max(z - 1, 0): ~0 in-band, ~linear out-of-band.
    over_soft = np.logaddexp(0.0, float(softflat_sharpness) * (z - 1.0)) / float(
        softflat_sharpness
    )
    return np.exp(-float(decay) * over_soft)


def _tempo_similarity_vector_to_matrix(
    query_tempo_bpm: float,
    query_confidence: float,
    tempo_bpm: np.ndarray,
    tempo_confidence: np.ndarray,
    *,
    bandwidth: float = 0.06,
    decay: float = 0.5,
    allow_octave: bool = True,
    octave_penalty: float = 0.5,
    tempo_similarity_shape: str = "gaussian",
    softflat_sharpness: float = 8.0,
    use_confidence: bool = True,
) -> np.ndarray:
    """Ratio-aware tempo similarity in log-tempo domain."""
    tempos = np.asarray(tempo_bpm, dtype=np.float32).reshape(-1)
    confs = np.asarray(tempo_confidence, dtype=np.float32).reshape(-1)

    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")

    out = np.zeros_like(tempos, dtype=np.float32)

    if not np.isfinite(query_tempo_bpm) or query_tempo_bpm <= 0:
        return out

    valid = np.isfinite(tempos) & (tempos > 0)
    if not np.any(valid):
        return out

    sigma = float(np.log1p(float(bandwidth)))
    xq = float(np.log(float(query_tempo_bpm)))
    x = np.log(tempos[valid].astype(np.float64))

    d_main = np.abs(x - xq)
    if allow_octave:
        l2 = float(np.log(2.0))
        d_half = np.abs((x - l2) - xq)
        d_double = np.abs((x + l2) - xq)
        stack = np.stack([d_main, d_half, d_double], axis=0)
        choice = np.argmin(stack, axis=0)
        d = np.min(stack, axis=0)
        penalty = np.where(choice == 0, 1.0, float(octave_penalty))
    else:
        d = d_main
        penalty = np.ones_like(d)

    score = penalty * _tempo_distance_to_similarity_score(
        d,
        sigma=sigma,
        decay=float(decay),
        shape=tempo_similarity_shape,
        softflat_sharpness=float(softflat_sharpness),
    )

    if use_confidence:
        qc = float(np.clip(query_confidence, 0.0, 1.0))
        dc = np.clip(confs[valid].astype(np.float64), 0.0, 1.0)
        score *= np.sqrt(qc * dc)

    out[valid] = np.clip(score, 0.0, 1.0).astype(np.float32)
    return out


def _pairwise_tempo_similarity_matrix(
    tempo_bpm: np.ndarray,
    tempo_confidence: np.ndarray,
    *,
    bandwidth: float = 0.06,
    decay: float = 0.5,
    allow_octave: bool = True,
    octave_penalty: float = 0.5,
    tempo_similarity_shape: str = "gaussian",
    softflat_sharpness: float = 8.0,
    use_confidence: bool = True,
) -> np.ndarray:
    tempos = np.asarray(tempo_bpm, dtype=np.float32).reshape(-1)
    confs = np.asarray(tempo_confidence, dtype=np.float32).reshape(-1)
    n = tempos.size

    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")

    out = np.zeros((n, n), dtype=np.float32)
    if n == 0:
        return out

    valid = np.isfinite(tempos) & (tempos > 0)
    if not np.any(valid):
        return out

    sigma = float(np.log1p(float(bandwidth)))
    xv = np.log(tempos[valid].astype(np.float64))

    d_main = np.abs(xv[:, None] - xv[None, :])
    if allow_octave:
        l2 = float(np.log(2.0))
        d_half = np.abs(xv[:, None] - (xv[None, :] - l2))
        d_double = np.abs(xv[:, None] - (xv[None, :] + l2))
        stack = np.stack([d_main, d_half, d_double], axis=0)
        choice = np.argmin(stack, axis=0)
        d = np.min(stack, axis=0)
        penalty = np.where(choice == 0, 1.0, float(octave_penalty))
    else:
        d = d_main
        penalty = np.ones_like(d)

    score = penalty * _tempo_distance_to_similarity_score(
        d,
        sigma=sigma,
        decay=float(decay),
        shape=tempo_similarity_shape,
        softflat_sharpness=float(softflat_sharpness),
    )

    if use_confidence:
        cv = np.clip(confs[valid].astype(np.float64), 0.0, 1.0)
        score *= np.sqrt(cv[:, None] * cv[None, :])

    score = np.clip(score, 0.0, 1.0).astype(np.float32)

    valid_idx = np.where(valid)[0]
    out[np.ix_(valid_idx, valid_idx)] = score
    return out


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-6)
    x = np.asarray(logits, dtype=np.float64) / t
    x = x - np.max(x)
    e = np.exp(x)
    z = np.sum(e)
    if z == 0.0:
        return np.zeros_like(x, dtype=np.float32)
    return (e / z).astype(np.float32)


def _load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")
    return np.load(path, allow_pickle=False)


def _filename_lookup(npz: np.lib.npyio.NpzFile, name: str) -> dict[str, int]:
    if name not in npz.files:
        raise KeyError(f"Missing '{name}' in NPZ. Keys: {sorted(npz.files)}")
    values = [str(v).strip().lower() for v in npz[name]]
    return {v: i for i, v in enumerate(values)}


def _default_npz_name(tracklist_csv: Path) -> str:
    stem = tracklist_csv.stem
    if stem.endswith("_tracks"):
        return f"{stem}.npz"
    return f"{stem}_tracks.npz"


def _resolve_collection_npz_path(
    *,
    path_or_dir: str | Path | None,
    default_dir: Path,
    npz_name: str,
) -> Path:
    if path_or_dir is None:
        return (default_dir / npz_name).expanduser().resolve()

    candidate = Path(path_or_dir).expanduser()
    # Heuristic: ".npz" means explicit file path; otherwise treat as directory.
    if candidate.suffix.lower() == ".npz":
        return candidate.resolve()
    return (candidate / npz_name).resolve()


def load_aries_mix_feature_set(
    mix_csv_path: str | Path = PROJECT_ROOT / "music" / "aries-mix" / "aries_mix_tracks.csv",
    maest_npz_path: str | Path | None = None,
    chroma_npz_path: str | Path | None = None,
    tempo_npz_path: str | Path | None = None,
    # Back-compat args (directory-style):
    maest_dir: str | Path | None = None,
    chroma_dir: str | Path | None = None,
    tempo_dir: str | Path | None = None,
    chroma_use_base_only: bool = True,
    # Back-compat placeholders (ignored now that DEAM is removed):
    deam_csv_path: str | Path | None = None,
    deam_normalize_to_unit: bool = True,
) -> SongFeatureSet:
    """Load aligned MAEST/chroma/tempo features for tracks in mix CSV order."""
    _ = deam_csv_path
    _ = deam_normalize_to_unit

    mix_csv = Path(mix_csv_path).expanduser().resolve()
    npz_name = _default_npz_name(mix_csv)

    maest_npz_file = _resolve_collection_npz_path(
        path_or_dir=maest_npz_path if maest_npz_path is not None else maest_dir,
        default_dir=PROJECT_ROOT / "data" / "maest_embeddings",
        npz_name=npz_name,
    )
    chroma_npz_file = _resolve_collection_npz_path(
        path_or_dir=chroma_npz_path if chroma_npz_path is not None else chroma_dir,
        default_dir=PROJECT_ROOT / "data" / "chroma_embeddings",
        npz_name=npz_name,
    )
    tempo_npz_file = _resolve_collection_npz_path(
        path_or_dir=tempo_npz_path if tempo_npz_path is not None else tempo_dir,
        default_dir=PROJECT_ROOT / "data" / "tempo_embeddings",
        npz_name=npz_name,
    )

    if not mix_csv.exists():
        raise FileNotFoundError(f"Mix CSV not found: {mix_csv}")

    maest_npz = _load_npz(maest_npz_file)
    chroma_npz = _load_npz(chroma_npz_file)
    tempo_npz = _load_npz(tempo_npz_file)

    maest_embeddings = np.asarray(maest_npz["embeddings"], dtype=np.float32)
    chroma_embeddings = np.asarray(chroma_npz["embeddings"], dtype=np.float32)

    if "tempo_bpm" in tempo_npz.files:
        tempo_bpm_all = np.asarray(tempo_npz["tempo_bpm"], dtype=np.float32)
    else:
        tempo_embeddings = np.asarray(tempo_npz["embeddings"], dtype=np.float32)
        if tempo_embeddings.shape[1] < 1:
            raise ValueError("Tempo NPZ embeddings must include tempo in column 0.")
        tempo_bpm_all = tempo_embeddings[:, 0].astype(np.float32)

    if "tempo_confidence" in tempo_npz.files:
        tempo_conf_all = np.asarray(tempo_npz["tempo_confidence"], dtype=np.float32)
    else:
        tempo_embeddings = np.asarray(tempo_npz["embeddings"], dtype=np.float32)
        if tempo_embeddings.shape[1] < 2:
            tempo_conf_all = np.ones_like(tempo_bpm_all, dtype=np.float32)
        else:
            tempo_conf_all = np.asarray(tempo_embeddings[:, 1], dtype=np.float32)

    maest_lookup = _filename_lookup(maest_npz, "filenames")
    chroma_lookup = _filename_lookup(chroma_npz, "filenames")
    tempo_lookup = _filename_lookup(tempo_npz, "filenames")

    chroma_base_dims = None
    if "chroma_base_embedding_dimension" in chroma_npz.files:
        chroma_base_dims = np.asarray(chroma_npz["chroma_base_embedding_dimension"], dtype=np.int32)

    chroma_pitch_means = None
    if "chroma_pitch_class_mean" in chroma_npz.files:
        pitch_means = np.asarray(chroma_npz["chroma_pitch_class_mean"], dtype=np.float32)
        if pitch_means.ndim == 2 and pitch_means.shape[1] >= 12:
            chroma_pitch_means = pitch_means[:, :12]

    chroma_center_baseline: float | None = None
    if "config_center_baseline" in chroma_npz.files:
        baseline_raw = np.asarray(chroma_npz["config_center_baseline"], dtype=np.float32).reshape(-1)
        if baseline_raw.size > 0 and np.isfinite(float(baseline_raw[0])):
            chroma_center_baseline = float(baseline_raw[0])
    if chroma_pitch_means is not None and chroma_center_baseline is None:
        mean_sum = float(np.nanmean(np.sum(chroma_pitch_means, axis=1)))
        if np.isfinite(mean_sum) and abs(mean_sum) < 1e-3:
            chroma_center_baseline = 1.0 / 12.0

    metadata: list[SongMetadata] = []
    maest_vectors: list[np.ndarray] = []
    chroma_vectors: list[np.ndarray] = []
    chroma_pitch_vectors: list[np.ndarray] = []
    tempo_bpms: list[float] = []
    tempo_confs: list[float] = []

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

            key = filename.lower()
            if key not in maest_lookup or key not in chroma_lookup or key not in tempo_lookup:
                continue

            mi = maest_lookup[key]
            ci = chroma_lookup[key]
            ti = tempo_lookup[key]

            maest_vec = np.asarray(maest_embeddings[mi], dtype=np.float32).reshape(-1)
            chroma_full = np.asarray(chroma_embeddings[ci], dtype=np.float32).reshape(-1)
            if maest_vec.size == 0 or chroma_full.size == 0:
                continue

            if chroma_use_base_only:
                if chroma_base_dims is not None and ci < chroma_base_dims.size:
                    base_dim = int(chroma_base_dims[ci])
                else:
                    base_dim = 24
                if base_dim <= 0 or base_dim > chroma_full.size:
                    continue
                chroma_vec = chroma_full[:base_dim]
            else:
                chroma_vec = chroma_full

            if chroma_pitch_means is not None and ci < chroma_pitch_means.shape[0]:
                chroma_pitch_vec = np.asarray(chroma_pitch_means[ci], dtype=np.float32).reshape(-1)
                if chroma_center_baseline is not None:
                    chroma_pitch_vec = chroma_pitch_vec + float(chroma_center_baseline)
            elif chroma_full.size >= 12:
                chroma_pitch_vec = np.asarray(chroma_full[:12], dtype=np.float32).reshape(-1)
            else:
                continue

            if chroma_pitch_vec.size != 12:
                continue
            chroma_pitch_vec = np.clip(chroma_pitch_vec, 0.0, None)
            if float(np.sum(chroma_pitch_vec)) == 0.0:
                continue

            tempo_bpm = float(tempo_bpm_all[ti]) if ti < tempo_bpm_all.size else float("nan")
            tempo_conf = float(tempo_conf_all[ti]) if ti < tempo_conf_all.size else 1.0

            metadata.append(
                SongMetadata(
                    track_number=track_number,
                    title=title if title else Path(filename).stem,
                    artist=artist,
                    filename=filename,
                    genre=genre,
                )
            )
            maest_vectors.append(maest_vec)
            chroma_vectors.append(chroma_vec)
            chroma_pitch_vectors.append(chroma_pitch_vec)
            tempo_bpms.append(tempo_bpm)
            tempo_confs.append(tempo_conf)

    if not metadata:
        raise RuntimeError("No tracks had all three embeddings (MAEST/chroma/tempo) available.")

    order = np.argsort([m.track_number for m in metadata])
    metadata_sorted = [metadata[i] for i in order]
    maest_arr = np.vstack([maest_vectors[i] for i in order]).astype(np.float32)
    chroma_arr = np.vstack([chroma_vectors[i] for i in order]).astype(np.float32)
    chroma_pitch_arr = np.vstack([chroma_pitch_vectors[i] for i in order]).astype(np.float32)
    tempo_bpm_arr = np.asarray([tempo_bpms[i] for i in order], dtype=np.float32)
    tempo_conf_arr = np.asarray([tempo_confs[i] for i in order], dtype=np.float32)

    return SongFeatureSet(
        metadata=metadata_sorted,
        maest=maest_arr,
        chroma=chroma_arr,
        chroma_pitch=chroma_pitch_arr,
        tempo_bpm=tempo_bpm_arr,
        tempo_confidence=tempo_conf_arr,
    )


def compatible_song_distribution(
    features: SongFeatureSet,
    seed_track_number: int,
    maest_weight: float = 0.5,
    chroma_weight: float = 0.3,
    harmonic_exact_weight: float = 1.0,
    harmonic_first_fifth_weight: float = 0.4,
    harmonic_second_fifth_weight: float = 0.15,
    harmonic_other_weight: float = 0.0,
    harmonic_self_normalize: bool = False,
    tempo_weight: float | None = None,
    temperature: float = 0.08,
    candidate_top_k: int | None = None,
    tempo_bandwidth: float = 0.06,
    tempo_decay: float = 0.5,
    tempo_allow_octave: bool = True,
    tempo_octave_penalty: float = 0.5,
    tempo_similarity_shape: str = "gaussian",
    tempo_softflat_sharpness: float = 8.0,
    tempo_use_confidence: bool = True,
    # Back-compat alias; if provided and tempo_weight is None, it is reused.
    deam_weight: float | None = None,
    deam_sigma: float | None = None,
) -> list[dict]:
    """Return ranked compatible-song distribution from a seed track.

    Compatibility logit is a weighted combination of:
    - MAEST cosine similarity (mapped from [-1,1] to [0,1])
    - Chroma fifth-aware similarity over 12-bin pitch-class profiles
      using tunable circle-of-fifths kernel weights
      with optional self-normalization `S_ij / sqrt(S_ii * S_jj)`
    - Tempo similarity (log-ratio kernel with optional half/double-tempo handling)
    """
    _ = deam_sigma

    if tempo_weight is None:
        tempo_weight = float(deam_weight) if deam_weight is not None else 0.2

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

    harmonic_kernel = _build_harmonic_kernel(
        exact_weight=float(harmonic_exact_weight),
        first_fifth_weight=float(harmonic_first_fifth_weight),
        second_fifth_weight=float(harmonic_second_fifth_weight),
        other_weight=float(harmonic_other_weight),
    )

    maest_cos = _cosine_similarity_vector_to_matrix(features.maest[seed_idx], features.maest)
    chroma_sim = _fifth_aware_similarity_vector_to_matrix(
        features.chroma_pitch[seed_idx],
        features.chroma_pitch,
        kernel=harmonic_kernel,
        normalize_by_self=bool(harmonic_self_normalize),
    )
    tempo_sim = _tempo_similarity_vector_to_matrix(
        float(features.tempo_bpm[seed_idx]),
        float(features.tempo_confidence[seed_idx]),
        features.tempo_bpm,
        features.tempo_confidence,
        bandwidth=float(tempo_bandwidth),
        decay=float(tempo_decay),
        allow_octave=bool(tempo_allow_octave),
        octave_penalty=float(tempo_octave_penalty),
        tempo_similarity_shape=str(tempo_similarity_shape),
        softflat_sharpness=float(tempo_softflat_sharpness),
        use_confidence=bool(tempo_use_confidence),
    )

    maest_sim = 0.5 * (maest_cos + 1.0)
    logits = (
        float(maest_weight) * maest_sim
        + float(chroma_weight) * chroma_sim
        + float(tempo_weight) * tempo_sim
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
                "tempo_similarity": float(tempo_sim[idx]),
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
    harmonic_exact_weight: float = 1.0,
    harmonic_first_fifth_weight: float = 0.4,
    harmonic_second_fifth_weight: float = 0.15,
    harmonic_other_weight: float = 0.0,
    harmonic_self_normalize: bool = False,
    tempo_weight: float | None = None,
    tempo_bandwidth: float = 0.06,
    tempo_decay: float = 0.5,
    tempo_allow_octave: bool = True,
    tempo_octave_penalty: float = 0.5,
    tempo_similarity_shape: str = "gaussian",
    tempo_softflat_sharpness: float = 8.0,
    tempo_use_confidence: bool = True,
    # Back-compat alias; if provided and tempo_weight is None, it is reused.
    deam_weight: float | None = None,
    deam_sigma: float | None = None,
) -> np.ndarray:
    """Pairwise weighted compatibility matrix across all songs."""
    _ = deam_sigma

    if tempo_weight is None:
        tempo_weight = float(deam_weight) if deam_weight is not None else 0.2

    harmonic_kernel = _build_harmonic_kernel(
        exact_weight=float(harmonic_exact_weight),
        first_fifth_weight=float(harmonic_first_fifth_weight),
        second_fifth_weight=float(harmonic_second_fifth_weight),
        other_weight=float(harmonic_other_weight),
    )

    maest_cos = _pairwise_cosine_similarity_matrix(features.maest)
    chroma_sim = _pairwise_fifth_aware_similarity_matrix(
        features.chroma_pitch,
        kernel=harmonic_kernel,
        normalize_by_self=bool(harmonic_self_normalize),
    )
    tempo_sim = _pairwise_tempo_similarity_matrix(
        features.tempo_bpm,
        features.tempo_confidence,
        bandwidth=float(tempo_bandwidth),
        decay=float(tempo_decay),
        allow_octave=bool(tempo_allow_octave),
        octave_penalty=float(tempo_octave_penalty),
        tempo_similarity_shape=str(tempo_similarity_shape),
        softflat_sharpness=float(tempo_softflat_sharpness),
        use_confidence=bool(tempo_use_confidence),
    )

    maest_sim = 0.5 * (maest_cos + 1.0)
    return (
        float(maest_weight) * maest_sim
        + float(chroma_weight) * chroma_sim
        + float(tempo_weight) * tempo_sim
    ).astype(np.float32)
