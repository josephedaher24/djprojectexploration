"""Generate a synthetic Aries feature set with fixed genre embeddings.

Each source Aries track is duplicated with random key transpositions and random
log-tempo scaling. MAEST genre embeddings are copied unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SLUG = "aries-mix"
OUTPUT_SLUG = "aries-mix-synthetic"
OUTPUT_STEM = "aries_mix_synthetic_tracks"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=False)
    return {key: raw[key] for key in raw.files}


def _save_npz(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def _repeat_array(values: np.ndarray, source_indices: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        return arr
    if arr.shape[0] != source_indices.max(initial=-1) + 1:
        return arr
    return arr[source_indices]


def _camelot_shift(key: str, fifth_steps: int) -> str:
    text = str(key or "").strip().upper()
    if len(text) < 2 or text[-1] not in {"A", "B"}:
        return text
    try:
        number = int(text[:-1])
    except ValueError:
        return text
    shifted = ((number - 1 + int(fifth_steps)) % 12) + 1
    return f"{shifted}{text[-1]}"


def _format_float(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def _build_synthetic_plan(
    rows: list[dict[str, str]],
    *,
    variants_per_track: int,
    seed: int,
    min_tempo_factor: float,
    max_tempo_factor: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    source_indices: list[int] = []
    fifth_steps: list[int] = []
    tempo_factors: list[float] = []
    out_rows: list[dict[str, Any]] = []

    for src_idx, row in enumerate(rows):
        original_filename = (row.get("mp3_name") or "").strip()
        original_stem = Path(original_filename).stem or f"track_{src_idx + 1:02d}"
        original_bpm = float(row.get("bpm") or 0.0)
        for variant_idx in range(1, variants_per_track + 1):
            fifth_shift = int(((variant_idx - 1) % 11) + 1)
            semitone_shift = int((7 * fifth_shift) % 12)
            factor = float(np.exp(rng.uniform(np.log(min_tempo_factor), np.log(max_tempo_factor))))
            synthetic_filename = f"{original_stem}__syn{variant_idx:02d}.mp3"
            source_indices.append(src_idx)
            fifth_steps.append(fifth_shift)
            tempo_factors.append(factor)
            out_rows.append(
                {
                    "track_number": len(out_rows) + 1,
                    "title": f"{row.get('title', '').strip()} [synthetic {variant_idx:02d}]",
                    "artists": row.get("artists", "").strip(),
                    "mp3_name": synthetic_filename,
                    "key": _camelot_shift(row.get("key", ""), fifth_shift),
                    "bpm": _format_float(original_bpm * factor, 3),
                    "onset-time": row.get("onset-time", ""),
                    "genre": row.get("genre", ""),
                    "key shift": semitone_shift,
                    "energy": row.get("energy", ""),
                    "source_track_number": row.get("track_number", ""),
                    "source_title": row.get("title", ""),
                    "source_artists": row.get("artists", ""),
                    "source_mp3_name": original_filename,
                    "source_key": row.get("key", ""),
                    "source_bpm": row.get("bpm", ""),
                    "synthetic_variant": variant_idx,
                    "synthetic_key_fifths_shift": fifth_shift,
                    "synthetic_key_semitone_shift": semitone_shift,
                    "synthetic_tempo_factor": _format_float(factor, 8),
                    "synthetic_log2_tempo_shift": _format_float(np.log2(factor), 8),
                    "synthetic_genre_embedding": "fixed_source_maest",
                }
            )

    return (
        out_rows,
        np.asarray(source_indices, dtype=np.int32),
        np.asarray(fifth_steps, dtype=np.int32),
        np.asarray(tempo_factors, dtype=np.float32),
    )


def _roll_pitch_classes(values: np.ndarray, semitone_shifts: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.empty((len(semitone_shifts), arr.shape[1]), dtype=np.float32)
    for i, shift in enumerate(semitone_shifts):
        out[i] = np.roll(arr[i], int(shift))
    return out


def _synthetic_common_fields(
    source_npz: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    source_indices: np.ndarray,
    *,
    embedding_type: str,
    embedding_dimension: int,
) -> dict[str, Any]:
    return {
        "embedding_type": np.asarray(embedding_type),
        "playlist_csv": np.asarray(f"music/{OUTPUT_SLUG}/{OUTPUT_STEM}.csv"),
        "created_utc": np.asarray(datetime.now(UTC).isoformat()),
        "num_tracks": np.asarray(len(rows), dtype=np.int32),
        "embedding_dimension": np.asarray(embedding_dimension, dtype=np.int32),
        "track_numbers": np.asarray([int(r["track_number"]) for r in rows], dtype=np.int32),
        "titles": np.asarray([str(r["title"]) for r in rows]),
        "artists": np.asarray([str(r["artists"]) for r in rows]),
        "filenames": np.asarray([str(r["mp3_name"]) for r in rows]),
        "audio_paths": np.asarray([f"music/{SOURCE_SLUG}/{r['source_mp3_name']}" for r in rows]),
        "genres": np.asarray([str(r["genre"]) for r in rows]),
        "keys": np.asarray([str(r["key"]) for r in rows]),
        "bpm": np.asarray([float(r["bpm"]) for r in rows], dtype=np.float32),
        "onset_time": _repeat_array(source_npz["onset_time"], source_indices).astype(np.float32),
        "key_shift": np.asarray([int(r["synthetic_key_semitone_shift"]) for r in rows], dtype=np.float32),
        "source_indices": source_indices.astype(np.int32),
        "synthetic_key_fifths_shift": np.asarray([int(r["synthetic_key_fifths_shift"]) for r in rows], dtype=np.int32),
        "synthetic_key_semitone_shift": np.asarray([int(r["synthetic_key_semitone_shift"]) for r in rows], dtype=np.int32),
        "synthetic_tempo_factor": np.asarray([float(r["synthetic_tempo_factor"]) for r in rows], dtype=np.float32),
        "synthetic_log2_tempo_shift": np.asarray([float(r["synthetic_log2_tempo_shift"]) for r in rows], dtype=np.float32),
    }


def _build_maest_npz(source: dict[str, np.ndarray], rows: list[dict[str, Any]], source_indices: np.ndarray) -> dict[str, Any]:
    embeddings = np.asarray(source["embeddings"], dtype=np.float32)[source_indices]
    out = _synthetic_common_fields(source, rows, source_indices, embedding_type="maest", embedding_dimension=embeddings.shape[1])
    out["embeddings"] = embeddings
    for key in ("maest_model_file", "maest_output_node"):
        if key in source:
            out[key] = source[key]
    for key in ("maest_reductions", "maest_raw_prediction_shapes"):
        if key in source:
            out[key] = _repeat_array(source[key], source_indices)
    return out


def _build_chroma_npz(
    source: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    source_indices: np.ndarray,
) -> dict[str, Any]:
    semitone_shifts = np.asarray([int(r["synthetic_key_semitone_shift"]) for r in rows], dtype=np.int32)
    embeddings = np.asarray(source["embeddings"], dtype=np.float32)[source_indices].copy()
    pitch_mean = np.asarray(source["chroma_pitch_class_mean"], dtype=np.float32)[source_indices]
    pitch_std = np.asarray(source["chroma_pitch_class_std"], dtype=np.float32)[source_indices]
    rolled_mean = _roll_pitch_classes(pitch_mean, semitone_shifts)
    rolled_std = _roll_pitch_classes(pitch_std, semitone_shifts)
    embeddings[:, :12] = rolled_mean
    embeddings[:, 12:24] = rolled_std

    out = _synthetic_common_fields(source, rows, source_indices, embedding_type="chroma", embedding_dimension=embeddings.shape[1])
    out["embeddings"] = embeddings
    for key in (
        "chroma_embedding_subtype",
        "chroma_base_embedding_dimension",
        "chroma_key_feature_dimension",
        "chroma_bins",
        "detected_scale",
        "detected_key_strength",
        "beat_count",
        "beat_source",
        "beat_phase_anchor_seconds",
    ):
        if key in source:
            out[key] = _repeat_array(source[key], source_indices)
    out["chroma_pitch_class_mean"] = rolled_mean
    out["chroma_pitch_class_std"] = rolled_std
    out["detected_key"] = np.asarray([str(r["key"]) for r in rows])
    out["beat_bpm"] = np.asarray([float(r["bpm"]) for r in rows], dtype=np.float32)
    for key in ("config_sample_rate", "config_frame_size", "config_hop_size", "config_include_key_features", "config_center_baseline"):
        if key in source:
            out[key] = source[key]
    return out


def _build_tempo_npz(
    source: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    source_indices: np.ndarray,
    tempo_factors: np.ndarray,
) -> dict[str, Any]:
    source_bpm = np.asarray(source["tempo_bpm"], dtype=np.float32)[source_indices]
    tempo_bpm = (source_bpm * tempo_factors).astype(np.float32)
    tempo_conf = np.asarray(source["tempo_confidence"], dtype=np.float32)[source_indices]
    embeddings = np.column_stack([tempo_bpm, tempo_conf]).astype(np.float32)

    out = _synthetic_common_fields(source, rows, source_indices, embedding_type="tempo", embedding_dimension=2)
    out["embeddings"] = embeddings
    out["tempo_bpm"] = tempo_bpm
    out["tempo_confidence"] = tempo_conf
    for key in (
        "tempo_confidence_active_agreement",
        "tempo_mean_active_probability",
        "tempo_active_fraction",
        "tempo_active_windows",
        "tempo_total_windows",
    ):
        if key in source:
            out[key] = _repeat_array(source[key], source_indices)

    starts: list[int] = []
    counts: list[int] = []
    bpm_flat: list[np.ndarray] = []
    prob_flat: list[np.ndarray] = []
    times_flat: list[np.ndarray] = []
    active_flat: list[np.ndarray] = []
    source_starts = np.asarray(source.get("tempo_local_start_index", []), dtype=np.int64)
    source_counts = np.asarray(source.get("tempo_local_count", []), dtype=np.int32)
    if source_starts.size and source_counts.size:
        src_bpm_flat = np.asarray(source["tempo_local_bpm_flat"], dtype=np.float32)
        src_prob_flat = np.asarray(source["tempo_local_probability_flat"], dtype=np.float32)
        src_times_flat = np.asarray(source["tempo_local_times_sec_flat"], dtype=np.float32)
        src_active_flat = np.asarray(source["tempo_local_active_mask_flat"], dtype=np.int8)
        cursor = 0
        for src_idx, factor in zip(source_indices, tempo_factors, strict=True):
            start = int(source_starts[src_idx])
            count = int(source_counts[src_idx])
            starts.append(cursor)
            counts.append(count)
            end = start + count
            bpm_flat.append((src_bpm_flat[start:end] * float(factor)).astype(np.float32))
            prob_flat.append(src_prob_flat[start:end])
            times_flat.append(src_times_flat[start:end])
            active_flat.append(src_active_flat[start:end])
            cursor += count
        out["tempo_local_start_index"] = np.asarray(starts, dtype=np.int64)
        out["tempo_local_count"] = np.asarray(counts, dtype=np.int32)
        out["tempo_local_bpm_flat"] = np.concatenate(bpm_flat).astype(np.float32)
        out["tempo_local_probability_flat"] = np.concatenate(prob_flat).astype(np.float32)
        out["tempo_local_times_sec_flat"] = np.concatenate(times_flat).astype(np.float32)
        out["tempo_local_active_mask_flat"] = np.concatenate(active_flat).astype(np.int8)

    for key in (
        "tempo_model_file",
        "tempo_model_url",
        "config_sample_rate",
        "config_resample_quality",
        "config_snippet_length_sec",
        "config_window_sec",
        "config_hop_sec",
        "config_rms_percentile",
    ):
        if key in source:
            out[key] = source[key]
    return out


def generate_synthetic_aries(
    *,
    project_root: Path = PROJECT_ROOT,
    variants_per_track: int = 11,
    seed: int = 20260626,
    min_tempo_factor: float = 2.0**-0.5,
    max_tempo_factor: float = 2.0**0.5,
) -> dict[str, Path]:
    project_root = project_root.expanduser().resolve()
    source_csv = project_root / "music" / SOURCE_SLUG / "aries_mix_tracks.csv"
    output_csv = project_root / "music" / OUTPUT_SLUG / f"{OUTPUT_STEM}.csv"
    rows = _read_csv(source_csv)
    synthetic_rows, source_indices, _fifth_steps, tempo_factors = _build_synthetic_plan(
        rows,
        variants_per_track=variants_per_track,
        seed=seed,
        min_tempo_factor=min_tempo_factor,
        max_tempo_factor=max_tempo_factor,
    )
    fieldnames = [
        "track_number",
        "title",
        "artists",
        "mp3_name",
        "key",
        "bpm",
        "onset-time",
        "genre",
        "key shift",
        "energy",
        "source_track_number",
        "source_title",
        "source_artists",
        "source_mp3_name",
        "source_key",
        "source_bpm",
        "synthetic_variant",
        "synthetic_key_fifths_shift",
        "synthetic_key_semitone_shift",
        "synthetic_tempo_factor",
        "synthetic_log2_tempo_shift",
        "synthetic_genre_embedding",
    ]
    _write_csv(output_csv, synthetic_rows, fieldnames)

    maest_source = _load_npz(project_root / "data" / "maest_embeddings" / "aries_mix_tracks.npz")
    chroma_source = _load_npz(project_root / "data" / "chroma_embeddings" / "aries_mix_tracks.npz")
    tempo_source = _load_npz(project_root / "data" / "tempo_embeddings" / "aries_mix_tracks.npz")

    maest_out = project_root / "data" / "maest_embeddings" / f"{OUTPUT_STEM}.npz"
    chroma_out = project_root / "data" / "chroma_embeddings" / f"{OUTPUT_STEM}.npz"
    tempo_out = project_root / "data" / "tempo_embeddings" / f"{OUTPUT_STEM}.npz"
    _save_npz(maest_out, _build_maest_npz(maest_source, synthetic_rows, source_indices))
    _save_npz(chroma_out, _build_chroma_npz(chroma_source, synthetic_rows, source_indices))
    _save_npz(tempo_out, _build_tempo_npz(tempo_source, synthetic_rows, source_indices, tempo_factors))

    manifest = {
        "source_slug": SOURCE_SLUG,
        "output_slug": OUTPUT_SLUG,
        "source_csv": str(source_csv.relative_to(project_root)),
        "output_csv": str(output_csv.relative_to(project_root)),
        "variants_per_track": variants_per_track,
        "seed": seed,
        "tempo_factor_range": [min_tempo_factor, max_tempo_factor],
        "genre_embedding": "fixed copy of source MAEST embedding",
        "key_generation": "deterministic Camelot fifth shifts 1..11 per source track, preserving A/B mode; chroma pitch-class mean/std rolled by equivalent semitone shift",
        "tempo_generation": "log-uniform tempo factor in the configured range; source tempo BPM and local tempo BPM values multiplied by factor",
        "num_source_tracks": len(rows),
        "num_synthetic_tracks": len(synthetic_rows),
        "files": {
            "maest_npz": str(maest_out.relative_to(project_root)),
            "chroma_npz": str(chroma_out.relative_to(project_root)),
            "tempo_npz": str(tempo_out.relative_to(project_root)),
        },
    }
    manifest_path = project_root / "data" / "synthetic" / f"{OUTPUT_STEM}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "csv": output_csv,
        "maest": maest_out,
        "chroma": chroma_out,
        "tempo": tempo_out,
        "manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Aries key/tempo feature variants.")
    parser.add_argument("--variants-per-track", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--min-tempo-factor", type=float, default=2.0**-0.5)
    parser.add_argument("--max-tempo-factor", type=float, default=2.0**0.5)
    args = parser.parse_args()
    outputs = generate_synthetic_aries(
        variants_per_track=args.variants_per_track,
        seed=args.seed,
        min_tempo_factor=args.min_tempo_factor,
        max_tempo_factor=args.max_tempo_factor,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
