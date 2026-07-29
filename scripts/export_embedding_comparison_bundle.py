#!/usr/bin/env python3
"""Export raw Aries/Ara inputs for reducer comparisons.

The bundle deliberately exposes three reducer components only:
Style, Rhythm, and Harmony. Rhythm is the current combined tempo/groove
component used by the canonical interactive exporter.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from djprojectexploration.interactive_pacmap_knn_simplex import (
    _component_matrices_style_rhythm_harmony,
    _load_combined_groove_embeddings,
)
from djprojectexploration.interactive_visualization_common import (
    PROJECT_ROOT,
    _load_combined_records_and_features,
    resolve_tracklist_sources,
)
from djprojectexploration.pacmap_settings import TransitionScoringSettings
from djprojectexploration.transition_scoring import blend_component_distances


def _knn(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.asarray(D, dtype=np.float32)
    n = distances.shape[0]
    effective_k = min(max(1, int(k)), n - 1)
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    indices = np.argsort(masked, axis=1, kind="stable")[:, :effective_k].astype(np.int32)
    values = np.take_along_axis(distances, indices, axis=1).astype(np.float32)
    adjacency = np.zeros((n, n), dtype=np.uint8)
    rows = np.repeat(np.arange(n), effective_k)
    adjacency[rows, indices.reshape(-1)] = 1
    return indices, values, adjacency


def _write_tracks(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "idx", "global_track_number", "mix_slug", "track_number", "title",
        "artists", "genre", "raw_genre", "key", "csv_bpm", "est_bpm",
        "est_conf", "filename",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def export_bundle(
    *,
    output_dir: Path,
    mix_slugs: list[str] | None = None,
    project_root: Path = PROJECT_ROOT,
    k: int = 15,
) -> Path:
    project_root = project_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mix_slugs = list(mix_slugs or ["aries-mix", "ara-mix"])
    sources = resolve_tracklist_sources(
        project_root=project_root,
        mix_slugs=mix_slugs,
    )

    records, features = _load_combined_records_and_features(
        project_root=project_root,
        tracklist_paths=[path for _, path in sources],
        maest_dir=project_root / "data" / "maest_embeddings",
        chroma_dir=project_root / "data" / "chroma_embeddings",
        tempo_dir=project_root / "data" / "tempo_embeddings",
        html_output_dir=output_dir,
        snippet_seconds=8.0,
        snippet_middle_fraction=0.66,
        snippet_hop_seconds=0.25,
        snippet_cache_overwrite=False,
    )
    groove = _load_combined_groove_embeddings(
        project_root=project_root,
        tracklist_paths=[path for _, path in sources],
        groove_dir=project_root / "data" / "groove_embeddings",
    )
    scoring = TransitionScoringSettings().validate()
    matrices = _component_matrices_style_rhythm_harmony(
        features,
        groove_embeddings=groove,
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
        rhythm_tempo_weight=scoring.rhythm_tempo_weight,
    )

    component_distances = {
        "style": matrices["maest_distance"],
        "rhythm": matrices["rhythm_distance"],
        "harmony": matrices["chroma_distance"],
    }
    component_similarities = {
        "style": matrices["maest_similarity"],
        "rhythm": matrices["rhythm_similarity"],
        "harmony": matrices["chroma_similarity"],
    }
    default_distance = blend_component_distances(
        component_distances["style"],
        component_distances["rhythm"],
        component_distances["harmony"],
        style_weight=scoring.style_weight,
        rhythm_weight=scoring.rhythm_weight,
        harmony_weight=scoring.harmony_weight,
    )

    raw_features_path = output_dir / "raw_features.npz"
    np.savez_compressed(
        raw_features_path,
        style_maest=np.asarray(features.maest, dtype=np.float32),
        rhythm_tempo_bpm=np.asarray(features.tempo_bpm, dtype=np.float32),
        rhythm_tempo_confidence=np.asarray(features.tempo_confidence, dtype=np.float32),
        rhythm_groove=np.asarray(groove, dtype=np.float32),
        harmony_pitch_class=np.asarray(features.chroma_pitch, dtype=np.float32),
        harmony_chroma=np.asarray(features.chroma, dtype=np.float32),
    )
    np.savez_compressed(
        output_dir / "component_matrices.npz",
        **{f"{name}_similarity": value for name, value in component_similarities.items()},
        **{f"{name}_distance": value for name, value in component_distances.items()},
    )
    np.savez_compressed(
        output_dir / "combined_distances.npz",
        default_style_rhythm_harmony_distance=default_distance,
    )

    neighbor_arrays: dict[str, np.ndarray] = {}
    for name, distance in {**component_distances, "default": default_distance}.items():
        indices, values, adjacency = _knn(distance, k)
        neighbor_arrays[f"{name}_indices"] = indices
        neighbor_arrays[f"{name}_distances"] = values
        neighbor_arrays[f"{name}_adjacency"] = adjacency
    np.savez_compressed(output_dir / "nearest_neighbors.npz", **neighbor_arrays)

    _write_tracks(output_dir / "tracks.csv", records)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": " / ".join(mix_slugs),
        "mix_slugs": mix_slugs,
        "track_count": len(records),
        "row_order": "tracks.csv order; idx is the zero-based row/column index in every matrix",
        "components": {
            "style": "MAEST row embeddings; cosine similarity remapped from [-1, 1] to [0, 1]",
            "rhythm": "0.70 * tempo similarity + 0.30 * groove similarity",
            "harmony": "Fifth-aware pitch-class compatibility using exact-key-only kernel",
        },
        "reducer_components": ["style", "rhythm", "harmony"],
        "similarity_files": "component_matrices.npz contains only the three reducer components",
        "distance_definition": (
            "Style and Harmony distances are 1 - similarity, normalized by their finite "
            "off-diagonal 95th percentile and clipped to [0, 1]. Rhythm distance is the "
            "0.70/0.30 blend of the separately normalized tempo and groove distances; use "
            "rhythm_distance for reducer input. rhythm_similarity is the raw combined score "
            "retained for reporting."
        ),
        "default_weights": scoring.to_dict(),
        "default_distance": (
            "0.50 * style_distance + 0.30 * rhythm_distance + 0.20 * harmony_distance; "
            "symmetric with a zero diagonal"
        ),
        "nearest_neighbors": {
            "k_requested": int(k),
            "file": "nearest_neighbors.npz",
            "arrays": "<component>_indices, <component>_distances, and <component>_adjacency",
        },
        "rhythm_parameters": {
            "tempo_bandwidth": 0.06,
            "tempo_decay": 0.5,
            "tempo_allow_octave": True,
            "tempo_octave_penalty": 0.5,
            "tempo_similarity_shape": "gaussian",
            "tempo_softflat_sharpness": 8.0,
            "tempo_use_confidence": False,
            "tempo_weight_within_rhythm": scoring.rhythm_tempo_weight,
            "groove_weight_within_rhythm": 1.0 - scoring.rhythm_tempo_weight,
        },
        "harmony_parameters": {
            "exact_weight": 1.0,
            "first_fifth_weight": 0.0,
            "second_fifth_weight": 0.0,
            "other_weight": 0.0,
            "normalize_by_self": True,
        },
        "files": {
            "tracks": "tracks.csv",
            "raw_features": "raw_features.npz",
            "component_matrices": "component_matrices.npz",
            "combined_distances": "combined_distances.npz",
            "nearest_neighbors": "nearest_neighbors.npz",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Aries/Ara reducer-comparison bundle\n\n"
        "Use `tracks.csv` to map matrix rows to tracks. The three component pairs in "
        "`component_matrices.npz` are the canonical Style/Rhythm/Harmony inputs. "
        "`raw_features.npz` contains the aligned features needed to recompute them; "
        "see `metadata.json` for the exact current parameters.\n\n"
        "The default 0.50/0.30/0.20 combined distance is in `combined_distances.npz`. "
        "`nearest_neighbors.npz` contains row-wise KNN indices, distances, and binary "
        "adjacency matrices for each component plus the default combined distance.\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mix-slug", action="append", dest="mix_slugs", default=None)
    parser.add_argument("--k", type=int, default=15, help="Number of nearest neighbors to export.")
    args = parser.parse_args()
    path = export_bundle(output_dir=args.output_dir, mix_slugs=args.mix_slugs, k=args.k)
    print(f"Wrote comparison bundle: {path}")


if __name__ == "__main__":
    main()
