#!/usr/bin/env python3
"""Export component matrices for the standalone PaCMAP folder.

This script may import djprojectexploration modules. The generated bundle is
plain JSON/NPZ data that `pacmap/standalone_pacmap.py` can consume without
importing the main project.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from djprojectexploration.interactive_pacmap_knn_simplex import _component_matrices_3way
from djprojectexploration.interactive_visualization_common import (
    PROJECT_ROOT,
    _load_combined_records_and_features,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def export_bundle(
    *,
    mix_slugs: list[str],
    output_dir: Path,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    project_root = project_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, features = _load_combined_records_and_features(
        project_root=project_root,
        mix_slugs=mix_slugs,
        maest_dir=project_root / "data" / "maest_embeddings",
        chroma_dir=project_root / "data" / "chroma_embeddings",
        tempo_dir=project_root / "data" / "tempo_embeddings",
        html_output_dir=project_root / "pacmap",
        snippet_seconds=8.0,
        snippet_middle_fraction=0.66,
        snippet_hop_seconds=0.25,
        snippet_cache_overwrite=False,
    )

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

    standalone_matrices = {
        "genre_similarity": matrices["maest_similarity"],
        "tempo_similarity": matrices["tempo_similarity"],
        "key_similarity": matrices["chroma_similarity"],
        "genre_distance": matrices["maest_distance"],
        "tempo_distance": matrices["tempo_distance"],
        "key_distance": matrices["chroma_distance"],
    }

    standalone_records: list[dict[str, Any]] = []
    for record in records:
        standalone_records.append(
            {
                "idx": int(record["idx"]),
                "global_track_number": int(record["global_track_number"]),
                "track_number": str(record["track_number"]),
                "mix_slug": str(record["mix_slug"]),
                "title": str(record["title"]),
                "artists": str(record["artists"]),
                "genre": str(record["genre"]),
                "raw_genre": str(record.get("raw_genre", record["genre"])),
                "key": str(record["key"]),
                "csv_bpm": str(record["csv_bpm"]),
                "est_bpm": float(record["est_bpm"]),
                "est_conf": float(record["est_conf"]),
                "filename": str(record["filename"]),
                "snippet_uri": str(record["snippet_uri"]),
                "snippet_start": float(record["snippet_start"]),
                "snippet_end": float(record["snippet_end"]),
                "snippet_rms": float(record["snippet_rms"]),
            }
        )

    tracks_path = output_dir / "tracks.json"
    matrices_path = output_dir / "component_matrices.npz"
    metadata_path = output_dir / "matrix_metadata.json"

    tracks_path.write_text(
        json.dumps(standalone_records, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(matrices_path, **standalone_matrices)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mix_slugs": mix_slugs,
        "row_order": "tracks.json order; record idx equals matrix row/column index",
        "components": {
            "genre": "MAEST cosine similarity remapped to [0, 1]",
            "tempo": "Tempo compatibility similarity, octave-aware settings from current prototype",
            "key": "Chroma/key fifth-aware similarity with exact-key-only kernel in current prototype",
        },
        "distance_normalization": (
            "distance = 1 - similarity, then each component distance matrix is scaled by its own "
            "finite off-diagonal 95th percentile and clipped to [0, 1]"
        ),
        "standalone_combined_distance": (
            "pacmap/standalone_pacmap.py combines distances with L1 weighted average: "
            "w_genre * genre_distance + w_tempo * tempo_distance + w_key * key_distance, "
            "after weights are normalized to sum to 1"
        ),
        "prototype_settings": {
            "tempo_bandwidth": 0.06,
            "tempo_decay": 0.5,
            "tempo_allow_octave": True,
            "tempo_octave_penalty": 0.5,
            "tempo_similarity_shape": "gaussian",
            "tempo_softflat_sharpness": 8.0,
            "tempo_use_confidence": False,
            "harmonic_exact_weight": 1.0,
            "harmonic_first_fifth_weight": 0.0,
            "harmonic_second_fifth_weight": 0.0,
            "harmonic_other_weight": 0.0,
            "harmonic_self_normalize": True,
        },
        "track_count": len(standalone_records),
        "matrix_files": {
            "tracks": tracks_path.name,
            "matrices": matrices_path.name,
            "metadata": metadata_path.name,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote tracks: {tracks_path}")
    print(f"Wrote matrices: {matrices_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Track count: {len(standalone_records)}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export standalone PaCMAP matrix bundle.")
    parser.add_argument("--mix-slug", action="append", dest="mix_slugs", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_bundle(mix_slugs=args.mix_slugs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
