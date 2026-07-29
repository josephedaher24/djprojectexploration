#!/usr/bin/env python3
"""Create the old standalone-pacmap data layout with current matrices.

Legacy key names are retained for the existing standalone loader:
genre_* -> Style, tempo_* -> combined Rhythm, key_* -> Harmony.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def export_compatibility_bundle(*, source_dir: Path, output_dir: Path) -> Path:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    with np.load(source_dir / "component_matrices.npz") as source:
        legacy_matrices = {
            "genre_similarity": source["style_similarity"],
            "tempo_similarity": source["rhythm_similarity"],
            "key_similarity": source["harmony_similarity"],
            "genre_distance": source["style_distance"],
            "tempo_distance": source["rhythm_distance"],
            "key_distance": source["harmony_distance"],
        }
    np.savez_compressed(data_dir / "component_matrices.npz", **legacy_matrices)

    with (source_dir / "tracks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    source_metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    tracks = []
    for row in rows:
        tracks.append(
            {
                "idx": int(row["idx"]),
                "global_track_number": int(row["global_track_number"]),
                "track_number": row["track_number"],
                "mix_slug": row["mix_slug"],
                "title": row["title"],
                "artists": row["artists"],
                "genre": row["raw_genre"],
                "key": row["key"],
                "csv_bpm": row["csv_bpm"],
                "est_bpm": float(row["est_bpm"]),
                "est_conf": float(row["est_conf"]),
                "filename": row["filename"],
                "snippet_uri": "",
                "snippet_start": 0.0,
                "snippet_end": 0.0,
                "snippet_rms": 0.0,
            }
        )
    (data_dir / "tracks.json").write_text(
        json.dumps(tracks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "dataset": source_metadata.get("dataset", "comparison dataset"),
        "mix_slugs": source_metadata.get("mix_slugs", []),
        "row_order": "tracks.json order; record idx equals matrix row/column index",
        "components": {
            "genre": "Legacy key name for Style: MAEST cosine similarity remapped to [0, 1]",
            "tempo": "Legacy key name for combined Rhythm: 0.70 * tempo + 0.30 * groove",
            "key": "Legacy key name for Harmony: fifth-aware pitch-class compatibility",
        },
        "distance_normalization": (
            "Style and Harmony use distance = 1 - similarity, then each component distance "
            "is scaled by its finite off-diagonal 95th percentile and clipped to [0, 1]. "
            "Rhythm distance blends the separately normalized tempo and groove distances."
        ),
        "combined_distance": (
            "w_genre * genre_distance + w_tempo * tempo_distance + w_key * key_distance, "
            "after weights are normalized to sum to 1; legacy names map to Style/Rhythm/Harmony"
        ),
        "prototype_settings": {
            "tempo_bandwidth": 0.06,
            "tempo_decay": 0.5,
            "tempo_allow_octave": True,
            "tempo_octave_penalty": 0.5,
            "tempo_similarity_shape": "gaussian",
            "tempo_softflat_sharpness": 8.0,
            "tempo_use_confidence": False,
            "rhythm_tempo_weight": 0.7,
            "rhythm_groove_weight": 0.3,
            "harmonic_exact_weight": 1.0,
            "harmonic_first_fifth_weight": 0.0,
            "harmonic_second_fifth_weight": 0.0,
            "harmonic_other_weight": 0.0,
            "harmonic_self_normalize": True,
        },
        "default_weights": {
            "style_weight": 0.5,
            "rhythm_weight": 0.3,
            "harmony_weight": 0.2,
        },
        "track_count": len(tracks),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_mapping": {
            "genre_*": "style_*",
            "tempo_*": "rhythm_*",
            "key_*": "harmony_*",
        },
    }
    (data_dir / "matrix_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        f"# {source_metadata.get('dataset', 'Dataset')} legacy PaCMAP compatibility export\n\n"
        "Copy or unzip the `data/` directory into the old `pacmap` folder. "
        "The NPZ keys retain the original names: `genre_*` means Style, "
        "`tempo_*` means the current combined Rhythm (70% tempo + 30% groove), "
        "and `key_*` means Harmony.\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(export_compatibility_bundle(source_dir=args.source_dir, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
