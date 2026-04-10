"""Chroma embedding extractor using Essentia HPCP/chroma pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from djprojectexploration.chroma_embedding import generate_chroma_embedding


DEFAULT_OUTPUT_FILENAME = "chroma_embedding.json"


def _to_project_relpath(path: Path, project_root: Path) -> str:
    """Return path relative to project_root when possible, else absolute path."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


def _first_mp3(music_dir: Path) -> Path | None:
    mp3_files = sorted(music_dir.glob("*.mp3"))
    return mp3_files[0] if mp3_files else None


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_music_dir = project_root / "music"
    default_output_file = default_music_dir / DEFAULT_OUTPUT_FILENAME

    parser = argparse.ArgumentParser(description="Extract beat-synchronous chroma embeddings.")
    parser.add_argument(
        "--audio-file",
        type=Path,
        default=None,
        help="Path to an input audio file. If omitted, the first .mp3 in --music-dir is used.",
    )
    parser.add_argument(
        "--music-dir",
        type=Path,
        default=default_music_dir,
        help="Directory searched for .mp3 files when --audio-file is not provided.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=default_output_file,
        help="Output JSON file path.",
    )
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--frame-size", type=int, default=4096)
    parser.add_argument("--hop-size", type=int, default=1024)
    parser.add_argument("--chroma-bins", type=int, default=12)
    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Optional manual BPM for beat-grid construction.",
    )
    parser.add_argument(
        "--onset-time-ms",
        type=float,
        default=None,
        help="Optional onset anchor in milliseconds (requires --bpm).",
    )
    parser.add_argument(
        "--exclude-key-features",
        action="store_true",
        help="Disable appended key features (default is enabled).",
    )
    parser.add_argument(
        "--center-baseline",
        type=float,
        default=None,
        help=(
            "Optional baseline subtracted from each unit-sum pitch-class bin before pooling. "
            "Default: no centering."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    if args.audio_file is None:
        audio_file = _first_mp3(args.music_dir)
        if audio_file is None:
            raise SystemExit(f"No .mp3 files found in: {args.music_dir}")
    else:
        audio_file = args.audio_file

    audio_file = audio_file.expanduser().resolve()
    if not audio_file.exists():
        raise SystemExit(f"Audio file not found: {audio_file}")

    payload = generate_chroma_embedding(
        audio_file=audio_file,
        sample_rate=int(args.sample_rate),
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        chroma_bins=int(args.chroma_bins),
        bpm=args.bpm,
        onset_time_ms=args.onset_time_ms,
        include_key_features=not bool(args.exclude_key_features),
        center_baseline=args.center_baseline,
    )

    output_file = args.output_file.expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print(f"Audio file: {_to_project_relpath(audio_file, project_root)}")
    print(f"Embedding type: {payload['embedding_type']}")
    print(f"Embedding dimension: {payload['embedding_dimension']}")
    print(f"Center baseline: {payload['config']['center_baseline']}")
    print(f"Saved chroma embedding JSON: {_to_project_relpath(output_file.resolve(), project_root)}")


if __name__ == "__main__":
    main()
