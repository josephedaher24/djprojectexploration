"""Tempo embedding extractor using Essentia TempoCNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from djprojectexploration.tempo_embedding import (
    DEFAULT_TEMPOCNN_MODEL_FILE,
    DEFAULT_TEMPOCNN_MODEL_URL,
    generate_tempo_embedding,
)


DEFAULT_OUTPUT_FILENAME = "tempo_embedding_tempocnn.json"


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

    parser = argparse.ArgumentParser(description="Extract TempoCNN tempo embeddings.")
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
        "--model-file",
        type=Path,
        default=DEFAULT_TEMPOCNN_MODEL_FILE,
        help="TempoCNN TensorFlow graph (.pb).",
    )
    parser.add_argument(
        "--auto-download-model",
        action="store_true",
        help=f"Download the default TempoCNN model from {DEFAULT_TEMPOCNN_MODEL_URL} if missing.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=default_output_file,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--snippet-length-sec",
        type=float,
        default=None,
        help="Optional max audio duration to analyze in seconds (full file by default).",
    )
    parser.add_argument("--sample-rate", type=int, default=11025)
    parser.add_argument("--resample-quality", type=int, default=4)
    parser.add_argument("--window-sec", type=float, default=12.0)
    parser.add_argument("--hop-sec", type=float, default=6.0)
    parser.add_argument("--rms-percentile", type=float, default=20.0)
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

    payload = generate_tempo_embedding(
        audio_file=audio_file,
        model_file=args.model_file,
        auto_download_model=bool(args.auto_download_model),
        sample_rate=int(args.sample_rate),
        resample_quality=int(args.resample_quality),
        snippet_length_sec=args.snippet_length_sec,
        window_sec=float(args.window_sec),
        hop_sec=float(args.hop_sec),
        rms_percentile=float(args.rms_percentile),
    )

    output_file = args.output_file.expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print(f"Audio file: {_to_project_relpath(audio_file, project_root)}")
    print(f"Tempo (BPM): {payload['tempo_bpm']:.2f}")
    print(f"Confidence [0-1]: {payload['confidence']:.3f}")
    print(f"Local windows: {payload['total_windows']}")
    print(f"Saved tempo embedding JSON: {_to_project_relpath(output_file.resolve(), project_root)}")


if __name__ == "__main__":
    main()

