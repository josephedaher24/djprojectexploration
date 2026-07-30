"""Groove embedding extractor using TempoCNN and multiband onset flux."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from djprojectexploration.groove_embedding import (
    DEFAULT_BEATS_PER_BAR,
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_PHRASE_BARS,
    DEFAULT_POOLING_MODE,
    DEFAULT_POOLING_TOPK,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SUBDIVISIONS_PER_BEAT,
    DEFAULT_TEMPOCNN_SAMPLE_RATE,
    generate_groove_embedding,
)
from djprojectexploration.tempo_embedding import DEFAULT_TEMPOCNN_MODEL_FILE, DEFAULT_TEMPOCNN_MODEL_URL


DEFAULT_OUTPUT_FILENAME = "groove_embedding.json"


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

    parser = argparse.ArgumentParser(description="Extract beat/phrase-synchronous groove embeddings.")
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
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--tempocnn-sample-rate", type=int, default=DEFAULT_TEMPOCNN_SAMPLE_RATE)
    parser.add_argument("--snippet-length-sec", type=float, default=None)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--manual-bpm", type=float, default=None)
    parser.add_argument(
        "--onset-time-sec",
        type=float,
        default=0.0,
        help="Optional beat-grid phase anchor in seconds.",
    )
    parser.add_argument(
        "--disable-phase-align",
        action="store_true",
        help="Disable onset-envelope phase alignment.",
    )
    parser.add_argument(
        "--phase-align-mode",
        choices=["full", "low_mid", "adaptive"],
        default="adaptive",
    )
    parser.add_argument("--phase-align-max-shift-sec", type=float, default=0.18)
    parser.add_argument("--phase-align-step-sec", type=float, default=0.002)
    parser.add_argument(
        "--disable-prepend-start-beats",
        action="store_true",
        help="Disable near-zero missing-start-beat repair.",
    )
    parser.add_argument("--subdivisions-per-beat", type=int, default=DEFAULT_SUBDIVISIONS_PER_BEAT)
    parser.add_argument("--beats-per-bar", type=int, default=DEFAULT_BEATS_PER_BAR)
    parser.add_argument("--phrase-bars", type=int, default=DEFAULT_PHRASE_BARS)
    parser.add_argument(
        "--profile-mode",
        choices=["beat", "phrase"],
        default="phrase",
        help="Which profile to flatten as the exported embedding.",
    )
    parser.add_argument(
        "--pooling-mode",
        choices=["center", "mean", "max", "topk_mean"],
        default=DEFAULT_POOLING_MODE,
    )
    parser.add_argument("--pooling-topk", type=int, default=DEFAULT_POOLING_TOPK)
    parser.add_argument(
        "--disable-normalize-per-beat",
        action="store_true",
        help="Disable per-beat activity normalization before averaging.",
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

    payload = generate_groove_embedding(
        audio_file=audio_file,
        sample_rate=int(args.sample_rate),
        tempocnn_sample_rate=int(args.tempocnn_sample_rate),
        model_file=args.model_file,
        auto_download_model=bool(args.auto_download_model),
        snippet_length_sec=args.snippet_length_sec,
        hop_length=int(args.hop_length),
        n_fft=int(args.n_fft),
        manual_bpm=args.manual_bpm,
        onset_time_sec=args.onset_time_sec,
        auto_phase_align=not bool(args.disable_phase_align),
        phase_align_mode=args.phase_align_mode,
        phase_align_max_shift_sec=float(args.phase_align_max_shift_sec),
        phase_align_step_sec=float(args.phase_align_step_sec),
        auto_prepend_start_beats=not bool(args.disable_prepend_start_beats),
        subdivisions_per_beat=int(args.subdivisions_per_beat),
        beats_per_bar=int(args.beats_per_bar),
        phrase_bars=int(args.phrase_bars),
        pooling_mode=args.pooling_mode,
        pooling_topk=int(args.pooling_topk),
        profile_mode=args.profile_mode,
        normalize_per_beat=not bool(args.disable_normalize_per_beat),
    )

    output_file = args.output_file.expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    beat_pooling = payload["beat_pooling"]
    print(f"Audio file: {_to_project_relpath(audio_file, project_root)}")
    print(f"Embedding type: {payload['embedding_subtype']}")
    print(f"Embedding dimension: {payload['embedding_dimension']}")
    print(f"BPM: {beat_pooling['bpm']:.2f}")
    print(f"Beat source: {beat_pooling['beat_source']}")
    print(f"Complete phrases: {beat_pooling['complete_phrases']}")
    print(f"Saved groove embedding JSON: {_to_project_relpath(output_file.resolve(), project_root)}")


if __name__ == "__main__":
    main()
