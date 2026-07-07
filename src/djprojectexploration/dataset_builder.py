"""Orchestrate creation of an app-ready DJ dataset from a folder or tracklist."""

from __future__ import annotations

import argparse
from pathlib import Path

from djprojectexploration.audio_snippets import DEFAULT_SCAN_HOP_SECONDS, ensure_cached_snippet
from djprojectexploration.energy_features import create_energy_embedding_npz, create_energy_feature_csv
from djprojectexploration.energy_sequence_builder import export_dj_sequence
from djprojectexploration.music_folder_ingest import ingest_music_folder
from djprojectexploration.playlist_embedding_pipeline import (
    create_chroma_playlist_embeddings_npz,
    create_groove_playlist_embeddings_npz,
    create_maest_playlist_embeddings_npz,
    create_tempo_playlist_embeddings_npz,
)
from djprojectexploration.tracklists import PROJECT_ROOT, load_playlist_tracks
from djprojectexploration.waveform_features import create_waveform_playlist_features_npz


def _slug(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    return "-".join(part for part in text.split("-") if part) or "music-library"


def _tracklist_stem(mix_slug: str) -> str:
    return f"{mix_slug.replace('-', '_')}_tracks"


def _default_tracklist_path(project_root: Path, mix_slug: str) -> Path:
    return project_root / "music" / mix_slug / f"{_tracklist_stem(mix_slug)}.csv"


def _build_snippet_cache(tracklist_csv: Path, *, music_dir: Path | None, skip_missing_audio: bool, overwrite: bool) -> Path:
    tracks = load_playlist_tracks(tracklist_csv, music_dir=music_dir, skip_missing_audio=skip_missing_audio)
    output_dir = PROJECT_ROOT / "data" / "snippets" / tracklist_csv.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(tracks)
    for index, track in enumerate(tracks, start=1):
        print(f"[{index}/{total}] Caching snippet: {track.title}")
        ensure_cached_snippet(
            audio_path=track.audio_path,
            output_dir=output_dir,
            key=track.mp3_name,
            snippet_seconds=8.0,
            middle_fraction=0.66,
            hop_seconds=DEFAULT_SCAN_HOP_SECONDS,
            overwrite=overwrite,
            project_root=PROJECT_ROOT,
        )
    return output_dir


def build_dataset(
    source: str | Path,
    *,
    name: str,
    tracklist: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
    overwrite_tracklist: bool = False,
    skip_missing_audio: bool = False,
    skip_snippets: bool = False,
    skip_waveforms: bool = False,
    skip_embeddings: bool = False,
    skip_energy: bool = False,
    skip_app_export: bool = False,
    static_layout: bool = False,
) -> dict[str, Path]:
    """Build tracklist, media caches, feature bundles, energy NPZ, and app HTML."""
    project_root = project_root.expanduser().resolve()
    mix_slug = _slug(name)
    source_path = Path(source).expanduser().resolve()
    if tracklist is None:
        tracklist_path = _default_tracklist_path(project_root, mix_slug)
        tracklist_path = ingest_music_folder(
            source_path,
            output_file=tracklist_path,
            name=mix_slug,
            overwrite=overwrite_tracklist,
        )
        music_dir = source_path
    else:
        tracklist_path = Path(tracklist).expanduser().resolve()
        music_dir = source_path if source_path.is_dir() else tracklist_path.parent

    outputs: dict[str, Path] = {"tracklist": tracklist_path}

    if not skip_snippets:
        outputs["snippets"] = _build_snippet_cache(
            tracklist_path,
            music_dir=music_dir,
            skip_missing_audio=skip_missing_audio,
            overwrite=False,
        )

    if not skip_waveforms:
        outputs["waveforms"] = create_waveform_playlist_features_npz(
            tracklist_path,
            music_dir=music_dir,
            skip_missing_audio=skip_missing_audio,
        )

    if not skip_embeddings:
        outputs["maest"] = create_maest_playlist_embeddings_npz(
            tracklist_path,
            music_dir=music_dir,
            skip_missing_audio=skip_missing_audio,
        )
        outputs["chroma"] = create_chroma_playlist_embeddings_npz(
            tracklist_path,
            music_dir=music_dir,
            skip_missing_audio=skip_missing_audio,
        )
        outputs["tempo"] = create_tempo_playlist_embeddings_npz(
            tracklist_path,
            music_dir=music_dir,
            skip_missing_audio=skip_missing_audio,
        )
        outputs["groove"] = create_groove_playlist_embeddings_npz(
            tracklist_path,
            music_dir=music_dir,
            skip_missing_audio=skip_missing_audio,
        )

    if not skip_energy:
        energy_csv = create_energy_feature_csv(
            tracklist_path,
            music_dir=music_dir,
            skip_missing_audio=skip_missing_audio,
        )
        outputs["energy_features"] = energy_csv
        outputs["energy_npz"] = create_energy_embedding_npz(
            [energy_csv],
            name=f"{mix_slug.replace('-', '_')}_energy_features",
        )

    if not skip_app_export:
        energy_npz = outputs.get("energy_npz")
        if energy_npz is None:
            energy_npz = project_root / "data" / "energy_embeddings" / f"{mix_slug.replace('-', '_')}_energy_features.npz"
        outputs["sequence_html"] = export_dj_sequence(
            project_root=project_root,
            mix_slugs=[mix_slug],
            energy_npz_path=energy_npz,
            static_layout=static_layout,
        )

    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an app-ready DJ dataset from a music folder or tracklist.")
    parser.add_argument("source", type=Path, help="Music folder. With --tracklist, this is used as --music-dir.")
    parser.add_argument("--name", required=True, help="Dataset/mix slug, e.g. my-set.")
    parser.add_argument("--tracklist", type=Path, default=None, help="Use an existing tracklist instead of scanning source.")
    parser.add_argument("--overwrite-tracklist", action="store_true")
    parser.add_argument("--skip-missing-audio", action="store_true")
    parser.add_argument("--skip-snippets", action="store_true")
    parser.add_argument("--skip-waveforms", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-energy", action="store_true")
    parser.add_argument("--skip-app-export", action="store_true")
    parser.add_argument("--static-layout", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build_dataset(
        args.source,
        name=args.name,
        tracklist=args.tracklist,
        overwrite_tracklist=bool(args.overwrite_tracklist),
        skip_missing_audio=bool(args.skip_missing_audio),
        skip_snippets=bool(args.skip_snippets),
        skip_waveforms=bool(args.skip_waveforms),
        skip_embeddings=bool(args.skip_embeddings),
        skip_energy=bool(args.skip_energy),
        skip_app_export=bool(args.skip_app_export),
        static_layout=bool(args.static_layout),
    )
    print("Dataset build outputs:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
