"""Orchestrate creation of an app-ready DJ dataset from a folder or tracklist."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from djprojectexploration.energy_features import (
    DEFAULT_FROZEN_ENERGY_MODEL_FILE,
    create_energy_embedding_npz,
    create_energy_feature_csv,
    energy_model_artifact_uses_maest_full_peak30,
)
from djprojectexploration.energy_sequence_builder import export_dj_sequence
from djprojectexploration.interactive_pacmap_knn_simplex import export_dj_pacmap
from djprojectexploration.music_folder_ingest import ingest_music_folder
from djprojectexploration.pacmap_settings import PacmapSettings, add_pacmap_args, pacmap_settings_from_args
from djprojectexploration.playlist_embedding_pipeline import (
    create_chroma_playlist_embeddings_npz,
    create_groove_playlist_embeddings_npz,
    create_maest_playlist_embeddings_npz,
    create_tempo_playlist_embeddings_npz,
)
from djprojectexploration.tracklists import PROJECT_ROOT
from djprojectexploration.tracklist_validation import validate_tracklist, write_validation_json
from djprojectexploration.waveform_features import create_waveform_playlist_features_npz


def _slug(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    return "-".join(part for part in text.split("-") if part) or "music-library"


def _tracklist_stem(mix_slug: str) -> str:
    return f"{mix_slug.replace('-', '_')}_tracks"


def _default_tracklist_path(project_root: Path, mix_slug: str) -> Path:
    return project_root / "music" / mix_slug / f"{_tracklist_stem(mix_slug)}.csv"


def _default_energy_model_path(project_root: Path) -> Path:
    return project_root / "data" / "energy_models" / DEFAULT_FROZEN_ENERGY_MODEL_FILE.name


def _resolve_energy_model_file(
    project_root: Path,
    energy_model_file: str | Path | None,
    *,
    refit_energy_model: bool,
) -> Path | None:
    if refit_energy_model:
        return None
    if energy_model_file is None:
        return _default_energy_model_path(project_root)
    return Path(energy_model_file).expanduser().resolve()


def _copy_tracklist_if_needed(source: Path, target: Path, *, overwrite: bool) -> Path:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    if source == target:
        return target
    if target.exists() and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {rem:.0f}s"


@contextmanager
def _stage(name: str) -> Iterator[None]:
    start = time.monotonic()
    print(f"[stage] {name}: start", flush=True)
    try:
        yield
    finally:
        print(f"[stage] {name}: done in {_format_duration(time.monotonic() - start)}", flush=True)


def build_dataset(
    source: str | Path,
    *,
    name: str,
    tracklist: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
    overwrite_tracklist: bool = False,
    skip_missing_audio: bool = False,
    skip_waveforms: bool = False,
    skip_embeddings: bool = False,
    skip_energy: bool = False,
    energy_model_file: str | Path | None = None,
    refit_energy_model: bool = False,
    skip_sequence_export: bool = False,
    skip_pacmap_export: bool = False,
    force: bool = False,
    pacmap_settings: PacmapSettings | None = None,
) -> dict[str, Path]:
    """Build tracklist, waveform caches, feature bundles, energy NPZ, and app HTML."""
    project_root = project_root.expanduser().resolve()
    mix_slug = _slug(name)
    source_path = Path(source).expanduser().resolve()
    with _stage("tracklist"):
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
            tracklist_path = _copy_tracklist_if_needed(
                Path(tracklist),
                _default_tracklist_path(project_root, mix_slug),
                overwrite=overwrite_tracklist,
            )
            music_dir = source_path if source_path.is_dir() else tracklist_path.parent

    outputs: dict[str, Path] = {"tracklist": tracklist_path}
    with _stage("validation"):
        validation = validate_tracklist(tracklist_path, music_dir=music_dir, skip_missing_audio=skip_missing_audio)
        outputs["validation"] = write_validation_json(
            validation,
            project_root / "data" / "exports" / f"{mix_slug}_validation.json",
        )
    pacmap_settings = (pacmap_settings or PacmapSettings()).validate()
    resolved_energy_model_file = _resolve_energy_model_file(
        project_root,
        energy_model_file,
        refit_energy_model=refit_energy_model,
    )
    energy_model_needs_peak30_maest = False
    if not skip_energy and resolved_energy_model_file is not None:
        if not resolved_energy_model_file.exists():
            raise FileNotFoundError(f"Energy model artifact not found: {resolved_energy_model_file}")
        energy_model_needs_peak30_maest = energy_model_artifact_uses_maest_full_peak30(resolved_energy_model_file)

    def should_build(path: Path | None) -> bool:
        return force or path is None or not path.exists()

    waveform_path = project_root / "data" / "waveform_features" / f"{tracklist_path.stem}.npz"
    if not skip_waveforms:
        with _stage("waveforms"):
            outputs["waveforms"] = (
                create_waveform_playlist_features_npz(
                    tracklist_path,
                    music_dir=music_dir,
                    skip_missing_audio=skip_missing_audio,
                )
                if should_build(waveform_path)
                else waveform_path
            )

    maest_path = project_root / "data" / "maest_embeddings" / f"{tracklist_path.stem}.npz"
    maest_peak30_path = project_root / "data" / "maest_embeddings" / f"{tracklist_path.stem}_peak30.npz"
    chroma_path = project_root / "data" / "chroma_embeddings" / f"{tracklist_path.stem}.npz"
    tempo_path = project_root / "data" / "tempo_embeddings" / f"{tracklist_path.stem}.npz"
    groove_path = project_root / "data" / "groove_embeddings" / f"{tracklist_path.stem}.npz"
    if not skip_embeddings:
        with _stage("maest"):
            outputs["maest"] = create_maest_playlist_embeddings_npz(
                tracklist_path,
                music_dir=music_dir,
                skip_missing_audio=skip_missing_audio,
            ) if should_build(maest_path) else maest_path
        if energy_model_needs_peak30_maest:
            with _stage("maest peak30"):
                outputs["maest_peak30"] = create_maest_playlist_embeddings_npz(
                    tracklist_path,
                    music_dir=music_dir,
                    skip_missing_audio=skip_missing_audio,
                    section="peak30",
                ) if should_build(maest_peak30_path) else maest_peak30_path
        with _stage("chroma"):
            outputs["chroma"] = create_chroma_playlist_embeddings_npz(
                tracklist_path,
                music_dir=music_dir,
                skip_missing_audio=skip_missing_audio,
            ) if should_build(chroma_path) else chroma_path
        with _stage("tempo"):
            outputs["tempo"] = create_tempo_playlist_embeddings_npz(
                tracklist_path,
                music_dir=music_dir,
                skip_missing_audio=skip_missing_audio,
            ) if should_build(tempo_path) else tempo_path
        with _stage("groove"):
            outputs["groove"] = create_groove_playlist_embeddings_npz(
                tracklist_path,
                music_dir=music_dir,
                skip_missing_audio=skip_missing_audio,
                tempo_embeddings=outputs.get("tempo", tempo_path),
            ) if should_build(groove_path) else groove_path

    energy_csv_path = project_root / "data" / "energy_features" / f"{tracklist_path.stem}_energy_features.csv"
    energy_npz_path = project_root / "data" / "energy_embeddings" / f"{mix_slug.replace('-', '_')}_energy_features.npz"
    if not skip_energy:
        with _stage("energy features"):
            energy_csv = create_energy_feature_csv(
                tracklist_path,
                music_dir=music_dir,
                skip_missing_audio=skip_missing_audio,
                tempo_embeddings=outputs.get("tempo", tempo_path),
            ) if should_build(energy_csv_path) else energy_csv_path
            outputs["energy_features"] = energy_csv
        with _stage("energy npz"):
            outputs["energy_npz"] = create_energy_embedding_npz(
                [energy_csv],
                name=f"{mix_slug.replace('-', '_')}_energy_features",
                model_file=resolved_energy_model_file,
            ) if should_build(energy_npz_path) else energy_npz_path

    if not skip_sequence_export:
        with _stage("sequence export"):
            energy_npz = outputs.get("energy_npz")
            if energy_npz is None:
                energy_npz = project_root / "data" / "energy_embeddings" / f"{mix_slug.replace('-', '_')}_energy_features.npz"
            outputs["sequence_html"] = export_dj_sequence(
                project_root=project_root,
                mix_slugs=[mix_slug],
                energy_npz_path=energy_npz,
                output_file=project_root / "data" / "exports" / f"{mix_slug}_sequence_builder.html",
                pacmap_settings=pacmap_settings,
            )
    if not skip_pacmap_export:
        with _stage("pacmap export"):
            outputs["pacmap_html"] = export_dj_pacmap(
                project_root=project_root,
                mix_slugs=[mix_slug],
                dataset_name=mix_slug,
                output_file=project_root / "data" / "exports" / f"{mix_slug}_pacmap.html",
                pacmap_settings=pacmap_settings,
                control_mode="genre-mixability",
            )

    manifest = {
        "dataset_name": mix_slug,
        "tracklist": str(tracklist_path),
        "validation": validation.to_dict(),
        "pacmap_settings": pacmap_settings.to_dict(),
        "energy_model_file": None if resolved_energy_model_file is None else str(resolved_energy_model_file),
        "refit_energy_model": bool(refit_energy_model),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    manifest_path = project_root / "data" / "exports" / f"{mix_slug}_build_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs["manifest"] = manifest_path

    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an app-ready DJ dataset from a music folder or tracklist.")
    parser.add_argument("source", type=Path, help="Music folder. With --tracklist, this is used as --music-dir.")
    parser.add_argument("--name", required=True, help="Dataset/mix slug, e.g. my-set.")
    parser.add_argument("--tracklist", type=Path, default=None, help="Use an existing tracklist instead of scanning source.")
    parser.add_argument("--overwrite-tracklist", action="store_true")
    parser.add_argument("--skip-missing-audio", action="store_true")
    parser.add_argument("--skip-snippets", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-waveforms", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-energy", action="store_true")
    parser.add_argument(
        "--energy-model-file",
        type=Path,
        default=None,
        help=(
            "Frozen energy model artifact to apply. Defaults to "
            f"data/energy_models/{DEFAULT_FROZEN_ENERGY_MODEL_FILE.name}."
        ),
    )
    parser.add_argument(
        "--refit-energy-model",
        action="store_true",
        help="Fit an energy model from this dataset instead of applying the default frozen model artifact.",
    )
    parser.add_argument("--skip-app-export", action="store_true", help="Deprecated alias for --skip-sequence-export.")
    parser.add_argument("--skip-sequence-export", action="store_true")
    parser.add_argument("--skip-pacmap-export", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate artifacts even when default output files exist.")
    add_pacmap_args(parser, include_static_layout=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pacmap_settings = pacmap_settings_from_args(args)
    outputs = build_dataset(
        args.source,
        name=args.name,
        tracklist=args.tracklist,
        overwrite_tracklist=bool(args.overwrite_tracklist),
        skip_missing_audio=bool(args.skip_missing_audio),
        skip_waveforms=bool(args.skip_waveforms),
        skip_embeddings=bool(args.skip_embeddings),
        skip_energy=bool(args.skip_energy),
        energy_model_file=args.energy_model_file,
        refit_energy_model=bool(args.refit_energy_model),
        skip_sequence_export=bool(args.skip_app_export or args.skip_sequence_export),
        skip_pacmap_export=bool(args.skip_pacmap_export),
        force=bool(args.force),
        pacmap_settings=pacmap_settings,
    )
    print("Dataset build outputs:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
