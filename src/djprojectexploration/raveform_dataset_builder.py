"""Normalize Raveform tracklists and run the canonical dataset builder in batches."""

from __future__ import annotations

import argparse
import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {
    "track_number",
    "track_name",
    "artist",
    "start_seconds",
    "audio_file",
}
NORMALIZED_COLUMNS = [
    "track_number",
    "title",
    "artists",
    "filepath",
    "mix_start_seconds",
    "transition_group",
    "raveform_source_csv",
]


@dataclass(frozen=True)
class RaveformMix:
    index: int
    csv_path: Path
    audio_dir: Path
    dataset_name: str


@dataclass(frozen=True)
class NormalizedRaveformMix:
    mix: RaveformMix
    rows: list[dict[str, str]]
    transition_groups: int

    @property
    def layered_entries(self) -> int:
        return len(self.rows) - self.transition_groups


def _slug(value: str) -> str:
    text = "".join(character.lower() if character.isalnum() else "-" for character in value.strip())
    return "-".join(part for part in text.split("-") if part) or "raveform-mix"


def discover_raveform_mixes(
    raveform_root: str | Path,
    *,
    start: int = 1,
    end: int = 20,
    name_prefix: str = "raveform",
) -> list[RaveformMix]:
    root = Path(raveform_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Raveform mix directory not found: {root}")
    if start <= 0 or end < start:
        raise ValueError("Expected positive mix bounds with end >= start.")

    mixes: list[RaveformMix] = []
    for index in range(start, end + 1):
        candidates = sorted(
            path
            for path in root.glob(f"{index:03d}_*.csv")
            if not path.name.endswith(".youtube.csv")
        )
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one source CSV for mix {index:03d}, found {len(candidates)}."
            )

        csv_path = candidates[0].resolve()
        audio_dir = (root / csv_path.stem).resolve()
        if not audio_dir.is_dir():
            raise NotADirectoryError(f"Audio directory not found for mix {index:03d}: {audio_dir}")

        dataset_name = _slug(f"{name_prefix}-{csv_path.stem}")
        mixes.append(
            RaveformMix(
                index=index,
                csv_path=csv_path,
                audio_dir=audio_dir,
                dataset_name=dataset_name,
            )
        )
    return mixes


def _resolve_audio_path(raw_path: str, *, mix: RaveformMix, raveform_root: Path) -> Path | None:
    if not raw_path.strip():
        return None

    raw = Path(raw_path).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        raveform_root.parent / raw,
        mix.csv_path.parent / raw,
        mix.audio_dir / raw.name,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return candidates[0].resolve()


def normalize_raveform_mix(
    mix: RaveformMix,
    *,
    raveform_root: str | Path,
    group_tolerance_seconds: float = 1.0,
) -> NormalizedRaveformMix:
    root = Path(raveform_root).expanduser().resolve()
    tolerance = float(group_tolerance_seconds)
    if tolerance < 0:
        raise ValueError("group_tolerance_seconds must be nonnegative.")

    with mix.csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - fields)
        if missing_columns:
            raise ValueError(
                f"{mix.csv_path.name} is missing required columns: {missing_columns}"
            )
        source_rows = list(reader)

    if not source_rows:
        raise ValueError(f"Raveform CSV contains no tracks: {mix.csv_path}")

    normalized_rows: list[dict[str, str]] = []
    missing_audio: list[str] = []
    seen_track_numbers: set[int] = set()
    previous_start: float | None = None
    transition_group = 0

    for row_number, row in enumerate(source_rows, start=2):
        try:
            track_number = int((row.get("track_number") or "").strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid track_number at {mix.csv_path.name}:{row_number}"
            ) from exc
        if track_number in seen_track_numbers:
            raise ValueError(
                f"Duplicate track_number={track_number} in {mix.csv_path.name}"
            )
        seen_track_numbers.add(track_number)

        title = (row.get("track_name") or "").strip()
        artists = (row.get("artist") or "").strip()
        if not title:
            raise ValueError(f"Missing track_name at {mix.csv_path.name}:{row_number}")

        try:
            start_seconds = float((row.get("start_seconds") or "").strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid start_seconds at {mix.csv_path.name}:{row_number}"
            ) from exc
        if start_seconds < 0:
            raise ValueError(
                f"Negative start_seconds at {mix.csv_path.name}:{row_number}"
            )
        if previous_start is not None and start_seconds <= previous_start:
            raise ValueError(
                f"start_seconds must be strictly increasing in {mix.csv_path.name}"
            )
        if previous_start is None or start_seconds - previous_start > tolerance:
            transition_group += 1
        previous_start = start_seconds

        audio_path = _resolve_audio_path(
            row.get("audio_file") or "",
            mix=mix,
            raveform_root=root,
        )
        if audio_path is None or not audio_path.is_file():
            missing_audio.append(f"{track_number}: {title}")
            continue

        normalized_rows.append(
            {
                "track_number": str(track_number),
                "title": title,
                "artists": artists,
                "filepath": str(audio_path),
                "mix_start_seconds": f"{start_seconds:.6f}".rstrip("0").rstrip("."),
                "transition_group": str(transition_group),
                "raveform_source_csv": str(mix.csv_path),
            }
        )

    if missing_audio:
        details = "\n  ".join(missing_audio)
        raise FileNotFoundError(
            f"{mix.csv_path.name} has {len(missing_audio)} missing audio file(s):\n  {details}"
        )

    ordered_numbers = [int(row["track_number"]) for row in normalized_rows]
    if ordered_numbers != sorted(ordered_numbers):
        raise ValueError(f"track_number order is not increasing in {mix.csv_path.name}")

    return NormalizedRaveformMix(
        mix=mix,
        rows=normalized_rows,
        transition_groups=transition_group,
    )


def write_normalized_tracklist(
    normalized: NormalizedRaveformMix,
    output_file: str | Path,
) -> Path:
    destination = Path(output_file).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized.rows)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize Raveform tracklists and run the full app-ready dataset builder "
            "for a numbered mix range."
        )
    )
    parser.add_argument("raveform_root", type=Path, help="Directory containing 001_*.csv and audio folders.")
    parser.add_argument("--start", type=int, default=1, help="First numbered mix to build.")
    parser.add_argument("--end", type=int, default=20, help="Last numbered mix to build.")
    parser.add_argument("--name-prefix", default="raveform", help="Prefix used for generated dataset names.")
    parser.add_argument(
        "--group-tolerance-seconds",
        type=float,
        default=1.0,
        help="Treat adjacent cue timestamps within this interval as one layered transition group.",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate discovery, metadata, ordering, and every audio path without building.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate feature artifacts and exports, including already completed mixes.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-enter completed builds while retaining the builder's per-artifact resume behavior.",
    )
    parser.set_defaults(skip_existing=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-metadata-enrichment", action="store_true")
    parser.add_argument("--skip-energy", action="store_true")
    parser.add_argument("--skip-sequence-export", action="store_true")
    parser.add_argument("--skip-pacmap-export", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raveform_root = args.raveform_root.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    mixes = discover_raveform_mixes(
        raveform_root,
        start=int(args.start),
        end=int(args.end),
        name_prefix=str(args.name_prefix),
    )

    validated: list[NormalizedRaveformMix] = []
    for mix in mixes:
        normalized = normalize_raveform_mix(
            mix,
            raveform_root=raveform_root,
            group_tolerance_seconds=float(args.group_tolerance_seconds),
        )
        validated.append(normalized)
        print(
            f"[validate] {mix.index:03d}: {len(normalized.rows)} tracks, "
            f"{normalized.transition_groups} transition groups, "
            f"{normalized.layered_entries} layered entries"
        )

    print(
        f"Validated {len(validated)} mixes and "
        f"{sum(len(item.rows) for item in validated)} audio files."
    )
    if args.dry_run:
        print("Dry run complete; no project files were written.")
        return 0

    from djprojectexploration.dataset_builder import build_dataset

    failures: list[tuple[RaveformMix, Exception]] = []
    completed = 0
    skipped = 0
    with tempfile.TemporaryDirectory(prefix="raveform-normalized-tracklists-") as temporary:
        temporary_root = Path(temporary)
        for normalized in validated:
            mix = normalized.mix
            manifest_path = (
                project_root / "data" / "exports" / f"{mix.dataset_name}_build_manifest.json"
            )
            if manifest_path.exists() and bool(args.skip_existing) and not bool(args.force):
                print(f"[skip] {mix.index:03d}: completed manifest exists: {manifest_path}")
                skipped += 1
                continue

            normalized_tracklist = write_normalized_tracklist(
                normalized,
                temporary_root / f"{mix.dataset_name}_tracks.csv",
            )
            print(f"[build] {mix.index:03d}: {mix.dataset_name}")
            try:
                build_dataset(
                    mix.audio_dir,
                    name=mix.dataset_name,
                    tracklist=normalized_tracklist,
                    project_root=project_root,
                    overwrite_tracklist=True,
                    skip_missing_audio=False,
                    skip_metadata_enrichment=bool(args.skip_metadata_enrichment),
                    skip_energy=bool(args.skip_energy),
                    skip_sequence_export=bool(args.skip_sequence_export),
                    skip_pacmap_export=bool(args.skip_pacmap_export),
                    force=bool(args.force),
                )
                completed += 1
            except Exception as exc:
                failures.append((mix, exc))
                print(f"[failed] {mix.index:03d}: {exc}")
                if args.fail_fast:
                    break

    print(
        f"Batch result: completed={completed}, skipped={skipped}, failed={len(failures)}"
    )
    if failures:
        for mix, error in failures:
            print(f"  {mix.index:03d} {mix.dataset_name}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
