"""Generate preview-section sidecars from cached snippet metadata."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PreviewSource:
    mix_slug: str
    tracklist_csv: Path
    snippet_dir: Path
    output_csv: Path


DEFAULT_SOURCES = {
    "aries-mix": PreviewSource(
        mix_slug="aries-mix",
        tracklist_csv=PROJECT_ROOT / "music" / "aries-mix" / "aries_mix_tracks.csv",
        snippet_dir=PROJECT_ROOT / "data" / "snippets" / "aries_mix_tracks",
        output_csv=PROJECT_ROOT / "music" / "aries-mix" / "aries_mix_preview_sections.csv",
    ),
    "ara-mix": PreviewSource(
        mix_slug="ara-mix",
        tracklist_csv=PROJECT_ROOT / "music" / "ara-mix" / "ara_mix_tracks.csv",
        snippet_dir=PROJECT_ROOT / "data" / "snippets" / "ara_mix_tracks",
        output_csv=PROJECT_ROOT / "music" / "ara-mix" / "ara_mix_preview_sections.csv",
    ),
}

OUTPUT_FIELDS = [
    "track_id",
    "mix_slug",
    "track_number",
    "title",
    "artists",
    "filename",
    "preview_start_seconds",
    "preview_end_seconds",
    "preview_duration_seconds",
    "preview_score",
    "method",
    "snippet_path",
    "snippet_json",
    "source_audio_path",
    "sample_rate",
    "snippet_seconds_requested",
    "middle_fraction",
    "hop_seconds",
    "updated_at",
]


def _relpath(path_value: str | Path | None, *, project_root: Path = PROJECT_ROOT) -> str:
    if not path_value:
        return ""
    path = Path(path_value).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path_value)


def _basename_key(path_value: str | Path | None) -> str:
    if not path_value:
        return ""
    return Path(str(path_value)).name.casefold()


def _track_number_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    value = str(row.get("track_number", "")).strip()
    try:
        return (int(value), value)
    except ValueError:
        return (10**9, value)


def _load_tracklist(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows: list[dict[str, str]] = []
    by_filename: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            filename = _basename_key(row.get("mp3_name") or row.get("filename") or row.get("path"))
            if filename:
                by_filename[filename] = row
    return rows, by_filename


def _format_float(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def generate_preview_sections(source: PreviewSource) -> tuple[int, list[str]]:
    track_rows, by_filename = _load_tracklist(source.tracklist_csv)
    seen_track_numbers: set[str] = set()
    warnings: list[str] = []
    output_rows: list[dict[str, str]] = []

    for json_path in sorted(source.snippet_dir.glob("*.json")):
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(f"{json_path}: invalid JSON ({exc})")
            continue

        filename_key = _basename_key(meta.get("source_audio_path"))
        track = by_filename.get(filename_key)
        if track is None:
            warnings.append(f"{json_path}: no tracklist row for {filename_key or 'missing source_audio_path'}")
            continue

        track_number = str(track.get("track_number") or track.get("#") or "").strip()
        if not track_number:
            warnings.append(f"{json_path}: matched {filename_key} but track_number is blank")
            continue

        seen_track_numbers.add(track_number)
        output_rows.append(
            {
                "track_id": f"{source.mix_slug}:{track_number}",
                "mix_slug": source.mix_slug,
                "track_number": track_number,
                "title": str(track.get("title") or "").strip(),
                "artists": str(track.get("artists") or "").strip(),
                "filename": str(track.get("mp3_name") or filename_key).strip(),
                "preview_start_seconds": _format_float(meta.get("start_seconds"), 6),
                "preview_end_seconds": _format_float(meta.get("end_seconds"), 6),
                "preview_duration_seconds": _format_float(meta.get("duration_seconds"), 6),
                "preview_score": _format_float(meta.get("rms"), 9),
                "method": "rms_middle_scan",
                "snippet_path": _relpath(meta.get("snippet_path")),
                "snippet_json": _relpath(json_path),
                "source_audio_path": _relpath(meta.get("source_audio_path")),
                "sample_rate": str(meta.get("sample_rate") or ""),
                "snippet_seconds_requested": _format_float(meta.get("snippet_seconds_requested"), 3),
                "middle_fraction": _format_float(meta.get("middle_fraction"), 3),
                "hop_seconds": _format_float(meta.get("hop_seconds"), 3),
                "updated_at": str(meta.get("created_utc") or ""),
            }
        )

    output_rows.sort(key=_track_number_sort_key)
    missing_numbers = [
        str(row.get("track_number") or row.get("#") or "").strip()
        for row in track_rows
        if str(row.get("track_number") or row.get("#") or "").strip() not in seen_track_numbers
    ]
    if missing_numbers:
        warnings.append(
            f"{source.mix_slug}: no snippet metadata for track numbers {', '.join(missing_numbers)}"
        )

    source.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with source.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    return len(output_rows), warnings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate preview-section CSVs from snippet JSON metadata.")
    parser.add_argument(
        "--mix",
        action="append",
        choices=sorted(DEFAULT_SOURCES),
        default=None,
        help="Default mix to generate. Repeatable. Defaults to aries-mix and ara-mix.",
    )
    parser.add_argument("--tracklist", type=Path, help="Custom tracklist CSV.")
    parser.add_argument("--snippet-dir", type=Path, help="Custom directory containing *.wav.json metadata.")
    parser.add_argument("--output", type=Path, help="Custom output CSV.")
    parser.add_argument("--mix-slug", help="Custom mix slug for track_id values.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tracklist or args.snippet_dir or args.output or args.mix_slug:
        if not (args.tracklist and args.snippet_dir and args.output and args.mix_slug):
            raise SystemExit("--tracklist, --snippet-dir, --output, and --mix-slug are required together.")
        sources = [
            PreviewSource(
                mix_slug=args.mix_slug,
                tracklist_csv=args.tracklist,
                snippet_dir=args.snippet_dir,
                output_csv=args.output,
            )
        ]
    else:
        selected = args.mix or list(DEFAULT_SOURCES)
        sources = [DEFAULT_SOURCES[mix] for mix in selected]

    exit_code = 0
    for source in sources:
        count, warnings = generate_preview_sections(source)
        print(f"Wrote {count} preview sections: {source.output_csv}")
        for warning in warnings:
            print(f"Warning: {warning}")
        if warnings:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
