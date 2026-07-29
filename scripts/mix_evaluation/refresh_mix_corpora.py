#!/usr/bin/env python3
"""Register the Raveform and supplementary evaluation corpora.

The script intentionally keeps these additions separate from the established
``primary`` corpus. It can be rerun after evaluating the listed manifests to
replace their registry and combined-metric rows without touching other mixes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_COLUMNS = [
    "mix_slug",
    "manifest_dataset_name",
    "corpus",
    "title",
    "dj_or_curator",
    "date",
    "year",
    "mix_type",
    "genre_tags",
    "event_or_series",
    "n_tracks",
    "n_transitions",
    "duration_seconds",
    "manifest_path",
    "audio_filename",
    "source_url",
    "tracklist_url",
    "metadata_reference_url",
    "included",
    "metadata_basis",
    "metadata_confidence",
    "needs_review",
    "review_notes",
]

SUPPLEMENTARY_RENDERED: dict[str, dict[str, str]] = {
    "helenahauff2017dekmantel": {
        "title": "Helena Hauff - Dekmantel 2017",
        "dj_or_curator": "Helena Hauff",
        "year": "2017",
        "mix_type": "live_set",
        "genre_tags": "electro;techno",
        "event_or_series": "Dekmantel",
    },
    "jeffmills1996mixupvol2": {
        "title": "Jeff Mills - Mix Up Vol. 2 (1996)",
        "dj_or_curator": "Jeff Mills",
        "year": "1996",
        "mix_type": "radio_mix",
        "genre_tags": "techno",
        "event_or_series": "Mix Up",
    },
    "kaytranada2013boilerroommontreal": {
        "title": "Kaytranada - Boiler Room Montreal 2013",
        "dj_or_curator": "Kaytranada",
        "year": "2013",
        "mix_type": "livestream",
        "genre_tags": "electronic;hip-hop;r&b",
        "event_or_series": "Boiler Room Montreal",
    },
    "kilimanjaro2025mixmaglab": {
        "title": "Kilimanjaro - Mixmag Lab 2025",
        "dj_or_curator": "Kilimanjaro",
        "year": "2025",
        "mix_type": "radio_mix",
        "genre_tags": "afro house;house",
        "event_or_series": "Mixmag Lab",
    },
    "majorleaguedjzdbngogo2021balconymix": {
        "title": "Major League DJz and DBN Gogo - Balcony Mix 2021",
        "dj_or_curator": "Major League DJz and DBN Gogo",
        "year": "2021",
        "mix_type": "livestream",
        "genre_tags": "amapiano;house",
        "event_or_series": "Balcony Mix",
    },
    "mala2015boilerroomlondon": {
        "title": "Mala - Boiler Room London 2015",
        "dj_or_curator": "Mala",
        "year": "2015",
        "mix_type": "livestream",
        "genre_tags": "dubstep;dub",
        "event_or_series": "Boiler Room London",
    },
    "motorcitydrumensemble2014dekmantel": {
        "title": "Motor City Drum Ensemble - Dekmantel 2014",
        "dj_or_curator": "Motor City Drum Ensemble",
        "year": "2014",
        "mix_type": "live_set",
        "genre_tags": "house;disco",
        "event_or_series": "Dekmantel",
    },
    "nooriyah2022boilerroommiddleofnowhere": {
        "title": "Nooriyah - Boiler Room Middle of Nowhere 2022",
        "dj_or_curator": "Nooriyah",
        "year": "2022",
        "mix_type": "livestream",
        "genre_tags": "global bass;club",
        "event_or_series": "Boiler Room Middle of Nowhere",
    },
    "prydz2025toronto": {
        "title": "Eric Prydz - Toronto 2025",
        "dj_or_curator": "Eric Prydz",
        "year": "2025",
        "mix_type": "live_set",
        "genre_tags": "progressive house;techno",
        "event_or_series": "Toronto",
    },
    "rosapistola2017boilerroomnewyork": {
        "title": "Rosa Pistola - Boiler Room New York 2017",
        "dj_or_curator": "Rosa Pistola",
        "year": "2017",
        "mix_type": "livestream",
        "genre_tags": "reggaeton;global bass",
        "event_or_series": "Boiler Room New York",
    },
    "sherelle2019boilerroomfestival": {
        "title": "Sherelle - Boiler Room Festival 2019",
        "dj_or_curator": "Sherelle",
        "year": "2019",
        "mix_type": "livestream",
        "genre_tags": "jungle;drum and bass",
        "event_or_series": "Boiler Room Festival",
    },
}

RAVEFORM_METADATA: dict[int, dict[str, str]] = {
    1: {"title": "Cosmic Gate - Trancefusion Autumn, Industrial Palace Prague", "dj_or_curator": "Cosmic Gate", "mix_type": "live_set", "genre_tags": "trance", "event_or_series": "Trancefusion Autumn"},
    2: {"title": "Lane 8 - The Anjunadeep Edition 028", "dj_or_curator": "Lane 8", "mix_type": "podcast", "genre_tags": "deep house;progressive house", "event_or_series": "The Anjunadeep Edition"},
    3: {"title": "Daniel Avery - Boiler Room Dekmantel Festival (Day 2)", "dj_or_curator": "Daniel Avery", "mix_type": "livestream", "genre_tags": "techno;electronic", "event_or_series": "Boiler Room Dekmantel"},
    4: {"title": "Tensnake - On Repeat 2 Promo Mix", "dj_or_curator": "Tensnake", "mix_type": "promo_mix", "genre_tags": "house;disco", "event_or_series": "On Repeat"},
    5: {"title": "Dominik Eulberg - Resident Advisor RA.012", "dj_or_curator": "Dominik Eulberg", "mix_type": "podcast", "genre_tags": "techno;minimal techno", "event_or_series": "Resident Advisor"},
    6: {"title": "Magit Cacoon - Viva La Electronica", "dj_or_curator": "Magit Cacoon", "mix_type": "podcast", "genre_tags": "techno", "event_or_series": "Viva La Electronica"},
    7: {"title": "Claptone - Clapcast 182", "dj_or_curator": "Claptone", "mix_type": "podcast", "genre_tags": "house;deep house", "event_or_series": "Clapcast"},
    8: {"title": "Claptone - Clapcast 148", "dj_or_curator": "Claptone", "mix_type": "podcast", "genre_tags": "house;deep house", "event_or_series": "Clapcast"},
    9: {"title": "Headhunterz - Hard With Style 29", "dj_or_curator": "Headhunterz", "mix_type": "podcast", "genre_tags": "hardstyle", "event_or_series": "Hard With Style"},
    10: {"title": "Sven Tasnadi - My Favourite Freaks Podcast 157", "dj_or_curator": "Sven Tasnadi", "mix_type": "podcast", "genre_tags": "house;tech house", "event_or_series": "My Favourite Freaks Podcast"},
    11: {"title": "Lukas Sawicki - Off the Record, IDA Radio", "dj_or_curator": "Lukas Sawicki", "mix_type": "radio_mix", "genre_tags": "house;techno", "event_or_series": "IDA Radio"},
    12: {"title": "Spartaque - 10 Years Forsage, Kiev (Supreme 231)", "dj_or_curator": "Spartaque", "mix_type": "live_set", "genre_tags": "techno", "event_or_series": "Forsage"},
    13: {"title": "Vladimir Acic - 1605 Podcast 248", "dj_or_curator": "Vladimir Acic", "mix_type": "podcast", "genre_tags": "techno", "event_or_series": "1605 Podcast"},
    14: {"title": "Dave Seaman - Suara Podcast 61", "dj_or_curator": "Dave Seaman", "mix_type": "podcast", "genre_tags": "progressive house;techno", "event_or_series": "Suara Podcast"},
    15: {"title": "Adam Beyer - Drumcode Radio DCR334 X-Mas Mix", "dj_or_curator": "Adam Beyer", "mix_type": "radio_mix", "genre_tags": "techno", "event_or_series": "Drumcode Radio"},
    16: {"title": "Jan Blomqvist - Mayan Warrior, Burning Man", "dj_or_curator": "Jan Blomqvist", "mix_type": "live_set", "genre_tags": "melodic techno;live electronic", "event_or_series": "Mayan Warrior"},
    17: {"title": "Alex M.O.R.P.H. - Universal Nation UN068, DI.FM", "dj_or_curator": "Alex M.O.R.P.H.", "mix_type": "radio_mix", "genre_tags": "trance", "event_or_series": "Universal Nation"},
    18: {"title": "Movement Machina - Anjunabeats Worldwide 472", "dj_or_curator": "Movement Machina", "mix_type": "radio_mix", "genre_tags": "trance;progressive house", "event_or_series": "Anjunabeats Worldwide"},
    19: {"title": "Andy C b2b Shimon - One Nation, Bagleys London", "dj_or_curator": "Andy C b2b Shimon", "mix_type": "live_set", "genre_tags": "drum and bass;jungle", "event_or_series": "One Nation"},
    20: {"title": "Ferry Corsten - Corsten's Countdown 312", "dj_or_curator": "Ferry Corsten", "mix_type": "radio_mix", "genre_tags": "trance", "event_or_series": "Corsten's Countdown"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--merge-metrics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _slug_from_manifest(path: Path) -> str:
    for suffix in ("_rendered_set_manifest.json", "_build_manifest.json"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    raise ValueError(f"Unrecognized manifest name: {path.name}")


def _manifest_counts(manifest: dict[str, Any]) -> tuple[int, int, str]:
    tracks = manifest.get("tracks")
    if isinstance(tracks, list) and tracks:
        duration = tracks[-1].get("playback_end_seconds")
        return len(tracks), len(tracks) - 1, "" if duration is None else f"{float(duration):.3f}"
    validation = manifest.get("validation")
    if isinstance(validation, dict) and validation.get("rows") is not None:
        n_tracks = int(validation["rows"])
        return n_tracks, max(0, n_tracks - 1), ""
    raise ValueError("Manifest has neither rendered tracks nor a validation row count.")


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _empty_record() -> dict[str, str]:
    return {column: "" for column in REGISTRY_COLUMNS}


def _record(
    *,
    manifest_path: Path,
    project_root: Path,
    corpus: str,
    metadata: dict[str, str],
    notes: str,
) -> dict[str, str]:
    manifest = _read_json(manifest_path)
    slug = _slug_from_manifest(manifest_path)
    n_tracks, n_transitions, duration = _manifest_counts(manifest)
    source_audio = str(manifest.get("source_audio_path") or "")
    record = _empty_record()
    record.update(
        {
            "mix_slug": slug,
            "manifest_dataset_name": str(manifest.get("dataset_name") or slug),
            "corpus": corpus,
            "title": metadata["title"],
            "dj_or_curator": metadata["dj_or_curator"],
            "date": metadata.get("date", ""),
            "year": metadata["year"],
            "mix_type": metadata["mix_type"],
            "genre_tags": metadata["genre_tags"],
            "event_or_series": metadata["event_or_series"],
            "n_tracks": str(n_tracks),
            "n_transitions": str(n_transitions),
            "duration_seconds": duration,
            "manifest_path": _relative(manifest_path, project_root),
            "audio_filename": Path(source_audio).name if source_audio else "",
            "included": "true",
            "metadata_basis": "manifest;source_filename",
            "metadata_confidence": "medium",
            "needs_review": "true",
            "review_notes": notes,
        }
    )
    return record


def build_records(project_root: Path) -> list[dict[str, str]]:
    exports = project_root / "data" / "exports"
    records: list[dict[str, str]] = []

    for slug, metadata in SUPPLEMENTARY_RENDERED.items():
        manifest_path = exports / f"{slug}_rendered_set_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        records.append(
            _record(
                manifest_path=manifest_path,
                project_root=project_root,
                corpus="supplementary_rendered",
                metadata=metadata,
                notes="Supplementary rendered-set corpus; confirm source and tracklist metadata before formal reporting.",
            )
        )

    pattern = re.compile(r"raveform-(\d{3})-.*_build_manifest\.json$")
    raveform_paths = sorted(exports.glob("raveform-*_build_manifest.json"))
    if len(raveform_paths) != len(RAVEFORM_METADATA):
        raise ValueError(f"Expected {len(RAVEFORM_METADATA)} Raveform manifests, found {len(raveform_paths)}.")
    for manifest_path in raveform_paths:
        match = pattern.match(manifest_path.name)
        if match is None:
            raise ValueError(f"Unrecognized Raveform manifest name: {manifest_path.name}")
        index = int(match.group(1))
        metadata = {**RAVEFORM_METADATA[index]}
        date_match = re.search(r"-(\d{4})-(\d{2})-(\d{2})-", manifest_path.name)
        if date_match is None:
            raise ValueError(f"Missing date in Raveform manifest name: {manifest_path.name}")
        metadata["date"] = "-".join(date_match.groups())
        metadata["year"] = date_match.group(1)
        records.append(
            _record(
                manifest_path=manifest_path,
                project_root=project_root,
                corpus="raveform_tracklist",
                metadata=metadata,
                notes=(
                    "Raveform cue-tracklist corpus; layered cue entries remain ordered by source "
                    "track number and are evaluated as adjacent transitions."
                ),
            )
        )
    return records


def write_registry(path: Path, records: list[dict[str, str]], *, dry_run: bool) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        existing = list(reader)
    if fieldnames != REGISTRY_COLUMNS:
        raise ValueError(f"Unexpected registry schema in {path}")
    replacement_slugs = {record["mix_slug"] for record in records}
    merged = [row for row in existing if row["mix_slug"] not in replacement_slugs] + records
    if dry_run:
        print(f"[dry-run] registry: {len(existing)} -> {len(merged)} rows")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(merged)
    print(f"Wrote {path} with {len(merged)} rows.")


def merge_metric_rows(project_root: Path, records: list[dict[str, str]], *, dry_run: bool) -> None:
    evaluation_dir = project_root / "data" / "mix_evaluation"
    combined_path = evaluation_dir / "all_processed_simplex_metrics.csv"
    replacement_slugs = {record["mix_slug"] for record in records}
    new_rows: list[dict[str, str]] = []
    new_columns: list[str] = []
    for slug in sorted(replacement_slugs):
        path = evaluation_dir / f"{slug}_simplex_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Evaluate this manifest before merging metrics: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            new_columns.extend(column for column in reader.fieldnames or [] if column not in new_columns)
            new_rows.extend(reader)
    with combined_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing_columns = list(reader.fieldnames or [])
        existing_rows = [row for row in reader if row["mix_slug"] not in replacement_slugs]
    columns = existing_columns + [column for column in new_columns if column not in existing_columns]
    merged = existing_rows + new_rows
    if dry_run:
        print(f"[dry-run] metrics: {len(existing_rows)} retained + {len(new_rows)} replacement rows")
        return
    with combined_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(merged)
    print(f"Wrote {combined_path} with {len(merged)} rows across {len({row['mix_slug'] for row in merged})} mixes.")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    records = build_records(project_root)
    corpus_counts: dict[str, int] = {}
    for record in records:
        corpus_counts[record["corpus"]] = corpus_counts.get(record["corpus"], 0) + 1
    print(f"Prepared {len(records)} registry records: {corpus_counts}")
    write_registry(project_root / "data" / "mix_evaluation" / "mix_registry.csv", records, dry_run=bool(args.dry_run))
    if args.merge_metrics:
        merge_metric_rows(project_root, records, dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
