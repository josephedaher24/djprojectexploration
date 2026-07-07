"""Import Rekordbox XML beatgrid and cue labels into playlist CSV metadata."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

from djprojectexploration.playlist_embedding_pipeline import PROJECT_ROOT


DEFAULT_TRACKLIST = PROJECT_ROOT / "music" / "aries-mix" / "aries_mix_tracks.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "music" / "aries-mix" / "aries_mix_tracks_rekordbox.csv"
DEFAULT_CUE_OUTPUT = PROJECT_ROOT / "music" / "aries-mix" / "aries_mix_cues.csv"
CUE_COLUMN_BY_NAME = {
    "IN 1": "in_1_seconds",
    "IN 2": "in_2_seconds",
    "OUT 1": "out_1_seconds",
    "OUT 2": "out_2_seconds",
}
OUTPUT_COLUMNS = [
    "rekordbox_bpm",
    "rekordbox_onset_time",
    "in_1_seconds",
    "in_2_seconds",
    "out_1_seconds",
    "out_2_seconds",
    "cue_source",
    "rekordbox_location",
]
CUE_FIELDNAMES = [
    "track_number",
    "mp3_name",
    "title",
    "artists",
    "cue_name",
    "cue_role",
    "cue_index",
    "start_seconds",
    "end_seconds",
    "source",
    "rekordbox_type",
    "rekordbox_num",
    "rekordbox_location",
]


@dataclass(frozen=True)
class RekordboxCue:
    name: str
    start_seconds: float
    end_seconds: float | None
    rekordbox_type: str
    rekordbox_num: str


@dataclass(frozen=True)
class RekordboxTrack:
    title: str
    artist: str
    location: str
    path: Path | None
    bpm: float | None
    onset_time: float | None
    cues: list[RekordboxCue]


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _location_to_path(location: str) -> Path | None:
    if not location:
        return None
    parsed = urlparse(location)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser()
    if parsed.scheme:
        return None
    return Path(unquote(location)).expanduser()


def _norm_token(value: str) -> str:
    return Path(str(value).strip()).name.lower()


def _norm_stem(value: str) -> str:
    return Path(str(value).strip()).stem.lower()


def _cue_name(value: str | None) -> str:
    return " ".join(str(value or "").strip().upper().split())


def _cue_role_and_index(name: str) -> tuple[str, str]:
    parts = _cue_name(name).split()
    if not parts:
        return "", ""
    if len(parts) >= 2 and parts[-1].isdigit():
        return " ".join(parts[:-1]).lower(), parts[-1]
    return parts[0].lower(), ""


def parse_rekordbox_xml(xml_path: Path) -> list[RekordboxTrack]:
    root = ET.parse(xml_path).getroot()
    out: list[RekordboxTrack] = []
    for elem in root.findall(".//TRACK"):
        title = elem.attrib.get("Name", "").strip()
        artist = elem.attrib.get("Artist", "").strip()
        location = elem.attrib.get("Location", "").strip()
        path = _location_to_path(location)

        tempos = elem.findall("TEMPO")
        first_tempo = tempos[0] if tempos else None
        bpm = _optional_float(first_tempo.attrib.get("Bpm") if first_tempo is not None else None)
        onset = _optional_float(first_tempo.attrib.get("Inizio") if first_tempo is not None else None)

        cues: list[RekordboxCue] = []
        if onset is not None:
            cues.append(
                RekordboxCue(
                    name="START",
                    start_seconds=float(onset),
                    end_seconds=None,
                    rekordbox_type="beatgrid",
                    rekordbox_num="",
                )
            )
        for mark in elem.findall("POSITION_MARK"):
            name = _cue_name(mark.attrib.get("Name"))
            start = _optional_float(mark.attrib.get("Start"))
            if start is None:
                continue
            cues.append(
                RekordboxCue(
                    name=name or f"CUE {mark.attrib.get('Num', '').strip()}".strip(),
                    start_seconds=float(start),
                    end_seconds=_optional_float(mark.attrib.get("End")),
                    rekordbox_type=mark.attrib.get("Type", "").strip(),
                    rekordbox_num=mark.attrib.get("Num", "").strip(),
                )
            )

        out.append(
            RekordboxTrack(
                title=title,
                artist=artist,
                location=location,
                path=path,
                bpm=bpm,
                onset_time=onset,
                cues=cues,
            )
        )
    return out


def _build_match_index(records: list[RekordboxTrack]) -> dict[str, RekordboxTrack]:
    index: dict[str, RekordboxTrack] = {}
    for record in records:
        candidates: set[str] = set()
        if record.path is not None:
            candidates.add(f"name:{record.path.name.lower()}")
            candidates.add(f"stem:{record.path.stem.lower()}")
            candidates.add(f"path:{str(record.path.expanduser()).lower()}")
        if record.title:
            candidates.add(f"title:{record.title.lower()}")
        for key in candidates:
            index.setdefault(key, record)
    return index


def _match_row(
    row: dict[str, str],
    index: dict[str, RekordboxTrack],
    *,
    tracklist_csv: Path,
) -> RekordboxTrack | None:
    candidates: list[str] = []
    for key in ("filepath", "location"):
        value = (row.get(key) or "").strip()
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = tracklist_csv.parent / path
            candidates.append(f"path:{str(path.resolve()).lower()}")
            candidates.append(f"name:{_norm_token(value)}")
            candidates.append(f"stem:{_norm_stem(value)}")

    mp3_name = (row.get("mp3_name") or row.get("filename") or "").strip()
    if mp3_name:
        path = Path(mp3_name).expanduser()
        if not path.is_absolute():
            path = tracklist_csv.parent / path
        candidates.append(f"path:{str(path.resolve()).lower()}")
        candidates.append(f"name:{_norm_token(mp3_name)}")
        candidates.append(f"stem:{_norm_stem(mp3_name)}")

    title = (row.get("title") or row.get("name") or "").strip()
    if title:
        candidates.append(f"title:{title.lower()}")

    for key in candidates:
        found = index.get(key)
        if found is not None:
            return found
    return None


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def import_rekordbox_cues(
    *,
    rekordbox_xml: Path,
    tracklist_csv: Path = DEFAULT_TRACKLIST,
    output_csv: Path = DEFAULT_OUTPUT,
    cue_output_csv: Path = DEFAULT_CUE_OUTPUT,
    overwrite_timing: bool = False,
) -> tuple[Path, Path, int, int, int]:
    rb_tracks = parse_rekordbox_xml(rekordbox_xml)
    index = _build_match_index(rb_tracks)

    with tracklist_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {tracklist_csv}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    for column in OUTPUT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    matched = 0
    cue_matched = 0
    cue_rows: list[dict[str, str]] = []
    for row in rows:
        record = _match_row(row, index, tracklist_csv=tracklist_csv)
        if record is None:
            continue
        matched += 1
        if record.bpm is not None:
            row["rekordbox_bpm"] = _format_float(record.bpm)
            if overwrite_timing:
                row["bpm"] = _format_float(record.bpm)
        if record.onset_time is not None:
            row["rekordbox_onset_time"] = _format_float(record.onset_time)
            if overwrite_timing:
                row["onset-time"] = _format_float(record.onset_time)
        for cue in record.cues:
            column = CUE_COLUMN_BY_NAME.get(_cue_name(cue.name))
            if column is not None and not row.get(column):
                row[column] = _format_float(cue.start_seconds)
            cue_role, cue_index = _cue_role_and_index(cue.name)
            cue_rows.append(
                {
                    "track_number": str(row.get("track_number") or row.get("#") or ""),
                    "mp3_name": str(row.get("mp3_name") or row.get("filename") or ""),
                    "title": str(row.get("title") or row.get("name") or record.title),
                    "artists": str(row.get("artists") or row.get("artist") or record.artist),
                    "cue_name": cue.name,
                    "cue_role": cue_role,
                    "cue_index": cue_index,
                    "start_seconds": _format_float(cue.start_seconds),
                    "end_seconds": _format_float(cue.end_seconds),
                    "source": "rekordbox",
                    "rekordbox_type": cue.rekordbox_type,
                    "rekordbox_num": cue.rekordbox_num,
                    "rekordbox_location": record.location,
                }
            )
        if record.cues:
            cue_matched += 1
            row["cue_source"] = "rekordbox"
        row["rekordbox_location"] = record.location

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    cue_rows.sort(
        key=lambda r: (
            int(r["track_number"]) if str(r["track_number"]).isdigit() else 999999,
            float(r["start_seconds"]) if r["start_seconds"] else float("inf"),
            r["cue_name"],
        )
    )
    cue_output_csv.parent.mkdir(parents=True, exist_ok=True)
    with cue_output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CUE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(cue_rows)

    return output_csv, cue_output_csv, matched, cue_matched, len(cue_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import Rekordbox XML cues named IN 1/IN 2/OUT 1/OUT 2 into a playlist CSV.",
    )
    parser.add_argument("rekordbox_xml", type=Path)
    parser.add_argument("--tracklist", type=Path, default=DEFAULT_TRACKLIST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cue-output", type=Path, default=DEFAULT_CUE_OUTPUT)
    parser.add_argument(
        "--overwrite-timing",
        action="store_true",
        help="Also replace bpm and onset-time with Rekordbox beatgrid values.",
    )
    args = parser.parse_args(argv)

    output, cue_output, matched, cue_matched, cue_count = import_rekordbox_cues(
        rekordbox_xml=args.rekordbox_xml,
        tracklist_csv=args.tracklist,
        output_csv=args.output,
        cue_output_csv=args.cue_output,
        overwrite_timing=bool(args.overwrite_timing),
    )
    print(f"Wrote enriched CSV: {output}")
    print(f"Wrote long cue CSV: {cue_output}")
    print(f"Matched tracks: {matched}")
    print(f"Tracks with supported cue labels: {cue_matched}")
    print(f"Cue rows: {cue_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
