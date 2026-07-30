"""Tracklist validation helpers for dataset builds."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from djprojectexploration.tracklists import load_playlist_tracks, read_csv_rows


@dataclass(frozen=True)
class TracklistValidation:
    tracklist: str
    rows: int
    resolved_audio: int
    missing_audio: int
    duplicate_filenames: int
    duplicate_track_numbers: int
    missing_title: int
    missing_artist: int
    missing_genre: int
    missing_key: int
    missing_bpm: int
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _present(row: dict[str, str], *keys: str) -> bool:
    return any((row.get(key) or "").strip() for key in keys)


def validate_tracklist(
    tracklist_csv: str | Path,
    *,
    music_dir: str | Path | None = None,
    skip_missing_audio: bool = False,
) -> TracklistValidation:
    path = Path(tracklist_csv).expanduser().resolve()
    rows = read_csv_rows(path)
    warnings: list[str] = []

    filenames: list[str] = []
    track_numbers: list[str] = []
    missing_title = missing_artist = missing_genre = missing_key = missing_bpm = 0
    for row in rows:
        filename = (row.get("filename") or row.get("mp3_name") or Path(row.get("filepath") or "").name).strip()
        if filename:
            filenames.append(filename.lower())
        track_number = (row.get("track_number") or row.get("#") or "").strip()
        if track_number:
            track_numbers.append(track_number)
        if not _present(row, "title", "name"):
            missing_title += 1
        if not _present(row, "artists", "artist"):
            missing_artist += 1
        if not _present(row, "genre"):
            missing_genre += 1
        if not _present(row, "key"):
            missing_key += 1
        if not _present(row, "bpm"):
            missing_bpm += 1

    try:
        tracks = load_playlist_tracks(path, music_dir=music_dir, skip_missing_audio=skip_missing_audio)
        resolved_audio = len(tracks)
    except Exception as exc:
        resolved_audio = 0
        warnings.append(str(exc))

    duplicate_filenames = len(filenames) - len(set(filenames))
    duplicate_track_numbers = len(track_numbers) - len(set(track_numbers))
    missing_audio = max(0, len(rows) - resolved_audio)
    if missing_audio:
        warnings.append(f"{missing_audio} row(s) did not resolve to audio files.")
    if duplicate_filenames:
        warnings.append(f"{duplicate_filenames} duplicate filename token(s).")
    if duplicate_track_numbers:
        warnings.append(f"{duplicate_track_numbers} duplicate track number(s).")

    return TracklistValidation(
        tracklist=str(path),
        rows=len(rows),
        resolved_audio=resolved_audio,
        missing_audio=missing_audio,
        duplicate_filenames=duplicate_filenames,
        duplicate_track_numbers=duplicate_track_numbers,
        missing_title=missing_title,
        missing_artist=missing_artist,
        missing_genre=missing_genre,
        missing_key=missing_key,
        missing_bpm=missing_bpm,
        warnings=warnings,
    )


def write_validation_json(validation: TracklistValidation, output_file: str | Path) -> Path:
    path = Path(output_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validation.to_dict(), indent=2), encoding="utf-8")
    return path
