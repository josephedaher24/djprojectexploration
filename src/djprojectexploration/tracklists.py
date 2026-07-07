"""Shared tracklist loading and path helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PlaylistTrack:
    """Normalized track metadata row from a playlist CSV."""

    track_number: int
    title: str
    artists: str
    mp3_name: str
    audio_path: Path
    genre: str
    key: str
    bpm: float | None
    onset_time: float | None
    key_shift: float | None


def to_project_relpath(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(project_root.expanduser().resolve()))
    except ValueError:
        return str(resolved)


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def norm_token(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def read_csv_rows(path: str | Path, *, missing_ok: bool = False) -> list[dict[str, str]]:
    csv_path = Path(path).expanduser()
    if missing_ok and not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_audio_path(
    *,
    csv_dir: Path,
    music_dir: Path | None,
    mp3_name: str,
    filepath_raw: str,
) -> Path | None:
    if filepath_raw:
        filepath = Path(filepath_raw).expanduser()
        if filepath.is_absolute():
            return filepath
        return csv_dir / filepath

    if not mp3_name:
        return None

    filename_path = Path(mp3_name).expanduser()
    if filename_path.is_absolute():
        return filename_path

    base_dir = music_dir if music_dir is not None else csv_dir
    return base_dir / filename_path


def load_playlist_tracks(
    tracklist_csv: str | Path,
    *,
    music_dir: str | Path | None = None,
    skip_missing_audio: bool = False,
) -> list[PlaylistTrack]:
    """Load and normalize playlist tracks from CSV.

    Supported CSV schemas:
    - Notebook style: track_number,title,artists,mp3_name,key,bpm,onset-time,genre,key shift
    - Apple Music export style: name,artist,album,genre,bpm,filepath

    Track order is sorted by ``track_number`` when present; otherwise row order is
    used with synthetic 1-based numbering.
    """
    csv_path = Path(tracklist_csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Tracklist CSV not found: {csv_path}")

    resolved_music_dir = None
    if music_dir is not None:
        resolved_music_dir = Path(music_dir).expanduser().resolve()

    parsed_rows: list[tuple[int, int, PlaylistTrack]] = []
    skipped_missing: list[tuple[int, str]] = []

    for row_index, row in enumerate(read_csv_rows(csv_path), start=1):
        track_number_raw = (row.get("track_number") or row.get("#") or "").strip()
        if track_number_raw:
            try:
                track_number = int(track_number_raw)
            except ValueError as exc:
                raise ValueError(f"Invalid track_number '{track_number_raw}' at row {row_index} in {csv_path}") from exc
        else:
            track_number = row_index

        title = (row.get("title") or row.get("name") or "").strip()
        artists = (row.get("artists") or row.get("artist") or "").strip()
        genre = (row.get("genre") or "").strip()
        key = (row.get("key") or "").strip()

        bpm = optional_float(row.get("bpm"))
        onset_time = optional_float(row.get("onset-time") or row.get("onset_time") or row.get("onset"))
        key_shift = optional_float(row.get("key shift") or row.get("key_shift"))

        mp3_name = (row.get("mp3_name") or "").strip()
        filepath_raw = (row.get("filepath") or row.get("location") or "").strip()
        audio_path = resolve_audio_path(
            csv_dir=csv_path.parent,
            music_dir=resolved_music_dir,
            mp3_name=mp3_name,
            filepath_raw=filepath_raw,
        )

        if audio_path is None:
            raise ValueError(
                f"Could not resolve audio path at row {row_index} in {csv_path}. "
                "Need either filepath/location or mp3_name (+ music_dir/csv-dir)."
            )

        resolved_audio = audio_path.expanduser().resolve()

        if not mp3_name:
            mp3_name = resolved_audio.name
        if not title:
            title = Path(mp3_name).stem

        if not resolved_audio.exists():
            if skip_missing_audio:
                skipped_missing.append((row_index, str(resolved_audio)))
                continue
            raise FileNotFoundError(
                f"Audio file not found for row {row_index}: {resolved_audio}. "
                "Use --skip-missing-audio to continue without this track."
            )

        track = PlaylistTrack(
            track_number=track_number,
            title=title,
            artists=artists,
            mp3_name=mp3_name,
            audio_path=resolved_audio,
            genre=genre,
            key=key,
            bpm=bpm,
            onset_time=onset_time,
            key_shift=key_shift,
        )
        parsed_rows.append((track_number, row_index, track))

    if not parsed_rows:
        raise RuntimeError(
            f"No usable tracks found in {csv_path}."
            + (" All rows were missing audio files." if skipped_missing else "")
        )

    parsed_rows.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in parsed_rows]
