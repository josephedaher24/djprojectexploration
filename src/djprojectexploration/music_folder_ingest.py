"""Create normalized tracklist CSVs from folders of audio files."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from djprojectexploration.tracklists import PROJECT_ROOT, load_playlist_tracks, optional_float


AUDIO_EXTENSIONS = {
    ".aif",
    ".aiff",
    ".alac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}
TRACKLIST_COLUMNS = [
    "track_number",
    "title",
    "artists",
    "mp3_name",
    "filepath",
    "key",
    "bpm",
    "onset-time",
    "genre",
    "key shift",
    "energy",
]


@dataclass(frozen=True)
class IngestedTrack:
    track_number: int
    title: str
    artists: str
    mp3_name: str
    filepath: str
    key: str
    bpm: float | None
    genre: str


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return text.strip("-") or "music-library"


def _snake(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return text.strip("_") or "music_library"


def _audio_files(directory: Path, *, recursive: bool) -> list[Path]:
    iterator: Iterable[Path] = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS),
        key=lambda path: path.relative_to(directory).as_posix().lower(),
    )


def _tag_text(audio: Any, keys: tuple[str, ...]) -> str:
    if audio is None or getattr(audio, "tags", None) is None:
        return ""
    tags = audio.tags
    for key in keys:
        try:
            value = tags.get(key)
        except AttributeError:
            value = None
        if value is None:
            continue
        if isinstance(value, list):
            value = value[0] if value else ""
        text = str(value).strip()
        if text:
            return text
    return ""


def _bpm_from_tags(audio: Any) -> float | None:
    text = _tag_text(audio, ("bpm", "BPM", "TBPM"))
    if not text:
        return None
    return optional_float(text.split("/", 1)[0])


def _key_from_tags(audio: Any) -> str:
    return _tag_text(audio, ("initialkey", "INITIALKEY", "TKEY", "key", "KEY"))


def _load_tags(path: Path) -> Any:
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        return None

    try:
        return MutagenFile(str(path), easy=True)
    except Exception:
        return None


def _display_path(path: Path, *, relative_to: Path | None) -> str:
    if relative_to is not None:
        try:
            return path.resolve().relative_to(relative_to.resolve()).as_posix()
        except ValueError:
            pass
    return str(path.resolve())


def ingest_music_folder(
    music_dir: str | Path,
    *,
    output_file: str | Path | None = None,
    name: str | None = None,
    recursive: bool = True,
    relative_filepaths: bool = True,
    overwrite: bool = False,
) -> Path:
    """Scan a folder of audio files and write a normalized tracklist CSV."""
    source_dir = Path(music_dir).expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise NotADirectoryError(f"Music folder not found: {source_dir}")

    library_name = _slug(name or source_dir.name)
    if output_file is None:
        output_path = source_dir / f"{_snake(library_name)}_tracks.csv"
    else:
        output_path = Path(output_file).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")

    files = _audio_files(source_dir, recursive=recursive)
    if not files:
        raise RuntimeError(f"No supported audio files found in {source_dir}")

    filepath_base = output_path.parent if relative_filepaths else None
    tracks: list[IngestedTrack] = []
    for index, path in enumerate(files, start=1):
        audio = _load_tags(path)
        title = _tag_text(audio, ("title", "TITLE")) or path.stem
        artists = _tag_text(audio, ("artist", "ARTIST", "albumartist", "ALBUMARTIST"))
        genre = _tag_text(audio, ("genre", "GENRE"))
        key = _key_from_tags(audio)
        bpm = _bpm_from_tags(audio)
        rel_from_source = path.relative_to(source_dir).as_posix()
        tracks.append(
            IngestedTrack(
                track_number=index,
                title=title,
                artists=artists,
                mp3_name=rel_from_source,
                filepath=_display_path(path, relative_to=filepath_base),
                key=key,
                bpm=bpm,
                genre=genre,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACKLIST_COLUMNS)
        writer.writeheader()
        for track in tracks:
            writer.writerow(
                {
                    "track_number": track.track_number,
                    "title": track.title,
                    "artists": track.artists,
                    "mp3_name": track.mp3_name,
                    "filepath": track.filepath,
                    "key": track.key,
                    "bpm": "" if track.bpm is None else f"{track.bpm:g}",
                    "onset-time": "",
                    "genre": track.genre,
                    "key shift": "",
                    "energy": "",
                }
            )

    # Validate that the generated CSV is immediately accepted by the shared loader.
    load_playlist_tracks(output_path, skip_missing_audio=False)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a normalized tracklist CSV from a folder of audio files.")
    parser.add_argument("music_dir", type=Path, help="Folder containing audio files.")
    parser.add_argument("--name", default=None, help="Library/mix name used for the default output filename.")
    parser.add_argument("--output-file", type=Path, default=None, help="Exact output CSV path.")
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Only scan files directly inside music_dir; by default subdirectories are scanned recursively.",
    )
    parser.add_argument(
        "--absolute-filepaths",
        action="store_true",
        help="Write absolute filepath values instead of paths relative to the output CSV directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output CSV.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = ingest_music_folder(
        args.music_dir,
        output_file=args.output_file,
        name=args.name,
        recursive=not bool(args.flat),
        relative_filepaths=not bool(args.absolute_filepaths),
        overwrite=bool(args.overwrite),
    )
    rel = output_path
    try:
        rel = output_path.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    print(f"Saved normalized tracklist: {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
