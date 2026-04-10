"""Audio snippet caching utilities for lightweight sharing and playback."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNIPPET_SECONDS = 8.0
DEFAULT_MIDDLE_FRACTION = 0.66
DEFAULT_SCAN_HOP_SECONDS = 0.25
DEFAULT_TARGET_SAMPLE_RATE = 22050


@dataclass(frozen=True)
class SnippetInfo:
    """Resolved snippet metadata for one source track."""

    source_audio_path: Path
    snippet_path: Path
    snippet_src: str
    start_seconds: float
    end_seconds: float
    rms: float


def _to_project_relpath(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _sanitize_slug(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "track"


def _meta_path(snippet_path: Path) -> Path:
    return snippet_path.with_suffix(snippet_path.suffix + ".json")


def _default_snippet_name(audio_path: Path, key: str | None) -> str:
    key_text = key or audio_path.stem
    stem = _sanitize_slug(Path(str(key_text)).stem)
    if len(stem) > 80:
        stem = stem[:80].rstrip("._-")
    if not stem:
        stem = "track"
    digest = hashlib.sha1(str(audio_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}.wav"


def _audio_src_for_notebook(snippet_path: Path, project_root: Path) -> str:
    """Build a notebook/browser-friendly path under Jupyter's /files handler."""
    rel = snippet_path.resolve().relative_to(project_root.resolve())
    return f"/files/{rel.as_posix()}"


def _window_rms(power_cumsum: np.ndarray, start: np.ndarray, win_size: int) -> np.ndarray:
    energy = power_cumsum[start + win_size] - power_cumsum[start]
    return np.sqrt(np.maximum(energy / float(win_size), 0.0))


def _resample_if_needed(audio: np.ndarray, src_sr: int, target_sr: int | None) -> tuple[np.ndarray, int]:
    if target_sr is None or target_sr <= 0 or target_sr == src_sr:
        return audio, int(src_sr)
    if audio.size == 0:
        return audio, int(target_sr)

    g = math.gcd(int(src_sr), int(target_sr))
    up = int(target_sr // g)
    down = int(src_sr // g)
    resampled = signal.resample_poly(audio.astype(np.float32), up, down).astype(np.float32)
    return resampled, int(target_sr)


def select_rms_focused_window(
    audio: np.ndarray,
    sample_rate: int,
    *,
    snippet_seconds: float = DEFAULT_SNIPPET_SECONDS,
    middle_fraction: float = DEFAULT_MIDDLE_FRACTION,
    hop_seconds: float = DEFAULT_SCAN_HOP_SECONDS,
) -> tuple[int, int, float]:
    """Pick a high-RMS snippet window, constrained to the middle portion of a track."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    n = int(audio.shape[0])
    if n < 2 or sample_rate <= 0:
        return 0, 0, 0.0

    middle_fraction = float(np.clip(middle_fraction, 0.1, 1.0))
    snippet_seconds = float(max(0.5, snippet_seconds))
    hop_seconds = float(max(0.01, hop_seconds))

    win_size = int(round(snippet_seconds * sample_rate))
    if win_size <= 1:
        win_size = 1
    if win_size >= n:
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        return 0, n, rms

    margin_fraction = (1.0 - middle_fraction) * 0.5
    region_start = int(round(n * margin_fraction))
    region_end = int(round(n * (1.0 - margin_fraction)))

    max_start = n - win_size
    low = int(np.clip(region_start, 0, max_start))
    high = int(np.clip(region_end - win_size, low, max_start))
    if high < low:
        center_start = max(0, (n - win_size) // 2)
        power = np.square(audio, dtype=np.float64)
        csum = np.concatenate(([0.0], np.cumsum(power)))
        rms = float(_window_rms(csum, np.asarray([center_start]), win_size)[0])
        return center_start, center_start + win_size, rms

    step = max(1, int(round(hop_seconds * sample_rate)))
    starts = np.arange(low, high + 1, step, dtype=np.int64)
    if starts.size == 0 or starts[-1] != high:
        starts = np.append(starts, high)

    power = np.square(audio, dtype=np.float64)
    csum = np.concatenate(([0.0], np.cumsum(power)))
    rms_vals = _window_rms(csum, starts, win_size)
    if rms_vals.size == 0:
        center_start = max(0, (n - win_size) // 2)
        return center_start, center_start + win_size, 0.0

    max_rms = float(np.max(rms_vals))
    candidate_idx = np.flatnonzero(np.isclose(rms_vals, max_rms, rtol=1e-6, atol=1e-9))
    if candidate_idx.size <= 1:
        best_i = int(np.argmax(rms_vals))
    else:
        mid = n * 0.5
        centers = starts[candidate_idx] + (win_size * 0.5)
        best_i = int(candidate_idx[np.argmin(np.abs(centers - mid))])

    start = int(starts[best_i])
    end = int(start + win_size)
    return start, end, float(rms_vals[best_i])


def ensure_cached_snippet(
    *,
    audio_path: Path | None,
    output_dir: Path,
    key: str | None = None,
    snippet_seconds: float = DEFAULT_SNIPPET_SECONDS,
    middle_fraction: float = DEFAULT_MIDDLE_FRACTION,
    hop_seconds: float = DEFAULT_SCAN_HOP_SECONDS,
    target_sample_rate: int | None = DEFAULT_TARGET_SAMPLE_RATE,
    overwrite: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> SnippetInfo | None:
    """Create or reuse a cached snippet and return metadata for notebook playback."""
    if audio_path is None:
        return None
    source = Path(audio_path).expanduser()
    if not source.exists():
        return None

    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    snippet_path = output_root / _default_snippet_name(source, key=key)
    meta_path = _meta_path(snippet_path)

    if snippet_path.exists() and meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return SnippetInfo(
            source_audio_path=source.resolve(),
            snippet_path=snippet_path.resolve(),
            snippet_src=_audio_src_for_notebook(snippet_path.resolve(), project_root=project_root),
            start_seconds=float(meta.get("start_seconds", 0.0)),
            end_seconds=float(meta.get("end_seconds", 0.0)),
            rms=float(meta.get("rms", 0.0)),
        )

    audio, sr = sf.read(str(source), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < 2 or sr <= 0:
        return None

    start_idx, end_idx, rms = select_rms_focused_window(
        audio,
        sr,
        snippet_seconds=snippet_seconds,
        middle_fraction=middle_fraction,
        hop_seconds=hop_seconds,
    )
    if end_idx <= start_idx:
        return None

    snippet = audio[start_idx:end_idx]
    if snippet.size == 0:
        return None

    snippet_out, sr_out = _resample_if_needed(snippet, int(sr), target_sample_rate)
    if snippet_out.size == 0:
        return None

    sf.write(str(snippet_path), snippet_out, sr_out, format="WAV", subtype="PCM_16")
    start_seconds = start_idx / float(sr)
    end_seconds = end_idx / float(sr)
    metadata = {
        "source_audio_path": str(source.resolve()),
        "snippet_path": str(snippet_path.resolve()),
        "start_seconds": float(start_seconds),
        "end_seconds": float(end_seconds),
        "duration_seconds": float(end_seconds - start_seconds),
        "rms": float(rms),
        "sample_rate": int(sr_out),
        "snippet_seconds_requested": float(snippet_seconds),
        "middle_fraction": float(middle_fraction),
        "hop_seconds": float(hop_seconds),
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return SnippetInfo(
        source_audio_path=source.resolve(),
        snippet_path=snippet_path.resolve(),
        snippet_src=_audio_src_for_notebook(snippet_path.resolve(), project_root=project_root),
        start_seconds=float(start_seconds),
        end_seconds=float(end_seconds),
        rms=float(rms),
    )


def _default_output_dir(tracklist_csv: Path) -> Path:
    return PROJECT_ROOT / "data" / "snippets" / tracklist_csv.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache per-track audio snippets using high RMS selection in the middle 66% of each track.",
    )
    parser.add_argument("tracklist_csv", type=Path, help="Path to playlist CSV.")
    parser.add_argument(
        "--music-dir",
        type=Path,
        default=None,
        help="Optional fallback music directory when CSV lacks `filepath`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output snippet directory. Defaults to data/snippets/<csv-stem>/",
    )
    parser.add_argument("--snippet-seconds", type=float, default=DEFAULT_SNIPPET_SECONDS)
    parser.add_argument("--middle-fraction", type=float, default=DEFAULT_MIDDLE_FRACTION)
    parser.add_argument("--hop-seconds", type=float, default=DEFAULT_SCAN_HOP_SECONDS)
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=DEFAULT_TARGET_SAMPLE_RATE,
        help=f"Resample output snippets to this sample rate (default: {DEFAULT_TARGET_SAMPLE_RATE}).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute snippets even if cached.")
    parser.add_argument(
        "--skip-missing-audio",
        action="store_true",
        help="Skip rows whose audio files are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from djprojectexploration.playlist_embedding_pipeline import load_playlist_tracks

    csv_path = args.tracklist_csv.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(csv_path)
    )
    tracks = load_playlist_tracks(
        csv_path,
        music_dir=args.music_dir,
        skip_missing_audio=args.skip_missing_audio,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "snippets_manifest.csv"
    fieldnames = [
        "track_number",
        "title",
        "artists",
        "mp3_name",
        "audio_path",
        "snippet_path",
        "snippet_src",
        "snippet_start",
        "snippet_end",
        "snippet_rms",
    ]

    created = 0
    missing = 0
    write_errors = 0
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for track in tracks:
            try:
                info = ensure_cached_snippet(
                    audio_path=track.audio_path,
                    output_dir=output_dir,
                    key=track.mp3_name,
                    snippet_seconds=args.snippet_seconds,
                    middle_fraction=args.middle_fraction,
                    hop_seconds=args.hop_seconds,
                    target_sample_rate=args.target_sample_rate,
                    overwrite=args.overwrite,
                    project_root=PROJECT_ROOT,
                )
            except Exception as exc:
                write_errors += 1
                free_bytes = shutil.disk_usage(output_dir).free
                print(
                    f"Warning: snippet write failed for '{track.mp3_name}' "
                    f"({type(exc).__name__}: {exc}). Free disk: {free_bytes / (1024 * 1024):.1f} MiB",
                )
                continue
            if info is None:
                missing += 1
                continue
            created += 1
            writer.writerow(
                {
                    "track_number": track.track_number,
                    "title": track.title,
                    "artists": track.artists,
                    "mp3_name": track.mp3_name,
                    "audio_path": _to_project_relpath(track.audio_path),
                    "snippet_path": _to_project_relpath(info.snippet_path),
                    "snippet_src": info.snippet_src,
                    "snippet_start": f"{info.start_seconds:.3f}",
                    "snippet_end": f"{info.end_seconds:.3f}",
                    "snippet_rms": f"{info.rms:.6f}",
                }
            )

    print(f"Snippet cache dir: {_to_project_relpath(output_dir)}")
    print(f"Snippet manifest: {_to_project_relpath(manifest_path)}")
    print(f"Prepared snippets: {created}/{len(tracks)} tracks")
    if missing:
        print(f"Skipped (missing/unreadable audio): {missing}")
    if write_errors:
        print(f"Skipped (write errors): {write_errors}")


if __name__ == "__main__":
    main()
