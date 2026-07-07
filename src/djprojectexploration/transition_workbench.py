"""Local web workbench for rendering and previewing track transitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import mimetypes
import sys
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from djprojectexploration.playlist_embedding_pipeline import PROJECT_ROOT, load_playlist_tracks
from djprojectexploration.transition_preview import (
    DEFAULT_OUTPUT_DIR,
    EQ_MODES,
    FILTER_MODES,
    TRANSITION_PRESETS,
    VOLUME_MODES,
    RenderedTransition,
    render_transition,
)
from djprojectexploration.transition_visualizer import export_transition_visualizer


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_TRACKLISTS = [
    PROJECT_ROOT / "music" / "aries-mix" / "aries_mix_tracks.csv",
    PROJECT_ROOT / "music" / "ara-mix" / "ara_mix_tracks.csv",
]


@dataclass(frozen=True)
class TrackSource:
    slug: str
    label: str
    tracklist: Path
    cue_table: Path


def _default_cue_table(tracklist: Path) -> Path:
    stem = tracklist.stem
    if stem.endswith("_tracks_rekordbox"):
        stem = stem[: -len("_tracks_rekordbox")]
    elif stem.endswith("_tracks"):
        stem = stem[: -len("_tracks")]
    return tracklist.with_name(f"{stem}_cues.csv")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _text_response(handler: BaseHTTPRequestHandler, status: int, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
    data = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _error_response(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    _json_response(handler, status, {"ok": False, "error": message})


def _parse_range_header(value: str, file_size: int) -> tuple[int, int] | None:
    if not value or not value.startswith("bytes=") or file_size <= 0:
        return None
    range_text = value[len("bytes=") :].split(",", 1)[0].strip()
    if "-" not in range_text:
        return None
    start_text, end_text = range_text.split("-", 1)
    try:
        if start_text == "":
            suffix = int(end_text)
            if suffix <= 0:
                return None
            start = max(0, file_size - suffix)
            end = file_size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
    except ValueError:
        return None
    if start < 0 or start >= file_size or end < start:
        return None
    return start, min(end, file_size - 1)


def _file_response(handler: BaseHTTPRequestHandler, path: Path, content_type: str) -> None:
    file_size = path.stat().st_size
    byte_range = _parse_range_header(handler.headers.get("Range", ""), file_size)
    try:
        if byte_range is None:
            data = path.read_bytes()
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(data)))
            handler.send_header("Accept-Ranges", "bytes")
            handler.end_headers()
            handler.wfile.write(data)
            return

        start, end = byte_range
        length = end - start + 1
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(length)
        handler.send_response(HTTPStatus.PARTIAL_CONTENT)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        handler.send_header("Accept-Ranges", "bytes")
        handler.end_headers()
        handler.wfile.write(data)
    except BrokenPipeError:
        pass


def _optional_str(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def _optional_mode(value: Any) -> str | None:
    text = _optional_str(value)
    if text in {None, "preset"}:
        return None
    return text


def _source_slug(tracklist: Path) -> str:
    parent = tracklist.expanduser().resolve().parent.name
    return parent or tracklist.stem


def _source_label(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


def _dedupe_sources(tracklists: list[Path], cue_tables: list[Path | None]) -> list[TrackSource]:
    counts: dict[str, int] = {}
    sources: list[TrackSource] = []
    for index, tracklist in enumerate(tracklists):
        resolved_tracklist = tracklist.expanduser().resolve()
        cue_table = cue_tables[index] if index < len(cue_tables) else None
        resolved_cue_table = (cue_table or _default_cue_table(resolved_tracklist)).expanduser().resolve()
        base_slug = _source_slug(resolved_tracklist)
        seen_count = counts.get(base_slug, 0)
        counts[base_slug] = seen_count + 1
        slug = base_slug if seen_count == 0 else f"{base_slug}-{seen_count + 1}"
        sources.append(
            TrackSource(
                slug=slug,
                label=_source_label(slug),
                tracklist=resolved_tracklist,
                cue_table=resolved_cue_table,
            )
        )
    return sources


class TransitionWorkbench:
    def __init__(
        self,
        *,
        tracklists: list[Path],
        cue_tables: list[Path | None] | None,
        output_dir: Path,
    ) -> None:
        if not tracklists:
            raise ValueError("At least one tracklist is required.")
        self.sources = _dedupe_sources(tracklists, cue_tables or [])
        self.sources_by_slug = {source.slug: source for source in self.sources}
        self.output_dir = output_dir.expanduser().resolve()
        self.generated_source_dir = self.output_dir / "_workbench_sources"

    def _cues_by_track(self, source: TrackSource) -> dict[int, list[dict[str, Any]]]:
        cue_rows = _read_csv_rows(source.cue_table)
        cues_by_track: dict[int, list[dict[str, Any]]] = {}
        seen_cues: set[tuple[int, str, str, str]] = set()
        for row in cue_rows:
            try:
                track_number = int(str(row.get("track_number", "")).strip())
            except ValueError:
                continue
            start = str(row.get("start_seconds", "")).strip()
            cue_name = str(row.get("cue_name", "")).strip()
            cue_role = str(row.get("cue_role", "")).strip()
            key = (track_number, cue_name.upper(), cue_role.lower(), start)
            if key in seen_cues:
                continue
            seen_cues.add(key)
            cues_by_track.setdefault(track_number, []).append(
                {
                    "name": cue_name,
                    "role": cue_role,
                    "index": row.get("cue_index", ""),
                    "start_seconds": float(start) if start else None,
                }
            )
        return cues_by_track

    def tracks_payload(self) -> dict[str, Any]:
        tracks = []
        for source in self.sources:
            cues_by_track = self._cues_by_track(source)
            for track in load_playlist_tracks(source.tracklist):
                tracks.append(
                    {
                        "id": f"{source.slug}:{track.track_number}",
                        "source_slug": source.slug,
                        "source_label": source.label,
                        "track_number": track.track_number,
                        "title": track.title,
                        "artists": track.artists,
                        "filename": track.mp3_name,
                        "key": track.key,
                        "bpm": track.bpm,
                        "onset_time": track.onset_time,
                        "cues": cues_by_track.get(track.track_number, []),
                    }
                )
        return {
            "ok": True,
            "sources": [
                {
                    "slug": source.slug,
                    "label": source.label,
                    "tracklist": str(source.tracklist),
                    "cue_table": str(source.cue_table),
                }
                for source in self.sources
            ],
            "tracks": tracks,
        }

    def options_payload(self) -> dict[str, Any]:
        return {
            "ok": True,
            "presets": sorted(TRANSITION_PRESETS),
            "preset_details": TRANSITION_PRESETS,
            "volume_modes": sorted(VOLUME_MODES),
            "eq_modes": sorted(EQ_MODES),
            "filter_modes": sorted(FILTER_MODES),
        }

    def _resolve_track_ref(self, value: str) -> tuple[TrackSource, Any]:
        if ":" not in value:
            raise ValueError(f"Track id must include a source prefix: {value}")
        source_slug, track_number_text = value.split(":", 1)
        source = self.sources_by_slug.get(source_slug)
        if source is None:
            raise ValueError(f"Unknown track source: {source_slug}")
        try:
            track_number = int(track_number_text)
        except ValueError as exc:
            raise ValueError(f"Invalid track number in track id: {value}") from exc
        matches = [track for track in load_playlist_tracks(source.tracklist) if track.track_number == track_number]
        if len(matches) != 1:
            raise ValueError(f"Could not resolve track id: {value}")
        return source, matches[0]

    def _source_cue_rows_for_track(
        self,
        *,
        source: TrackSource,
        original_track_number: int,
        rendered_track_number: int,
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for row in _read_csv_rows(source.cue_table):
            try:
                track_number = int(str(row.get("track_number", "")).strip())
            except ValueError:
                continue
            if track_number != original_track_number:
                continue
            next_row = dict(row)
            next_row["track_number"] = str(rendered_track_number)
            rows.append(next_row)
        return rows

    def _default_cue_name(self, *, source: TrackSource, track_number: int, role: str) -> str | None:
        rows = self._source_cue_rows_for_track(
            source=source,
            original_track_number=track_number,
            rendered_track_number=track_number,
        )
        role_normalized = role.strip().lower()
        for row in rows:
            if str(row.get("cue_role", "")).strip().lower() == role_normalized:
                cue_name = str(row.get("cue_name", "")).strip()
                if cue_name:
                    return cue_name
        for row in rows:
            cue_name = str(row.get("cue_name", "")).strip()
            if cue_name:
                return cue_name
        return None

    def _write_render_sources(self, from_id: str, to_id: str) -> tuple[Path, Path, str, str, str | None, str | None]:
        from_source, from_track = self._resolve_track_ref(from_id)
        to_source, to_track = self._resolve_track_ref(to_id)
        digest = hashlib.sha1(f"{from_id}|{to_id}".encode("utf-8")).hexdigest()[:12]
        stem = f"workbench_{digest}"
        tracklist = self.generated_source_dir / f"{stem}_tracks.csv"
        cue_table = self.generated_source_dir / f"{stem}_cues.csv"
        self.generated_source_dir.mkdir(parents=True, exist_ok=True)

        fieldnames = [
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
            "source_mix",
            "source_track_number",
        ]
        with tracklist.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rendered_track_number, source, track in (
                (1, from_source, from_track),
                (2, to_source, to_track),
            ):
                writer.writerow(
                    {
                        "track_number": rendered_track_number,
                        "title": track.title,
                        "artists": track.artists,
                        "mp3_name": track.mp3_name,
                        "filepath": str(track.audio_path),
                        "key": track.key,
                        "bpm": "" if track.bpm is None else f"{float(track.bpm):.6f}".rstrip("0").rstrip("."),
                        "onset-time": ""
                        if track.onset_time is None
                        else f"{float(track.onset_time):.6f}".rstrip("0").rstrip("."),
                        "genre": track.genre,
                        "key shift": "" if track.key_shift is None else str(track.key_shift),
                        "source_mix": source.slug,
                        "source_track_number": track.track_number,
                    }
                )

        cue_rows = self._source_cue_rows_for_track(
            source=from_source,
            original_track_number=from_track.track_number,
            rendered_track_number=1,
        ) + self._source_cue_rows_for_track(
            source=to_source,
            original_track_number=to_track.track_number,
            rendered_track_number=2,
        )
        cue_fieldnames: list[str] = []
        for row in cue_rows:
            for key in row:
                if key not in cue_fieldnames:
                    cue_fieldnames.append(key)
        if not cue_fieldnames:
            cue_fieldnames = ["track_number", "cue_name", "cue_role", "cue_index", "start_seconds"]
        with cue_table.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cue_fieldnames)
            writer.writeheader()
            writer.writerows(cue_rows)

        return (
            tracklist,
            cue_table,
            "1",
            "2",
            self._default_cue_name(source=from_source, track_number=from_track.track_number, role="out"),
            self._default_cue_name(source=to_source, track_number=to_track.track_number, role="in"),
        )

    def render_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        from_track = _optional_str(body.get("from_track"))
        to_track = _optional_str(body.get("to_track"))
        if not from_track or not to_track:
            raise ValueError("Both from_track and to_track are required.")
        render_tracklist, render_cue_table, from_query, to_query, default_from_cue, default_to_cue = self._write_render_sources(
            from_track,
            to_track,
        )

        result = render_transition(
            tracklist_csv=render_tracklist,
            cue_table=render_cue_table,
            output_dir=self.output_dir,
            from_query=from_query,
            to_query=to_query,
            from_cue=_optional_str(body.get("from_cue")) or default_from_cue,
            to_cue=_optional_str(body.get("to_cue")) or default_to_cue,
            overlap_bars=int(body.get("overlap_bars") or 8),
            front_padding_bars=int(body.get("front_padding_bars") or 0),
            back_padding_bars=int(body.get("back_padding_bars") or 0),
            from_nudge_beats=float(body.get("from_nudge_beats") or 0.0),
            to_nudge_beats=float(body.get("to_nudge_beats") or 0.0),
            from_pitch_shift=int(body.get("from_pitch_shift") or 0),
            to_pitch_shift=int(body.get("to_pitch_shift") or 0),
            preset=str(body.get("preset") or "auto"),
            volume_mode=_optional_mode(body.get("volume_mode")),
            eq_mode=_optional_mode(body.get("eq_mode")),
            filter_mode=_optional_mode(body.get("filter_mode")),
            overwrite=bool(body.get("overwrite", True)),
        )
        visualizer = export_transition_visualizer(transition_dir=result.preview_path.parent)
        return {
            "ok": True,
            "transition": _transition_payload(result),
            "paths": {
                "visualizer": str(visualizer),
                "preview": str(result.preview_path),
                "metadata": str(result.metadata_path),
                "waveform": str(result.waveform_path),
            },
            "urls": {
                "visualizer": _artifact_url(result.preview_path.parent.name, visualizer.name),
                "preview": _artifact_url(result.preview_path.parent.name, result.preview_path.name),
                "metadata": _artifact_url(result.preview_path.parent.name, result.metadata_path.name),
                "waveform": _artifact_url(result.preview_path.parent.name, result.waveform_path.name),
            },
        }

    def artifact_path(self, transition_id: str, filename: str) -> Path | None:
        safe_id = Path(transition_id).name
        safe_file = Path(filename).name
        path = (self.output_dir / safe_id / safe_file).resolve()
        try:
            path.relative_to(self.output_dir)
        except ValueError:
            return None
        if not path.exists() or not path.is_file():
            return None
        return path


def _transition_payload(result: RenderedTransition) -> dict[str, Any]:
    data = asdict(result)
    return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


def _artifact_url(transition_id: str, filename: str) -> str:
    return f"/artifacts/{transition_id}/{filename}"


def _workbench_html() -> str:
    presets = html.escape(", ".join(sorted(TRANSITION_PRESETS)))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transition Workbench</title>
<style>
  :root {{
    color-scheme: dark;
    --bg: #101010;
    --surface: #1d1d1d;
    --surface-2: #242424;
    --line: #363636;
    --text: #f2f2f2;
    --muted: #a7a7a7;
    --blue: #48c8f2;
    --yellow: #ffd13d;
    --accent: #61e8d2;
    --danger: #ff8a68;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  main {{
    min-height: 100vh;
    display: grid;
    grid-template-columns: minmax(320px, 380px) minmax(0, 1fr);
  }}
  aside {{
    border-right: 1px solid var(--line);
    background: #151515;
    padding: 16px;
    overflow: auto;
  }}
  section {{
    min-width: 0;
    display: grid;
    grid-template-rows: auto 1fr;
  }}
  h1 {{
    margin: 0 0 14px;
    font-size: 22px;
    line-height: 1.15;
    font-weight: 700;
  }}
  .grid {{
    display: grid;
    gap: 12px;
  }}
  .row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }}
  label {{
    display: grid;
    gap: 5px;
    color: var(--muted);
    font-size: 12px;
  }}
  select, input {{
    width: 100%;
    min-height: 36px;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: var(--surface);
    color: var(--text);
    padding: 7px 8px;
    font: inherit;
    font-size: 13px;
  }}
  input[type="number"] {{ font-variant-numeric: tabular-nums; }}
  button {{
    min-height: 38px;
    border: 0;
    border-radius: 4px;
    background: var(--text);
    color: #111;
    padding: 8px 12px;
    font: inherit;
    cursor: pointer;
  }}
  button.secondary {{
    border: 1px solid var(--line);
    background: var(--surface-2);
    color: var(--text);
  }}
  button:disabled {{
    opacity: .55;
    cursor: wait;
  }}
  .divider {{
    height: 1px;
    background: var(--line);
    margin: 4px 0;
  }}
  .status {{
    min-height: 38px;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: var(--surface);
    color: var(--muted);
    padding: 9px 10px;
    font-size: 12px;
    line-height: 1.4;
  }}
  .status.error {{
    border-color: color-mix(in srgb, var(--danger), var(--line) 45%);
    color: var(--danger);
  }}
  .topbar {{
    border-bottom: 1px solid var(--line);
    background: var(--surface);
    padding: 12px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    justify-content: space-between;
    min-width: 0;
  }}
  .meta {{
    min-width: 0;
    color: var(--muted);
    font-size: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .links {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  a {{
    color: var(--accent);
    text-decoration: none;
    font-size: 12px;
  }}
  #preview {{
    position: relative;
    min-height: 100%;
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    background: #111;
  }}
  .preview-toolbar {{
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
    padding: 10px;
    border-bottom: 1px solid var(--line);
    background: var(--surface-2);
  }}
  .preview-toolbar audio {{ flex: 1 1 420px; min-width: 260px; }}
  .canvas-wrap {{
    position: relative;
    min-height: 520px;
  }}
  .canvas-wrap canvas {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }}
  .empty {{
    display: grid;
    place-items: center;
    color: var(--muted);
    font-size: 13px;
    min-height: 100%;
  }}
  @media (max-width: 900px) {{
    main {{ grid-template-columns: 1fr; }}
    aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
    section {{ min-height: 720px; }}
  }}
</style>
</head>
<body>
<main>
  <aside>
    <h1>Transition Workbench</h1>
    <div class="grid">
      <label>Track 1<select id="from-track"></select></label>
      <label>Track 1 cue<select id="from-cue"></select></label>
      <label>Track 2<select id="to-track"></select></label>
      <label>Track 2 cue<select id="to-cue"></select></label>
      <div class="divider"></div>
      <div class="row">
        <label>Overlap bars<input id="overlap-bars" type="number" min="1" max="64" step="1" value="16"></label>
        <label>Padding bars<input id="padding-bars" type="number" min="0" max="16" step="1" value="2"></label>
      </div>
      <div class="row">
        <label>Front padding<input id="front-padding-bars" type="number" min="0" max="16" step="1" value="2"></label>
        <label>Back padding<input id="back-padding-bars" type="number" min="0" max="16" step="1" value="2"></label>
      </div>
      <div class="row">
        <label>Track 1 nudge<input id="from-nudge-beats" type="number" min="-16" max="16" step="0.25" value="0"></label>
        <label>Track 2 nudge<input id="to-nudge-beats" type="number" min="-16" max="16" step="0.25" value="0"></label>
      </div>
      <div class="row">
        <label>Track 1 pitch<select id="from-pitch-shift"></select></label>
        <label>Track 2 pitch<select id="to-pitch-shift"></select></label>
      </div>
      <div id="key-preview" class="status">Keys update after tracks load</div>
      <label>Preset<select id="preset"></select></label>
      <label>Volume<select id="volume-mode"></select></label>
      <label>EQ<select id="eq-mode"></select></label>
      <label>Filter<select id="filter-mode"></select></label>
      <div class="row">
        <button id="render">Render</button>
        <button id="swap" class="secondary">Swap</button>
      </div>
      <div id="status" class="status">Presets: {presets}</div>
    </div>
  </aside>
  <section>
    <div class="topbar">
      <div id="result-meta" class="meta">No render yet</div>
      <div id="result-links" class="links"></div>
    </div>
    <div id="preview" class="empty">Render a transition to load the preview</div>
  </section>
</main>
<script>
(() => {{
  const state = {{ tracks: [], options: null, lastRender: null, lastRenderedBody: null }};
  const els = {{
    fromTrack: document.getElementById('from-track'),
    toTrack: document.getElementById('to-track'),
    fromCue: document.getElementById('from-cue'),
    toCue: document.getElementById('to-cue'),
    overlapBars: document.getElementById('overlap-bars'),
    paddingBars: document.getElementById('padding-bars'),
    frontPaddingBars: document.getElementById('front-padding-bars'),
    backPaddingBars: document.getElementById('back-padding-bars'),
    fromNudgeBeats: document.getElementById('from-nudge-beats'),
    toNudgeBeats: document.getElementById('to-nudge-beats'),
    fromPitchShift: document.getElementById('from-pitch-shift'),
    toPitchShift: document.getElementById('to-pitch-shift'),
    keyPreview: document.getElementById('key-preview'),
    preset: document.getElementById('preset'),
    volumeMode: document.getElementById('volume-mode'),
    eqMode: document.getElementById('eq-mode'),
    filterMode: document.getElementById('filter-mode'),
    render: document.getElementById('render'),
    swap: document.getElementById('swap'),
    status: document.getElementById('status'),
    preview: document.getElementById('preview'),
    resultMeta: document.getElementById('result-meta'),
    resultLinks: document.getElementById('result-links'),
  }};
  function esc(s) {{
    return String(s == null ? '' : s).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
  }}
  function setStatus(text, error=false) {{
    els.status.textContent = text;
    els.status.classList.toggle('error', error);
  }}
  function trackLabel(track) {{
    const bpm = track.bpm ? ' / ' + Number(track.bpm).toFixed(1) + ' BPM' : '';
    const key = track.key ? ' / ' + track.key : '';
    const source = track.source_label ? track.source_label + ' / ' : '';
    return source + String(track.track_number).padStart(2, '0') + ' - ' + track.title + ' - ' + track.artists + bpm + key;
  }}
  function pitchLabel(value) {{
    const n = Number(value) || 0;
    return n === 0 ? '0 st' : (n > 0 ? '+' : '') + n + ' st';
  }}
  function shiftCamelot(key, semitones) {{
    const text = String(key || '').trim().toUpperCase();
    const match = text.match(/^(\\d{{1,2}})([AB])$/);
    if (!match) return text;
    const number = Number(match[1]);
    if (number < 1 || number > 12) return text;
    const shifted = ((number - 1 + (7 * Number(semitones || 0))) % 12 + 12) % 12 + 1;
    return String(shifted) + match[2];
  }}
  function fillSelect(select, items, valueFn, labelFn, selected=null) {{
    select.innerHTML = '';
    for (const item of items) {{
      const option = document.createElement('option');
      option.value = valueFn(item);
      option.textContent = labelFn(item);
      select.appendChild(option);
    }}
    if (selected != null) select.value = String(selected);
  }}
  function selectedTrack(select) {{
    return state.tracks.find(track => track.id === select.value) || null;
  }}
  function cuesFor(track, role) {{
    if (!track) return [];
    const cues = Array.isArray(track.cues) ? track.cues : [];
    return cues;
  }}
  function fillCues() {{
    const from = selectedTrack(els.fromTrack);
    const to = selectedTrack(els.toTrack);
    const fromCues = cuesFor(from, 'out');
    const toCues = cuesFor(to, 'in');
    const defaultFrom = fromCues.find(cue => String(cue.role || '').toLowerCase() === 'out')?.name || fromCues[0]?.name;
    const defaultTo = toCues.find(cue => String(cue.role || '').toLowerCase() === 'in')?.name || toCues[0]?.name;
    fillSelect(els.fromCue, fromCues, cue => cue.name, cue => cue.name + ' / ' + (cue.role || 'cue') + (cue.start_seconds == null ? '' : ' @ ' + Number(cue.start_seconds).toFixed(2) + 's'), defaultFrom);
    fillSelect(els.toCue, toCues, cue => cue.name, cue => cue.name + ' / ' + (cue.role || 'cue') + (cue.start_seconds == null ? '' : ' @ ' + Number(cue.start_seconds).toFixed(2) + 's'), defaultTo);
    updateKeyPreview();
  }}
  function presetDetails(name) {{
    return (state.options.preset_details || {{}})[name] || {{}};
  }}
  function applyPresetToControls(name) {{
    const preset = presetDetails(name || 'auto');
    els.volumeMode.value = preset.volume_mode || els.volumeMode.value || 'crossfade';
    els.eqMode.value = preset.eq_mode || els.eqMode.value || 'none';
    els.filterMode.value = preset.filter_mode || els.filterMode.value || 'none';
  }}
  function setCustomPreset() {{
    if (els.preset.value !== 'custom') els.preset.value = 'custom';
  }}
  function fillModes() {{
    fillSelect(els.preset, state.options.presets, x => x, x => x, 'auto');
    fillSelect(els.volumeMode, state.options.volume_modes, x => x, x => x);
    fillSelect(els.eqMode, state.options.eq_modes, x => x, x => x);
    fillSelect(els.filterMode, state.options.filter_modes, x => x, x => x);
    applyPresetToControls(els.preset.value);
    const pitchValues = [-3, -2, -1, 0, 1, 2, 3];
    fillSelect(els.fromPitchShift, pitchValues, x => x, pitchLabel, 0);
    fillSelect(els.toPitchShift, pitchValues, x => x, pitchLabel, 0);
  }}
  function updateKeyPreview() {{
    const from = selectedTrack(els.fromTrack);
    const to = selectedTrack(els.toTrack);
    const fromShift = Number(els.fromPitchShift.value) || 0;
    const toShift = Number(els.toPitchShift.value) || 0;
    const fromKey = shiftCamelot(from?.key, fromShift);
    const toKey = shiftCamelot(to?.key, toShift);
    els.keyPreview.textContent = 'Keys: ' + (from?.key || '-') + ' -> ' + (fromKey || '-') +
      ' / ' + (to?.key || '-') + ' -> ' + (toKey || '-');
  }}
  async function loadInitial() {{
    const [tracksRes, optionsRes] = await Promise.all([fetch('/api/tracks'), fetch('/api/options')]);
    const tracks = await tracksRes.json();
    const options = await optionsRes.json();
    if (!tracks.ok) throw new Error(tracks.error || 'Track load failed');
    if (!options.ok) throw new Error(options.error || 'Option load failed');
    state.tracks = tracks.tracks;
    state.options = options;
    fillSelect(els.fromTrack, state.tracks, t => t.id, trackLabel, state.tracks[0]?.id);
    fillSelect(els.toTrack, state.tracks, t => t.id, trackLabel, state.tracks[1]?.id || state.tracks[0]?.id);
    fillModes();
    fillCues();
    setStatus(state.tracks.length + ' tracks loaded from ' + (tracks.sources || []).map(source => source.label).join(' + '));
  }}
  function renderBody() {{
    return {{
      from_track: els.fromTrack.value,
      to_track: els.toTrack.value,
      from_cue: els.fromCue.value,
      to_cue: els.toCue.value,
      overlap_bars: Number(els.overlapBars.value),
      front_padding_bars: Number(els.frontPaddingBars.value),
      back_padding_bars: Number(els.backPaddingBars.value),
      from_nudge_beats: Number(els.fromNudgeBeats.value),
      to_nudge_beats: Number(els.toNudgeBeats.value),
      from_pitch_shift: Number(els.fromPitchShift.value),
      to_pitch_shift: Number(els.toPitchShift.value),
      preset: els.preset.value,
      volume_mode: els.volumeMode.value,
      eq_mode: els.eqMode.value,
      filter_mode: els.filterMode.value,
      overwrite: true,
    }};
  }}
  async function renderTransition() {{
    els.render.disabled = true;
    setStatus('Rendering...');
    const body = renderBody();
    try {{
      const res = await fetch('/api/render', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(body),
      }});
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || 'Render failed');
      state.lastRender = data;
      state.lastRenderedBody = body;
      els.preview.className = '';
      els.preview.innerHTML = '<div class="preview-toolbar"><audio id="preview-audio" controls src="' + esc(data.urls.preview) + '?t=' + Date.now() + '"></audio></div><div class="canvas-wrap"><canvas id="preview-bg"></canvas><canvas id="preview-fg"></canvas></div>';
      const audio = document.getElementById('preview-audio');
      if (audio) {{
        audio.addEventListener('timeupdate', drawPreview);
        audio.addEventListener('seeked', drawPreview);
        audio.addEventListener('play', tickPlayhead);
        audio.addEventListener('pause', drawPreview);
      }}
      const t = data.transition;
      els.resultMeta.textContent = t.from_track_number + ' ' + t.from_title + ' -> ' + t.to_track_number + ' ' + t.to_title + ' / ' + Number(t.duration_seconds).toFixed(2) + 's';
      els.resultLinks.innerHTML = '<a href="' + esc(data.urls.visualizer) + '" target="_blank">Visualizer</a>' +
        '<a href="' + esc(data.urls.preview) + '" target="_blank">WAV</a>' +
        '<a href="' + esc(data.urls.metadata) + '" target="_blank">Metadata</a>';
      setStatus('Rendered ' + t.from_title + ' -> ' + t.to_title);
      await loadPreview(data);
    }} catch (err) {{
      setStatus(err.message || String(err), true);
    }} finally {{
      els.render.disabled = false;
    }}
  }}
  function smoothstep(x) {{
    return x * x * (3 - (2 * x));
  }}
  function resolvedModes(body) {{
    return {{
      volume: body.volume_mode || 'crossfade',
      eq: body.eq_mode || 'none',
      filter: body.filter_mode || 'none',
    }};
  }}
  function currentSignature(body) {{
    const modes = resolvedModes(body);
    return JSON.stringify({{
      from_track: body.from_track,
      to_track: body.to_track,
      from_cue: body.from_cue,
      to_cue: body.to_cue,
      overlap_bars: body.overlap_bars,
      front_padding_bars: body.front_padding_bars,
      back_padding_bars: body.back_padding_bars,
      from_nudge_beats: body.from_nudge_beats,
      to_nudge_beats: body.to_nudge_beats,
      from_pitch_shift: body.from_pitch_shift,
      to_pitch_shift: body.to_pitch_shift,
      volume_mode: modes.volume,
      eq_mode: modes.eq,
      filter_mode: modes.filter,
    }});
  }}
  function renderedSignature() {{
    return state.lastRenderedBody ? currentSignature(state.lastRenderedBody) : '';
  }}
  function valueFor(mode, deck, x) {{
    const half = x < 0.5;
    if (mode === 'volume:crossfade') return deck === 'a' ? Math.cos(x * Math.PI * 0.5) : Math.sin(x * Math.PI * 0.5);
    if (mode === 'volume:overlap-crossfade') {{
      const floor = Math.pow(10, -6 / 20);
      const curve = deck === 'a' ? Math.cos(x * Math.PI * 0.5) : Math.sin(x * Math.PI * 0.5);
      return floor + ((1 - floor) * curve);
    }}
    if (mode === 'volume:smooth-crossfade') {{
      const t = smoothstep(x);
      return deck === 'a' ? Math.sqrt(Math.max(0, 1 - t)) : Math.sqrt(Math.max(0, t));
    }}
    if (mode === 'volume:overlap') return 1;
    if (mode === 'volume:fade-in-fade-out') return deck === 'a' ? (half ? 1 : 1 - ((x - 0.5) * 2)) : (half ? x * 2 : 1);
    if (mode === 'volume:cut-in-fade-out') return deck === 'a' ? (half ? 1 : 1 - ((x - 0.5) * 2)) : 1;
    if (mode === 'volume:fade-in-cut-out') return deck === 'a' ? 1 : (half ? x * 2 : 1);
    if (mode === 'volume:center-cut') return deck === 'a' ? (half ? 1 : 0) : (half ? 0 : 1);
    if (mode === 'eq:none') return 1;
    if (mode === 'eq:start-bass-swap') return deck === 'a' ? 0.15 : 1;
    if (mode === 'eq:end-bass-swap') return deck === 'b' ? 0.15 : 1;
    if (mode === 'eq:center-bass-swap') return deck === 'a' ? (half ? 1 : 0.15) : (half ? 0.15 : 1);
    if (mode === 'eq:long-bass-cut') return 0.15;
    if (mode === 'filter:none') return null;
    if (mode === 'filter:low-pass-filter-out') return deck === 'a' ? 1 - x : null;
    if (mode === 'filter:low-pass-filter-in') return deck === 'b' ? x : null;
    if (mode === 'filter:high-pass-filter-out') return deck === 'a' ? x : null;
    if (mode === 'filter:high-pass-filter-in') return deck === 'b' ? 1 - x : null;
    return null;
  }}
  async function loadPreview(data) {{
    const [metadata, waveform] = await Promise.all([
      fetch(data.urls.metadata + '?t=' + Date.now()).then(r => r.json()),
      fetch(data.urls.waveform + '?t=' + Date.now()).then(r => r.json()),
    ]);
    state.preview = {{ metadata, waveform }};
    drawPreview();
  }}
  function xFor(t, lane) {{
    return lane.x + (Math.max(0, Math.min(1, Number(t) || 0)) * lane.w);
  }}
  function yFor(v, lane) {{
    return lane.y + ((1 - Math.max(0, Math.min(1, Number(v) || 0))) * lane.h);
  }}
  function smoothPoint(points, key, i) {{
    const a = points[Math.max(0, i - 1)];
    const b = points[i];
    const c = points[Math.min(points.length - 1, i + 1)];
    return ((Number(a[key]) || 0) + (2 * (Number(b[key]) || 0)) + (Number(c[key]) || 0)) / 4;
  }}
  function drawEnvelope(ctx, points, lane, key, color, scale) {{
    if (!points || !points.length) return;
    const mid = lane.y + lane.h * 0.5;
    const amp = lane.h * 0.46 * scale;
    ctx.fillStyle = color;
    ctx.beginPath();
    points.forEach((p, i) => {{
      const x = xFor(p.t, lane);
      const y = mid - (Math.min(1, smoothPoint(points, key, i)) * amp);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    for (let i = points.length - 1; i >= 0; i -= 1) {{
      const p = points[i];
      const x = xFor(p.t, lane);
      const y = mid + (Math.min(1, smoothPoint(points, key, i)) * amp);
      ctx.lineTo(x, y);
    }}
    ctx.closePath();
    ctx.fill();
  }}
  function drawWave(ctx, points, lane) {{
    drawEnvelope(ctx, points, lane, 'low_amp', 'rgba(24,75,216,.86)', 1.0);
    drawEnvelope(ctx, points, lane, 'mid_amp', 'rgba(211,128,39,.80)', 0.70);
    drawEnvelope(ctx, points, lane, 'high_amp', 'rgba(248,242,226,.92)', 0.36);
  }}
  function drawLine(ctx, points, lane, color, width, dashed=false, valueKey='value', normalizedKey=null) {{
    if (!points || !points.length) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (dashed) ctx.setLineDash([7, 6]);
    ctx.beginPath();
    points.forEach((p, i) => {{
      const v = normalizedKey ? p[normalizedKey] : p[valueKey];
      const x = xFor(p.t, lane);
      const y = yFor(v, {{ ...lane, y: lane.y + 8, h: lane.h - 16 }});
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke();
    ctx.restore();
  }}
  function drawDashedLine(ctx, lane, color, fn) {{
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.setLineDash([7, 6]);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let started = false;
    for (let i = 0; i <= 120; i += 1) {{
      const t = i / 120;
      const v = fn(t);
      if (v == null || Number.isNaN(v)) {{
        started = false;
        continue;
      }}
      const x = lane.x + (t * lane.w);
      const y = lane.y + ((1 - Math.max(0, Math.min(1, v))) * lane.h);
      if (!started) {{
        ctx.moveTo(x, y);
        started = true;
      }} else {{
        ctx.lineTo(x, y);
      }}
    }}
    ctx.stroke();
    ctx.restore();
  }}
  function drawPlayhead(ctx, lanes) {{
    const audio = document.getElementById('preview-audio');
    const duration = Number(audio?.duration) || Number(state.preview?.metadata?.render?.duration_seconds) || 0;
    if (!audio || !duration || !lanes.length) return;
    const t = Math.max(0, Math.min(1, audio.currentTime / duration));
    const x = xFor(t, lanes[0]);
    ctx.save();
    ctx.strokeStyle = 'rgba(246,242,232,.92)';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(x, lanes[0].y);
    ctx.lineTo(x, lanes[lanes.length - 1].y + lanes[lanes.length - 1].h);
    ctx.stroke();
    ctx.restore();
  }}
  function tickPlayhead() {{
    drawPreview();
    const audio = document.getElementById('preview-audio');
    if (audio && !audio.paused && !audio.ended) requestAnimationFrame(tickPlayhead);
  }}
  function drawAutomationPreview() {{
    updateKeyPreview();
    drawPreview();
  }}
  function drawPreview() {{
    updateKeyPreview();
    const bg = document.getElementById('preview-bg');
    const fg = document.getElementById('preview-fg');
    if (!bg || !fg || !state.preview) return;
    const body = renderBody();
    const dirty = currentSignature(body) !== renderedSignature();
    const rect = bg.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    for (const canvas of [bg, fg]) {{
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    }}
    const ctx = bg.getContext('2d');
    const fctx = fg.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    fctx.clearRect(0, 0, rect.width, rect.height);
    const metadata = state.preview.metadata;
    const waveform = state.preview.waveform;
    const pad = 18;
    const labelW = Math.min(250, Math.max(142, rect.width * 0.24));
    const gap = 18;
    const laneH = (rect.height - (pad * 2) - gap) / 2;
    const laneW = rect.width - (pad * 2) - labelW;
    const lanes = [
      {{ x: pad + labelW, y: pad, w: laneW, h: laneH, deck: 'a', key: 'outgoing', track: metadata.from_track || {{}} }},
      {{ x: pad + labelW, y: pad + laneH + gap, w: laneW, h: laneH, deck: 'b', key: 'incoming', track: metadata.to_track || {{}} }},
    ];
    const render = metadata.render || {{}};
    const bars = Math.max(1, Number(render.timeline_bars || ((render.front_padding_bars || 0) + (render.overlap_bars || 1) + (render.back_padding_bars || 0))));
    ctx.fillStyle = '#121212';
    ctx.fillRect(0, 0, rect.width, rect.height);
    lanes.forEach(lane => {{
      ctx.fillStyle = '#1d1d1d';
      ctx.fillRect(lane.x, lane.y, lane.w, lane.h);
      for (let i = 0; i <= bars; i += 1) {{
        const x = lane.x + (lane.w * i / bars);
        ctx.strokeStyle = i % 4 === 0 ? '#444' : '#2d2d2d';
        ctx.lineWidth = i % 4 === 0 ? 1.2 : 1;
        ctx.beginPath();
        ctx.moveTo(x, lane.y);
        ctx.lineTo(x, lane.y + lane.h);
        ctx.stroke();
      }}
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#f2f2f2';
      ctx.font = '13px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText(lane.track.title || lane.key, lane.x - 12, lane.y + lane.h * 0.44);
      ctx.fillStyle = '#8f8f8f';
      ctx.font = '11px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText((lane.track.artists || '') + (lane.track.cue_name ? ' / ' + lane.track.cue_name : ''), lane.x - 12, lane.y + lane.h * 0.56);
      drawWave(ctx, (waveform.waveforms || {{}})[lane.key] || [], lane);
    }});
    const auto = waveform.automation || {{}};
    if (auto.volume) {{
      drawLine(ctx, auto.volume.outgoing, lanes[0], '#46c7f3', 3);
      drawLine(ctx, auto.volume.incoming, lanes[1], '#46c7f3', 3);
    }}
    if (auto.eq) {{
      drawLine(ctx, auto.eq.outgoing_low_gain, lanes[0], '#ffd13d', 2.5);
      drawLine(ctx, auto.eq.incoming_low_gain, lanes[1], '#ffd13d', 2.5);
    }}
    if (auto.filter) {{
      drawLine(ctx, auto.filter.outgoing_cutoff_hz, lanes[0], '#e58cff', 2.5, false, 'value', 'normalized');
      drawLine(ctx, auto.filter.incoming_cutoff_hz, lanes[1], '#e58cff', 2.5, false, 'value', 'normalized');
    }}
    if (dirty) {{
      const modes = resolvedModes(body);
      const front = Math.max(0, Number(body.front_padding_bars) || 0);
      const overlap = Math.max(1, Number(body.overlap_bars) || 1);
      const back = Math.max(0, Number(body.back_padding_bars) || 0);
      const total = front + overlap + back;
      const timelineValue = (mode, deck, t, frontValue, backValue) => {{
        const bar = t * total;
        if (bar < front) return frontValue;
        if (bar > front + overlap) return backValue;
        const x = Math.max(0, Math.min(1, (bar - front) / overlap));
        return valueFor(mode, deck, x);
      }};
      const volumeMode = 'volume:' + modes.volume;
      const eqMode = 'eq:' + modes.eq;
      const filterMode = 'filter:' + modes.filter;
      drawDashedLine(fctx, lanes[0], '#48c8f2', t => timelineValue(volumeMode, 'a', t, 1, 0));
      drawDashedLine(fctx, lanes[1], '#48c8f2', t => timelineValue(volumeMode, 'b', t, 0, 1));
      drawDashedLine(fctx, lanes[0], '#ffd13d', t => timelineValue(eqMode, 'a', t, 1, 1));
      drawDashedLine(fctx, lanes[1], '#ffd13d', t => timelineValue(eqMode, 'b', t, 1, 1));
      drawDashedLine(fctx, lanes[0], '#e58cff', t => timelineValue(filterMode, 'a', t, null, null));
      drawDashedLine(fctx, lanes[1], '#e58cff', t => timelineValue(filterMode, 'b', t, null, null));
      fctx.fillStyle = 'rgba(242,242,242,.82)';
      fctx.font = '12px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      fctx.fillText('unrendered settings preview', lanes[0].x, Math.max(18, lanes[0].y - 12));
    }}
    drawPlayhead(fctx, lanes);
  }}
  els.fromTrack.addEventListener('change', () => {{ fillCues(); drawAutomationPreview(); }});
  els.toTrack.addEventListener('change', () => {{ fillCues(); drawAutomationPreview(); }});
  els.fromCue.addEventListener('change', drawAutomationPreview);
  els.toCue.addEventListener('change', drawAutomationPreview);
  els.paddingBars.addEventListener('change', () => {{
    els.frontPaddingBars.value = els.paddingBars.value;
    els.backPaddingBars.value = els.paddingBars.value;
    drawAutomationPreview();
  }});
  els.preset.addEventListener('change', () => {{
    applyPresetToControls(els.preset.value);
    drawAutomationPreview();
  }});
  els.preset.addEventListener('input', () => {{
    applyPresetToControls(els.preset.value);
    drawAutomationPreview();
  }});
  [els.overlapBars, els.frontPaddingBars, els.backPaddingBars, els.fromNudgeBeats, els.toNudgeBeats, els.fromPitchShift, els.toPitchShift].forEach(el => {{
    el.addEventListener('change', drawAutomationPreview);
    el.addEventListener('input', drawAutomationPreview);
  }});
  [els.volumeMode, els.eqMode, els.filterMode].forEach(el => {{
    el.addEventListener('change', () => {{ setCustomPreset(); drawAutomationPreview(); }});
    el.addEventListener('input', () => {{ setCustomPreset(); drawAutomationPreview(); }});
  }});
  window.addEventListener('resize', drawAutomationPreview);
  els.render.addEventListener('click', renderTransition);
  els.swap.addEventListener('click', () => {{
    const fromTrack = els.fromTrack.value;
    const fromCue = els.fromCue.value;
    els.fromTrack.value = els.toTrack.value;
    els.toTrack.value = fromTrack;
    fillCues();
    if ([...els.toCue.options].some(option => option.value === fromCue)) els.toCue.value = fromCue;
  }});
  loadInitial().catch(err => setStatus(err.message || String(err), true));
}})();
</script>
</body>
</html>
"""


def make_handler(workbench: TransitionWorkbench) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "TransitionWorkbench/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path in {"/", "/index.html"}:
                _text_response(self, HTTPStatus.OK, _workbench_html(), "text/html; charset=utf-8")
                return
            if path == "/api/tracks":
                try:
                    _json_response(self, HTTPStatus.OK, workbench.tracks_payload())
                except Exception as exc:
                    _error_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            if path == "/api/options":
                _json_response(self, HTTPStatus.OK, workbench.options_payload())
                return
            if path.startswith("/artifacts/"):
                self._serve_artifact(path)
                return
            _error_response(self, HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/render":
                _error_response(self, HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object.")
                _json_response(self, HTTPStatus.OK, workbench.render_payload(payload))
            except Exception as exc:
                _error_response(self, HTTPStatus.BAD_REQUEST, str(exc))

        def _serve_artifact(self, path: str) -> None:
            parts = [unquote(part) for part in path.split("/") if part]
            if len(parts) != 3:
                _error_response(self, HTTPStatus.NOT_FOUND, "Artifact not found")
                return
            _, transition_id, filename = parts
            artifact = workbench.artifact_path(transition_id, filename)
            if artifact is None:
                _error_response(self, HTTPStatus.NOT_FOUND, "Artifact not found")
                return
            content_type = mimetypes.guess_type(str(artifact))[0] or "application/octet-stream"
            _file_response(self, artifact, content_type)

    return Handler


def serve_workbench(
    *,
    host: str,
    port: int,
    tracklists: list[Path],
    cue_tables: list[Path | None] | None,
    output_dir: Path,
) -> None:
    workbench = TransitionWorkbench(tracklists=tracklists, cue_tables=cue_tables, output_dir=output_dir)
    server = ThreadingHTTPServer((host, port), make_handler(workbench))
    print(f"Transition workbench: http://{host}:{port}/")
    for source in workbench.sources:
        print(f"Source {source.slug}:")
        print(f"  Tracklist: {source.tracklist}")
        print(f"  Cue table: {source.cue_table}")
    print(f"Output dir: {workbench.output_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping transition workbench.")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local transition render workbench.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--tracklist",
        type=Path,
        action="append",
        default=None,
        help="Playlist CSV to load. May be passed multiple times. Defaults to Aries + Ara.",
    )
    parser.add_argument(
        "--cue-table",
        type=Path,
        action="append",
        default=None,
        help="Cue CSV aligned with each --tracklist. Defaults to <mix>_cues.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    tracklists = args.tracklist or DEFAULT_TRACKLISTS
    cue_tables = args.cue_table or []
    if cue_tables and len(cue_tables) != len(tracklists):
        raise ValueError("--cue-table must be passed once per --tracklist when provided.")
    serve_workbench(
        host=args.host,
        port=args.port,
        tracklists=tracklists,
        cue_tables=cue_tables,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
