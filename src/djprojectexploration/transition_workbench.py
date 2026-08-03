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
from urllib.parse import urlparse

import numpy as np

from djprojectexploration.frontend_assets import frontend_asset_text, render_standalone_document
from djprojectexploration.local_http import (
    artifact_route_parts,
    error_response,
    file_response,
    json_response,
    text_response,
)
from djprojectexploration.loudness_matching import (
    DEFAULT_LOUDNESS_MATCH_MODE,
    DEFAULT_TARGET_LUFS,
    MAX_COMPENSATION_DB,
    LoudnessMeasurement,
    load_loudness_catalog,
    lookup_loudness,
    normalize_mode,
)
from djprojectexploration.tracklists import PROJECT_ROOT, load_playlist_tracks, read_csv_rows
from djprojectexploration.transition_preview import (
    AUTOMATION_LANES,
    DEFAULT_OUTPUT_DIR,
    EQ_MODES,
    FILTER_MODES,
    TRANSITION_PRESETS,
    TRANSITION_QUALITY_CHOICES,
    TRANSITION_QUALITY_PROFILES,
    VOLUME_MODES,
    PreparedLiveTransition,
    RenderedTransition,
    prepare_live_transition,
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
        render_visuals: bool = True,
    ) -> None:
        if not tracklists:
            raise ValueError("At least one tracklist is required.")
        self.sources = _dedupe_sources(tracklists, cue_tables or [])
        self.sources_by_slug = {source.slug: source for source in self.sources}
        self.output_dir = output_dir.expanduser().resolve()
        self.generated_source_dir = self.output_dir / "_workbench_sources"
        self._tempo_bpm_cache: dict[Path, dict[int, float]] = {}
        self._loudness_by_track_id: dict[str, LoudnessMeasurement] = {}
        for source in self.sources:
            by_filename, by_number = load_loudness_catalog(source.tracklist)
            for track in load_playlist_tracks(source.tracklist):
                self._loudness_by_track_id[f"{source.slug}:{track.track_number}"] = lookup_loudness(
                    by_filename,
                    by_number,
                    filename=track.mp3_name,
                    track_number=track.track_number,
                )
        self.render_visuals = bool(render_visuals)

    def _loudness_for_track_id(self, track_id: str) -> LoudnessMeasurement:
        return self._loudness_by_track_id.get(track_id, LoudnessMeasurement())

    def _tempo_embedding_path(self, source: TrackSource) -> Path:
        return PROJECT_ROOT / "data" / "tempo_embeddings" / f"{source.tracklist.stem}.npz"

    def _tempo_bpm_by_track(self, source: TrackSource) -> dict[int, float]:
        path = self._tempo_embedding_path(source).expanduser().resolve()
        if path in self._tempo_bpm_cache:
            return self._tempo_bpm_cache[path]
        values: dict[int, float] = {}
        if path.exists():
            with np.load(path, allow_pickle=True) as payload:
                track_numbers = np.asarray(payload.get("track_numbers", []), dtype=np.int64).reshape(-1)
                tempo_bpm = np.asarray(payload.get("tempo_bpm", []), dtype=np.float64).reshape(-1)
                for index, track_number in enumerate(track_numbers):
                    bpm = float(tempo_bpm[index]) if index < tempo_bpm.size else float("nan")
                    if np.isfinite(bpm) and bpm > 0:
                        values[int(track_number)] = bpm
        self._tempo_bpm_cache[path] = values
        return values

    def _resolved_bpm(self, source: TrackSource, track: Any) -> float | None:
        try:
            bpm = float(track.bpm)
        except (TypeError, ValueError):
            bpm = float("nan")
        if np.isfinite(bpm) and bpm > 0:
            return bpm
        return self._tempo_bpm_by_track(source).get(int(track.track_number))

    def _cues_by_track(self, source: TrackSource) -> dict[int, list[dict[str, Any]]]:
        cue_rows = read_csv_rows(source.cue_table, missing_ok=True)
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
                bpm = self._resolved_bpm(source, track)
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
                        "bpm": bpm,
                        "onset_time": track.onset_time,
                        "cues": cues_by_track.get(track.track_number, []),
                        "loudness": self._loudness_for_track_id(f"{source.slug}:{track.track_number}").payload(),
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
            "automation": {
                "decks": ["from", "to"],
                "lanes": list(AUTOMATION_LANES),
                "curves": ["linear", "smooth", "exponential"],
                "volume": {"minimum_db": -96, "maximum_db": 6, "mute_db": -96},
                "eq": {"type": "isolator", "minimum_db": -96, "maximum_db": 6, "crossovers_hz": [275, 5000]},
                "filter": {
                    "minimum": -1,
                    "maximum": 1,
                    "center": 0,
                    "lowpass_min_hz": 20,
                    "highpass_max_hz": 20000,
                    "edge_kill_start": 0.975,
                    "edge_kill_end": 0.995,
                },
                "delay": {"minimum": 0, "maximum": 1, "default_beats": 0.5, "beat_divisions": [0.125, 0.25, 0.5, 0.75, 1, 2, 4]},
                "reverb": {"minimum": 0, "maximum": 1, "default_decay_seconds": 1.2},
            },
            "transition_qualities": [
                {
                    "id": quality,
                    "label": "Preview — faster R2, 44.1 kHz" if quality == "preview" else "Final — higher-quality R3, 44.1 kHz",
                    "sample_rate": int(TRANSITION_QUALITY_PROFILES[quality]["sample_rate"]),
                }
                for quality in TRANSITION_QUALITY_CHOICES
            ],
            "loudness_matching": {
                "default_mode": DEFAULT_LOUDNESS_MATCH_MODE,
                "target_lufs": DEFAULT_TARGET_LUFS,
                "maximum_compensation_db": MAX_COMPENSATION_DB,
            },
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
        for row in read_csv_rows(source.cue_table, missing_ok=True):
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
                bpm = self._resolved_bpm(source, track)
                writer.writerow(
                    {
                        "track_number": rendered_track_number,
                        "title": track.title,
                        "artists": track.artists,
                        "mp3_name": track.mp3_name,
                        "filepath": str(track.audio_path),
                        "key": track.key,
                        "bpm": "" if bpm is None else f"{float(bpm):.6f}".rstrip("0").rstrip("."),
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
        automation = body.get("automation")
        if automation is not None and not isinstance(automation, dict):
            raise ValueError("Automation must be an object.")
        effects = body.get("effects")
        if effects is not None and not isinstance(effects, dict):
            raise ValueError("Effects must be an object.")
        loudness_mode = normalize_mode(body.get("loudness_match_mode"))
        from_loudness = self._loudness_for_track_id(from_track)
        to_loudness = self._loudness_for_track_id(to_track)

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
            from_beatgrid_ms=float(body.get("from_beatgrid_ms") or 0.0),
            to_beatgrid_ms=float(body.get("to_beatgrid_ms") or 0.0),
            from_pitch_shift=int(body.get("from_pitch_shift") or 0),
            to_pitch_shift=int(body.get("to_pitch_shift") or 0),
            preset=str(body.get("preset") or "auto"),
            volume_mode=_optional_mode(body.get("volume_mode")),
            eq_mode=_optional_mode(body.get("eq_mode")),
            filter_mode=_optional_mode(body.get("filter_mode")),
            automation=automation,
            effects=effects,
            loudness_match_mode=loudness_mode,
            from_loudness_lufs=from_loudness.integrated_lufs,
            to_loudness_lufs=to_loudness.integrated_lufs,
            quality=str(body.get("quality") or "preview"),
            overwrite=bool(body.get("overwrite", False)),
            write_waveform=self.render_visuals,
        )
        visualizer = (
            export_transition_visualizer(transition_dir=result.preview_path.parent)
            if self.render_visuals
            else None
        )
        return {
            "ok": True,
            "transition": _transition_payload(result),
            "paths": {
                "visualizer": "" if visualizer is None else str(visualizer),
                "preview": str(result.preview_path),
                "metadata": str(result.metadata_path),
                "waveform": str(result.waveform_path) if self.render_visuals else "",
            },
            "urls": {
                "visualizer": _artifact_url(result.preview_path.parent.name, visualizer.name) if visualizer is not None else "",
                "preview": _artifact_url(result.preview_path.parent.name, result.preview_path.name),
                "metadata": _artifact_url(result.preview_path.parent.name, result.metadata_path.name),
                "waveform": _artifact_url(result.preview_path.parent.name, result.waveform_path.name) if self.render_visuals else "",
            },
        }

    def prepare_live_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        """Prepare matched deck assets for the served app's Web Audio mixer."""
        from_track = _optional_str(body.get("from_track"))
        to_track = _optional_str(body.get("to_track"))
        if not from_track or not to_track:
            raise ValueError("Both from_track and to_track are required.")
        render_tracklist, render_cue_table, from_query, to_query, default_from_cue, default_to_cue = self._write_render_sources(
            from_track,
            to_track,
        )
        result = prepare_live_transition(
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
            from_bar=float(body.get("from_bar") or 0.0),
            to_bar=float(body.get("to_bar") or 0.0),
            from_nudge_beats=float(body.get("from_nudge_beats") or 0.0),
            to_nudge_beats=float(body.get("to_nudge_beats") or 0.0),
            from_beatgrid_ms=float(body.get("from_beatgrid_ms") or 0.0),
            to_beatgrid_ms=float(body.get("to_beatgrid_ms") or 0.0),
            from_pitch_shift=int(body.get("from_pitch_shift") or 0),
            to_pitch_shift=int(body.get("to_pitch_shift") or 0),
            guard_bars=int(body.get("live_guard_bars") or 4),
            quality="preview",
            overwrite=bool(body.get("overwrite", False)),
        )
        loudness_mode = normalize_mode(body.get("loudness_match_mode"))
        return self._prepared_live_payload(
            result,
            from_loudness=self._loudness_for_track_id(from_track).payload(loudness_mode),
            to_loudness=self._loudness_for_track_id(to_track).payload(loudness_mode),
            loudness_mode=loudness_mode,
        )

    @staticmethod
    def _prepared_live_payload(
        result: PreparedLiveTransition,
        *,
        from_loudness: dict[str, float | None] | None = None,
        to_loudness: dict[str, float | None] | None = None,
        loudness_mode: str = DEFAULT_LOUDNESS_MATCH_MODE,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "live": {
                "transition_id": result.transition_id,
                "sample_rate": result.sample_rate,
                "duration_seconds": result.duration_seconds,
                "transport_start_seconds": result.transport_start_seconds,
                "transport_end_seconds": result.transport_end_seconds,
                "transition_start_seconds": result.transition_start_seconds,
                "transition_end_seconds": result.transition_end_seconds,
                "loop_start_seconds": result.loop_start_seconds,
                "loop_end_seconds": result.loop_end_seconds,
                "overlap_bars": result.overlap_bars,
                "front_padding_bars": result.front_padding_bars,
                "back_padding_bars": result.back_padding_bars,
                "guard_bars": result.guard_bars,
                "beats_per_bar": result.beats_per_bar,
                "from_bpm": result.from_bpm,
                "to_bpm": result.to_bpm,
                "b_time_stretch_rate": result.b_time_stretch_rate,
            },
            "urls": {
                "from": _artifact_url(result.transition_id, result.from_path.name),
                "to": _artifact_url(result.transition_id, result.to_path.name),
                "metadata": _artifact_url(result.transition_id, "live_transition.json"),
            },
            "loudness": {"mode": loudness_mode, "from": from_loudness or {}, "to": to_loudness or {}},
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
    body_html = frontend_asset_text("templates/transition_workbench_body.html").replace("{{PRESETS}}", presets)
    return render_standalone_document(
        title="Transition Workbench",
        css_asset="static/transition_workbench.css",
        body_html=body_html,
        script_asset="static/transition_workbench.js",
    )



def make_handler(workbench: TransitionWorkbench) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "TransitionWorkbench/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path in {"/", "/index.html"}:
                text_response(self, HTTPStatus.OK, _workbench_html(), "text/html; charset=utf-8")
                return
            if path == "/api/tracks":
                try:
                    json_response(self, HTTPStatus.OK, workbench.tracks_payload())
                except Exception as exc:
                    error_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            if path == "/api/options":
                json_response(self, HTTPStatus.OK, workbench.options_payload())
                return
            if path.startswith("/artifacts/"):
                self._serve_artifact(path)
                return
            error_response(self, HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/render":
                error_response(self, HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object.")
                json_response(self, HTTPStatus.OK, workbench.render_payload(payload))
            except Exception as exc:
                error_response(self, HTTPStatus.BAD_REQUEST, str(exc))

        def _serve_artifact(self, path: str) -> None:
            artifact_parts = artifact_route_parts(path)
            if artifact_parts is None:
                error_response(self, HTTPStatus.NOT_FOUND, "Artifact not found")
                return
            transition_id, filename = artifact_parts
            artifact = workbench.artifact_path(transition_id, filename)
            if artifact is None:
                error_response(self, HTTPStatus.NOT_FOUND, "Artifact not found")
                return
            content_type = mimetypes.guess_type(str(artifact))[0] or "application/octet-stream"
            file_response(self, artifact, content_type)

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
