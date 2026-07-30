"""Local served app for the DJ sequence builder."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from djprojectexploration.energy_sequence_builder import CONTROL_MODE_CHOICES, PROJECT_ROOT, export_dj_sequence
from djprojectexploration.local_http import (
    artifact_route_parts,
    error_response,
    file_response,
    json_response,
    text_response,
)
from djprojectexploration.pacmap_settings import (
    PacmapSettings,
    SequenceBuilderUiSettings,
    add_pacmap_args,
    pacmap_settings_from_args,
    sequence_builder_run_settings_from_args,
    sequence_builder_ui_settings_from_args,
)
from djprojectexploration.tracklists import load_playlist_tracks
from djprojectexploration.transition_preview import DEFAULT_OUTPUT_DIR
from djprojectexploration.transition_workbench import DEFAULT_TRACKLISTS, TransitionWorkbench


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770

LOADING_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DJ Sequence Builder</title>
<style>
  body { margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center; background: #0b1020; color: #e5e7eb; font-family: system-ui, sans-serif; }
  .box { text-align: center; max-width: 32rem; padding: 2rem; }
  .spinner { width: 2.5rem; height: 2.5rem; margin: 0 auto 1.25rem; border: 3px solid rgba(20, 184, 166, 0.25); border-top-color: #14b8a6; border-radius: 50%; animation: spin 0.9s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .elapsed { margin-top: 0.75rem; color: #14b8a6; font-variant-numeric: tabular-nums; }
  .error { margin-top: 1rem; color: #f87171; display: none; white-space: pre-wrap; }
  .hint { margin-top: 0.5rem; color: #9ca3af; display: none; }
</style>
</head>
<body>
<div class="box">
  <div class="spinner" id="spinner"></div>
  <div>Preparing the track library &mdash; first launch computes map layouts and can take a minute.</div>
  <div class="elapsed"><span id="elapsed">__ELAPSED__</span>s elapsed</div>
  <div class="error" id="error"></div>
  <div class="hint" id="hint">Retry: restart the server, then reload this page.</div>
</div>
<script>
  var elapsed = __ELAPSED__;
  var elapsedEl = document.getElementById('elapsed');
  var ticker = setInterval(function () { elapsed += 1; elapsedEl.textContent = String(elapsed); }, 1000);
  var poller = setInterval(function () {
    fetch('/api/ready').then(function (res) { return res.json(); }).then(function (data) {
      if (data.ready) { location.reload(); return; }
      if (data.error) {
        clearInterval(poller);
        clearInterval(ticker);
        document.getElementById('spinner').style.display = 'none';
        var errorEl = document.getElementById('error');
        errorEl.textContent = 'Startup failed: ' + data.error;
        errorEl.style.display = 'block';
        document.getElementById('hint').style.display = 'block';
      }
    }).catch(function () {});
  }, 1500);
</script>
</body>
</html>
"""


def _route_path(raw_path: str) -> str:
    return raw_path.split("?", 1)[0].split("#", 1)[0]


def _tracklists_from_mix_slugs(project_root: Path, mix_slugs: list[str] | None) -> list[Path]:
    if not mix_slugs:
        return list(DEFAULT_TRACKLISTS)
    return [
        project_root / "music" / mix_slug / f"{mix_slug.replace('-', '_')}_tracks.csv"
        for mix_slug in mix_slugs
    ]


def _resolve_project_path(project_root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    path = path.expanduser()
    return path if path.is_absolute() else project_root / path


def _resolve_project_paths(project_root: Path, paths: list[Path] | None) -> list[Path] | None:
    if paths is None:
        return None
    return [_resolve_project_path(project_root, path) for path in paths if _resolve_project_path(project_root, path) is not None]


class SequenceBuilderApp:
    def __init__(
        self,
        *,
        project_root: Path,
        tracklists: list[Path],
        output_dir: Path,
        energy_npz_path: Path | None,
        sequence_length: int,
        control_mode: str,
        pacmap_settings: PacmapSettings,
        ui_settings: SequenceBuilderUiSettings | None,
        settings_preset_source: Path | None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.tracklists = [path.expanduser().resolve() for path in tracklists]
        self.mix_slugs = [path.parent.name for path in self.tracklists]
        self.workbench = TransitionWorkbench(tracklists=self.tracklists, cue_tables=None, output_dir=self.output_dir)
        self.audio_by_track_id: dict[str, Path] = {}
        for source in self.workbench.sources:
            for track in load_playlist_tracks(source.tracklist):
                self.audio_by_track_id[f"{source.slug}:{track.track_number}"] = track.audio_path
        self.html_file = self.project_root / "data" / "exports" / "dj_sequence_builder_app.html"
        self.html_file = export_dj_sequence(
            project_root=self.project_root,
            output_file=self.html_file,
            mix_slugs=self.mix_slugs,
            energy_npz_path=energy_npz_path,
            default_length=sequence_length,
            control_mode=control_mode,
            pacmap_settings=pacmap_settings,
            ui_settings=ui_settings,
            settings_preset_source=settings_preset_source,
            app_mode=True,
        )
        self.html = self.html_file.read_text(encoding="utf-8")

    def render_transition(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.workbench.render_payload(body)

    def artifact_path(self, transition_id: str, filename: str) -> Path | None:
        return self.workbench.artifact_path(transition_id, filename)

    def asset_path(self, path: str) -> Path | None:
        safe_rel = Path(unquote(path).lstrip("/"))
        candidate = (self.project_root / safe_rel).resolve()
        try:
            candidate.relative_to(self.project_root)
        except ValueError:
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    def audio_path(self, track_id: str) -> Path | None:
        path = self.audio_by_track_id.get(track_id)
        if path is None:
            return None
        resolved = path.expanduser().resolve()
        if not resolved.exists() or not resolved.is_file():
            return None
        return resolved


def make_handler(holder: dict[str, Any]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SequenceBuilderApp/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

        def do_GET(self) -> None:
            path = _route_path(self.path)
            app: SequenceBuilderApp | None = holder["app"]
            if path == "/api/ready":
                json_response(self, HTTPStatus.OK, {"ready": app is not None, "error": holder["error"]})
                return
            if app is None:
                if path in {"/", "/index.html"}:
                    elapsed = max(0, int(time.time() - holder["started_at"]))
                    text_response(self, HTTPStatus.OK, LOADING_HTML.replace("__ELAPSED__", str(elapsed)), "text/html; charset=utf-8")
                    return
                json_response(self, HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Server is still starting up."})
                return
            if path in {"/", "/index.html"}:
                text_response(self, HTTPStatus.OK, app.html, "text/html; charset=utf-8")
                return
            if path == "/api/tracks":
                try:
                    json_response(self, HTTPStatus.OK, app.workbench.tracks_payload())
                except Exception as exc:
                    error_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            if path == "/api/options":
                json_response(self, HTTPStatus.OK, app.workbench.options_payload())
                return
            if path.startswith("/artifacts/"):
                self._serve_artifact(app, path)
                return
            if path.startswith("/assets/"):
                self._serve_asset(app, path)
                return
            if path.startswith("/audio/"):
                self._serve_audio(app, path)
                return
            error_response(self, HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            path = _route_path(self.path)
            app: SequenceBuilderApp | None = holder["app"]
            if app is None:
                json_response(self, HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Server is still starting up."})
                return
            if path != "/api/render-transition":
                error_response(self, HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object.")
                json_response(self, HTTPStatus.OK, app.render_transition(payload))
            except Exception as exc:
                error_response(self, HTTPStatus.BAD_REQUEST, str(exc))

        def _serve_artifact(self, app: SequenceBuilderApp, path: str) -> None:
            artifact_parts = artifact_route_parts(path)
            if artifact_parts is None:
                error_response(self, HTTPStatus.NOT_FOUND, "Artifact not found")
                return
            transition_id, filename = artifact_parts
            artifact = app.artifact_path(transition_id, filename)
            if artifact is None:
                error_response(self, HTTPStatus.NOT_FOUND, "Artifact not found")
                return
            content_type = mimetypes.guess_type(str(artifact))[0] or "application/octet-stream"
            file_response(self, artifact, content_type)

        def _serve_asset(self, app: SequenceBuilderApp, path: str) -> None:
            asset = app.asset_path(path[len("/assets/") :])
            if asset is None:
                error_response(self, HTTPStatus.NOT_FOUND, "Asset not found")
                return
            content_type = mimetypes.guess_type(str(asset))[0] or "application/octet-stream"
            file_response(self, asset, content_type)

        def _serve_audio(self, app: SequenceBuilderApp, path: str) -> None:
            track_id = unquote(path[len("/audio/") :])
            audio = app.audio_path(track_id)
            if audio is None:
                error_response(self, HTTPStatus.NOT_FOUND, "Audio not found")
                return
            content_type = mimetypes.guess_type(str(audio))[0] or "application/octet-stream"
            file_response(self, audio, content_type)

    return Handler


def serve_sequence_builder_app(
    *,
    host: str,
    port: int,
    project_root: Path,
    tracklists: list[Path],
    output_dir: Path,
    energy_npz_path: Path | None,
    sequence_length: int,
    control_mode: str,
    pacmap_settings: PacmapSettings,
    ui_settings: SequenceBuilderUiSettings | None,
    settings_preset_source: Path | None,
) -> None:
    holder: dict[str, Any] = {"app": None, "error": None, "started_at": time.time()}
    server = ThreadingHTTPServer((host, port), make_handler(holder))
    print(f"Sequence builder app: http://{host}:{port}/")

    def start_app() -> None:
        try:
            app = SequenceBuilderApp(
                project_root=project_root,
                tracklists=tracklists,
                output_dir=output_dir,
                energy_npz_path=energy_npz_path,
                sequence_length=sequence_length,
                control_mode=control_mode,
                pacmap_settings=pacmap_settings,
                ui_settings=ui_settings,
                settings_preset_source=settings_preset_source,
            )
        except Exception as exc:
            holder["error"] = str(exc)
            print(f"Sequence builder startup failed: {exc}", file=sys.stderr)
            return
        holder["app"] = app
        print(f"HTML snapshot: {app.html_file}")
        for source in app.workbench.sources:
            print(f"Source {source.slug}:")
            print(f"  Tracklist: {source.tracklist}")
            print(f"  Cue table: {source.cue_table}")
        print(f"Output dir: {app.output_dir}")

    threading.Thread(target=start_app, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping sequence builder app.")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local served DJ sequence builder app.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--mix", action="append", dest="mix_slugs", default=None, help="Mix slug to include; repeatable.")
    parser.add_argument("--tracklist", action="append", type=Path, default=None, help="Tracklist CSV to include; repeatable.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--energy-npz", type=Path, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--control-mode", choices=CONTROL_MODE_CHOICES, default=None)
    add_pacmap_args(parser, include_static_layout=False)
    parser.add_argument(
        "--dynamic-layout",
        action="store_true",
        help="Generate the full dynamic PaCMAP layout grid at startup. This is the default unless --static-layout is passed.",
    )
    parser.add_argument(
        "--static-layout",
        action="store_true",
        help="Generate one fixed/default PaCMAP layout for faster startup.",
    )
    args = parser.parse_args(argv)
    pacmap_settings = pacmap_settings_from_args(args)
    ui_settings = sequence_builder_ui_settings_from_args(args)
    run_settings = sequence_builder_run_settings_from_args(args)
    use_dynamic_layout = not bool(args.static_layout)
    if bool(args.dynamic_layout) or bool(args.static_layout) or args.pacmap_preset is None:
        pacmap_settings = PacmapSettings(
            **{
                **pacmap_settings.to_dict(),
                "static_layout": not use_dynamic_layout,
            }
        ).validate()

    project_root = args.project_root.expanduser().resolve()
    preset_tracklists = _resolve_project_paths(project_root, run_settings.tracklists)
    tracklists = args.tracklist or preset_tracklists or _tracklists_from_mix_slugs(
        project_root,
        args.mix_slugs or run_settings.mix_slugs,
    )
    serve_sequence_builder_app(
        host=args.host,
        port=args.port,
        project_root=project_root,
        tracklists=tracklists,
        output_dir=_resolve_project_path(project_root, args.output_dir or run_settings.output_dir) or DEFAULT_OUTPUT_DIR,
        energy_npz_path=_resolve_project_path(project_root, args.energy_npz or run_settings.energy_npz_path),
        sequence_length=args.sequence_length or run_settings.sequence_length or 10,
        control_mode=args.control_mode or run_settings.control_mode or "genre-mixability",
        pacmap_settings=pacmap_settings,
        ui_settings=ui_settings,
        settings_preset_source=args.pacmap_preset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
