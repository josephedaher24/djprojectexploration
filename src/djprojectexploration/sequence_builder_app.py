"""Local served app for the DJ sequence builder."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from djprojectexploration.energy_sequence_builder import CONTROL_MODE_CHOICES, PROJECT_ROOT, export_dj_sequence
from djprojectexploration.local_http import (
    artifact_route_parts,
    error_response,
    file_response,
    json_response,
    text_response,
)
from djprojectexploration.transition_preview import DEFAULT_OUTPUT_DIR
from djprojectexploration.transition_workbench import DEFAULT_TRACKLISTS, TransitionWorkbench


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770


def _tracklists_from_mix_slugs(project_root: Path, mix_slugs: list[str] | None) -> list[Path]:
    if not mix_slugs:
        return list(DEFAULT_TRACKLISTS)
    return [
        project_root / "music" / mix_slug / f"{mix_slug.replace('-', '_')}_tracks.csv"
        for mix_slug in mix_slugs
    ]


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
        step: float,
        static_layout: bool,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.tracklists = [path.expanduser().resolve() for path in tracklists]
        self.mix_slugs = [path.parent.name for path in self.tracklists]
        self.workbench = TransitionWorkbench(tracklists=self.tracklists, cue_tables=None, output_dir=self.output_dir)
        self.html_file = self.project_root / "data" / "exports" / "dj_sequence_builder_app.html"
        self.html_file = export_dj_sequence(
            project_root=self.project_root,
            output_file=self.html_file,
            mix_slugs=self.mix_slugs,
            energy_npz_path=energy_npz_path,
            default_length=sequence_length,
            control_mode=control_mode,
            step=step,
            static_layout=static_layout,
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


def make_handler(app: SequenceBuilderApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SequenceBuilderApp/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
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
                self._serve_artifact(path)
                return
            if path.startswith("/assets/"):
                self._serve_asset(path)
                return
            error_response(self, HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/render-transition":
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

        def _serve_artifact(self, path: str) -> None:
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

        def _serve_asset(self, path: str) -> None:
            asset = app.asset_path(path[len("/assets/") :])
            if asset is None:
                error_response(self, HTTPStatus.NOT_FOUND, "Asset not found")
                return
            content_type = mimetypes.guess_type(str(asset))[0] or "application/octet-stream"
            file_response(self, asset, content_type)

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
    step: float,
    static_layout: bool,
) -> None:
    app = SequenceBuilderApp(
        project_root=project_root,
        tracklists=tracklists,
        output_dir=output_dir,
        energy_npz_path=energy_npz_path,
        sequence_length=sequence_length,
        control_mode=control_mode,
        step=step,
        static_layout=static_layout,
    )
    server = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"Sequence builder app: http://{host}:{port}/")
    print(f"HTML snapshot: {app.html_file}")
    for source in app.workbench.sources:
        print(f"Source {source.slug}:")
        print(f"  Tracklist: {source.tracklist}")
        print(f"  Cue table: {source.cue_table}")
    print(f"Output dir: {app.output_dir}")
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--energy-npz", type=Path, default=None)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--control-mode", choices=CONTROL_MODE_CHOICES, default="genre-mixability")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--dynamic-layout", action="store_true", help="Generate the full dynamic PaCMAP layout grid at startup.")
    args = parser.parse_args(argv)

    project_root = args.project_root.expanduser().resolve()
    tracklists = args.tracklist or _tracklists_from_mix_slugs(project_root, args.mix_slugs)
    serve_sequence_builder_app(
        host=args.host,
        port=args.port,
        project_root=project_root,
        tracklists=tracklists,
        output_dir=args.output_dir,
        energy_npz_path=args.energy_npz,
        sequence_length=args.sequence_length,
        control_mode=args.control_mode,
        step=args.step,
        static_layout=not bool(args.dynamic_layout),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
