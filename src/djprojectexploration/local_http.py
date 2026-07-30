"""Shared helpers for the small local HTTP apps."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import unquote


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def text_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    text: str,
    content_type: str = "text/plain; charset=utf-8",
) -> None:
    data = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def error_response(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    json_response(handler, status, {"ok": False, "error": message})


def parse_range_header(value: str, file_size: int) -> tuple[int, int] | None:
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


def file_response(handler: BaseHTTPRequestHandler, path: Path, content_type: str) -> None:
    file_size = path.stat().st_size
    byte_range = parse_range_header(handler.headers.get("Range", ""), file_size)
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
    except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
        pass


def artifact_route_parts(path: str, *, prefix: str = "/artifacts/") -> tuple[str, str] | None:
    prefix_part = prefix.strip("/")
    parts = [unquote(part) for part in path.split("/") if part]
    if len(parts) != 3 or parts[0] != prefix_part:
        return None
    return parts[1], parts[2]
