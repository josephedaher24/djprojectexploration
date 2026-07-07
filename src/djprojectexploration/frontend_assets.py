"""Helpers for rendering packaged frontend assets into standalone HTML."""

from __future__ import annotations

import html
from importlib.resources import files
from typing import Mapping


def frontend_asset_text(relative_path: str) -> str:
    return files("djprojectexploration").joinpath(relative_path).read_text(encoding="utf-8")


def render_standalone_document(
    *,
    title: str,
    css_asset: str,
    body_html: str,
    script_asset: str,
    data_scripts: str = "",
    script_replacements: Mapping[str, str] | None = None,
    head_extra: str = "",
) -> str:
    script = frontend_asset_text(script_asset)
    for old, new in (script_replacements or {}).items():
        script = script.replace(str(old), str(new))

    replacements = {
        "{{TITLE}}": html.escape(str(title)),
        "{{HEAD_EXTRA}}": head_extra,
        "{{CSS}}": frontend_asset_text(css_asset).rstrip(),
        "{{BODY}}": body_html.rstrip(),
        "{{DATA_SCRIPTS}}": data_scripts.rstrip(),
        "{{SCRIPT}}": script.rstrip(),
    }
    document = frontend_asset_text("templates/standalone_document.html")
    for marker, value in replacements.items():
        document = document.replace(marker, value)
    return document
