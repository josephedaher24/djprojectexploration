"""Utilities for noisy native-library warning output."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def suppress_native_stderr(enabled: bool = True) -> Iterator[None]:
    """Temporarily silence stderr written directly by native libraries."""
    if not enabled:
        yield
        return
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        yield
        return

    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
