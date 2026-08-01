"""Persistent on-disk cache for computed simplex layouts.

The layout functions in :mod:`interactive_pacmap_knn_simplex` are deterministic
functions of their keyword arguments, so their (very expensive) output can be
memoised across process restarts.  ``@cached_layouts`` does exactly that with no
call-site changes.

Deliberate design points:

* This module imports **stdlib + numpy only**.  It must never import pacmap,
  sklearn, numba, umap or plotly, because skipping ``import pacmap`` is part of
  the speedup on a cache hit.  Library versions are read through
  ``importlib.metadata`` which parses distribution metadata without importing.
* The cache key is derived from the decorated function's bound keyword
  arguments via a *generic* recursive walk, plus a transitive source hash of
  every ``djprojectexploration`` function the decorated function can reach.
  There is no hand-maintained field list to go stale.
* The key never involves frontend files (CSS/JS/templates) or the rendered HTML.
  Layouts are purely geometric; the user edits the frontend constantly and every
  restart must still hit.
* Every failure path degrades to "compute it fresh".  A corrupt, truncated,
  foreign or unreadable cache file is a MISS, never a crash and never a wrong
  answer.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import platform
import shutil
import sys
import time
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Bump when the on-disk record layout changes.  Mirrored in the filename prefix
# so stale-format files miss cleanly and can be swept with a single glob.
SCHEMA = 1

# Distributions whose version can move a float bit in the result.
_KEYED_DISTRIBUTIONS = ("pacmap", "numpy", "scikit-learn", "numba", "scipy", "umap-learn")

_PACKAGE = "djprojectexploration"

_source_digest_cache: dict[str, str] = {}
_positional_warned: set[str] = set()
_policy_warned = False


class _Uncanonical(Exception):
    """An argument had a type the key builder does not know how to hash."""


# --------------------------------------------------------------------------- #
# policy / location
# --------------------------------------------------------------------------- #


def _policy() -> str:
    """auto | off | refresh, from $DJPROJECTEXPLORATION_LAYOUT_CACHE."""
    global _policy_warned
    raw = os.environ.get("DJPROJECTEXPLORATION_LAYOUT_CACHE", "").strip().lower()
    if raw in ("", "1", "on", "auto", "true", "yes"):
        return "auto"
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw == "refresh":
        return "refresh"
    if not _policy_warned:
        _policy_warned = True
        print(
            f"[layout-cache] unknown DJPROJECTEXPLORATION_LAYOUT_CACHE={raw!r}; using 'auto'",
            flush=True,
        )
    return "auto"


def _cache_dir() -> Path:
    """Where cache entries live.

    Deliberately NOT under tempfile.gettempdir() (unlike the neighbouring
    ensure_numba_cache_dir): macOS periodically sweeps /var/folders, which would
    randomly reintroduce the exact 60 s startup this cache exists to remove.
    """
    raw = os.environ.get("DJPROJECTEXPLORATION_LAYOUT_CACHE_DIR", "").strip()
    path = Path(raw).expanduser() if raw else PROJECT_ROOT / "data" / "cache" / "layouts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _max_entries() -> int:
    try:
        value = int(os.environ.get("DJPROJECTEXPLORATION_LAYOUT_CACHE_MAX_ENTRIES", "32"))
    except ValueError:
        return 32
    return max(1, value)


# --------------------------------------------------------------------------- #
# key construction
# --------------------------------------------------------------------------- #


def _canon(value: Any) -> Any:
    """Recursively canonicalise a value into JSON-safe, exactly-typed form.

    The isinstance order matters: bool must precede int (bool is an int
    subclass) so ``True`` can never collide with ``1``.
    """
    if isinstance(value, np.ndarray):
        buf = np.ascontiguousarray(value)
        return [
            "ndarray",
            buf.dtype.str,
            list(buf.shape),
            hashlib.sha256(buf.tobytes()).hexdigest(),
        ]
    if value is None:
        return ["none", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        # float.hex() is exact and separates -0.0 from 0.0.
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, list):
        return ["list", [_canon(item) for item in value]]
    if isinstance(value, tuple):
        return ["tuple", [_canon(item) for item in value]]
    if isinstance(value, dict):
        try:
            items = sorted(value.items())
        except TypeError as exc:  # unorderable keys
            raise _Uncanonical("dict with unorderable keys") from exc
        return ["dict", [[str(k), _canon(v)] for k, v in items]]
    if isinstance(value, np.generic):
        return ["np", value.dtype.str, _canon(value.item())]
    raise _Uncanonical(type(value).__name__)


def _closure_functions(root: Callable[..., Any]) -> list[tuple[str, Any]]:
    """Every djprojectexploration function transitively reachable from ``root``."""
    seen: set[str] = set()
    out: list[tuple[str, Any]] = []
    stack = [root]
    while stack:
        fn = stack.pop()
        module = getattr(fn, "__module__", "") or ""
        qualname = f"{module}.{getattr(fn, '__qualname__', getattr(fn, '__name__', '?'))}"
        if qualname in seen:
            continue
        seen.add(qualname)
        out.append((qualname, fn))
        code = getattr(fn, "__code__", None)
        if code is None:
            continue
        globals_ = getattr(fn, "__globals__", {}) or {}
        for name in code.co_names:
            candidate = globals_.get(name)
            if isinstance(candidate, types.FunctionType) and (
                getattr(candidate, "__module__", "") or ""
            ).startswith(_PACKAGE):
                stack.append(candidate)
    return sorted(out, key=lambda item: item[0])


def _source_closure_digest(func: Callable[..., Any]) -> str:
    """sha256 over the source of the whole reachable djprojectexploration closure."""
    root_key = f"{func.__module__}.{func.__qualname__}"
    cached = _source_digest_cache.get(root_key)
    if cached is not None:
        return cached

    functions = _closure_functions(func)
    # Hash the whole source FILES of the reachable closure, not each function's
    # own source text.  Per-function hashing misses anything the math reads but
    # does not lexically contain: a module-level constant, a helper called as
    # `module.helper(...)` rather than by bare name, or a function reached through
    # a dict/registry.  Any of those can change the layouts while leaving the key
    # untouched, which serves a silently stale map - the one failure mode that is
    # worse than being slow.  Cost of the stricter rule: editing an unrelated part
    # of a module in the closure forces one recompute.  That is the right trade
    # here, since these modules change rarely and the frontend (which changes
    # constantly) is not among them.
    digest = hashlib.sha256()
    digest.update(b"file-closure-v2\0")
    files: set[str] = set()
    for qualname, fn in functions:
        digest.update(qualname.encode("utf-8"))
        digest.update(b"\0")
        try:
            files.add(inspect.getfile(fn))
        except (OSError, TypeError):
            continue
    for name in sorted(files):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(Path(name).read_bytes())
        except OSError:
            # Unreadable source: refuse to claim we know what the code is.
            digest.update(uuid.uuid4().bytes)
        digest.update(b"\0")

    value = digest.hexdigest()
    _source_digest_cache[root_key] = value
    return value


def _library_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    libs: dict[str, str] = {}
    for dist in _KEYED_DISTRIBUTIONS:
        try:
            libs[dist] = version(dist)
        except PackageNotFoundError:
            libs[dist] = "absent"
        except Exception:  # pragma: no cover - metadata corruption
            libs[dist] = "unknown"
    libs["python"] = f"{sys.version_info[0]}.{sys.version_info[1]}"
    return libs


def _key_document(func: Callable[..., Any], arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "func": f"{func.__module__}.{func.__qualname__}",
        "source": _source_closure_digest(func),
        "libs": _library_versions(),
        "platform": {"machine": platform.machine(), "sys": sys.platform},
        "args": {name: _canon(value) for name, value in sorted(arguments.items())},
    }


def _dumps(doc: Any) -> str:
    return json.dumps(doc, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _slug(func: Callable[..., Any]) -> str:
    name = func.__name__.strip("_")
    if name.endswith("_layouts"):
        name = name[: -len("_layouts")]
    return name


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #


def _rebuild(keys: list[str], coords: np.ndarray) -> dict[str, list[list[float]]]:
    """The single reconstruction recipe, used by both _load and _store's proof.

    Iterates ``keys`` in stored order so dict insertion order round-trips: two
    consumers depend on it (the opening view via next(iter(...)) and the ordered
    ``seq-layout-entries-json`` array embedded in the HTML).
    """
    return {key: [[float(x), float(y)] for x, y in coords[index]] for index, key in enumerate(keys)}


def _load(path: Path, key_doc: dict[str, Any]) -> dict[str, list[list[float]]] | None:
    """Return the cached layouts, or None for any reason at all."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"].item()))
            if meta.get("schema") != SCHEMA:
                return None
            # Full key document comparison: this is what makes truncating the
            # hash to 32 chars safe, and catches a hand-copied file.
            if _dumps(meta.get("key_doc")) != _dumps(key_doc):
                return None
            coords = archive["coords"]
            keys = [str(k) for k in archive["layout_keys"].tolist()]
        if coords.dtype != np.float32 or coords.ndim != 3 or coords.shape[2] != 2:
            return None
        if len(keys) != coords.shape[0] or len(set(keys)) != len(keys):
            return None
        if not bool(np.isfinite(coords).all()):
            return None
        # Cross-check the payload against the key document and our own meta.  The
        # key_doc match above only proves the INPUTS agree; without this a file
        # whose coords were rewritten (bit rot, a stray tool) is still served as a
        # hit, and a wrong track count wedges the consumer on every restart rather
        # than self-healing.  X_reference canonicalises to
        # ["ndarray", dtype, [n_tracks, n_dims], sha], so shape[1][0] is authoritative.
        expected_tracks = None
        reference = (key_doc.get("args") or {}).get("X_reference")
        if isinstance(reference, list) and len(reference) >= 3 and isinstance(reference[2], list) and reference[2]:
            expected_tracks = int(reference[2][0])
        if expected_tracks is not None and coords.shape[1] != expected_tracks:
            return None
        if int(meta.get("track_count", coords.shape[1])) != coords.shape[1]:
            return None
        if int(meta.get("layout_count", len(keys))) != len(keys):
            return None
        return _rebuild(keys, coords)
    except Exception:
        return None


def _store(
    path: Path,
    key_doc: dict[str, Any],
    layouts: dict[str, list[list[float]]],
    compute_seconds: float,
) -> bool:
    """Atomically write an entry.  Never raises; a failed write is just slow."""
    tmp: Path | None = None
    try:
        if not layouts:
            return False
        keys = list(layouts)
        coords = np.asarray([layouts[k] for k in keys], dtype=np.float32)
        if coords.ndim != 3 or coords.shape[2] != 2:
            return False
        if len(set(keys)) != len(keys):
            return False

        # Write-time round-trip proof: verify that what _load will reconstruct is
        # equal to what we were handed, including insertion order.  Turns
        # "float32 widens to float exactly" into a checked invariant.
        rebuilt = _rebuild(keys, coords)
        if rebuilt != layouts or list(rebuilt) != list(layouts):
            print(
                "[layout-cache] refusing to write: stored form does not round-trip exactly",
                flush=True,
            )
            return False

        meta = {
            "schema": SCHEMA,
            "key_doc": key_doc,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "layout_count": len(keys),
            "track_count": int(coords.shape[1]),
            "compute_seconds": round(float(compute_seconds), 3),
        }

        # A directory (or other non-file) sitting on the entry path makes os.replace
        # fail forever, silently costing a full recompute on every single launch.
        if path.exists() and not path.is_file():
            try:
                shutil.rmtree(path)
            except OSError:
                try:
                    path.unlink()
                except OSError:
                    pass

        # Same directory: os.replace is only atomic within one filesystem.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with open(tmp, "wb") as handle:
            # Pass the file object, not a path, so numpy does not append .npz.
            np.savez(
                handle,
                coords=coords,
                layout_keys=np.array(keys),
                meta=np.array(_dumps(meta)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        # Only rename after a fully flushed, fsynced write.  Renaming after a
        # failed write is what turns an ENOSPC into a permanently poisoned entry.
        os.replace(tmp, path)
        tmp = None
        _prune(path.parent)
        return True
    except Exception as exc:
        print(f"[layout-cache] could not write cache entry ({type(exc).__name__}: {exc})", flush=True)
        return False
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass


def _prune(directory: Path) -> None:
    """True-LRU trim (hits bump mtime via os.utime)."""
    try:
        limit = _max_entries()
        entries = []
        for candidate in directory.glob(f"v{SCHEMA}-*.npz"):
            try:
                entries.append((candidate.stat().st_mtime, candidate))
            except OSError:
                continue
        entries.sort(key=lambda item: item[0], reverse=True)
        for _, stale in entries[limit:]:
            try:
                stale.unlink()
            except (FileNotFoundError, PermissionError, OSError):
                continue
        # A hard kill between fsync and os.replace leaks a full-size .tmp that the
        # glob above never matches.  The age guard leaves a concurrent writer alone.
        for orphan in directory.glob("*.tmp"):
            try:
                if time.time() - orphan.stat().st_mtime > 3600:
                    orphan.unlink()
            except (FileNotFoundError, PermissionError, OSError):
                continue
    except Exception:
        return


# --------------------------------------------------------------------------- #
# miss diagnostics (best-effort; never affects correctness)
# --------------------------------------------------------------------------- #


def _flatten(doc: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in doc.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}.{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


def _miss_reason(path: Path, key_doc: dict[str, Any], slug: str) -> str:
    try:
        if path.exists():
            return "unreadable or mismatched"
        live = _flatten(key_doc)
        candidates = []
        for candidate in path.parent.glob(f"v{SCHEMA}-{slug}-*.npz"):
            try:
                candidates.append((candidate.stat().st_mtime, candidate))
            except OSError:
                continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        best: list[str] | None = None
        for _, candidate in candidates[:3]:
            try:
                with np.load(candidate, allow_pickle=False) as archive:
                    other = _flatten(json.loads(str(archive["meta"].item())).get("key_doc") or {})
            except Exception:
                continue
            changed = sorted(
                name
                for name in set(live) | set(other)
                if _dumps(live.get(name)) != _dumps(other.get(name))
            )
            if changed and (best is None or len(changed) < len(best)):
                best = changed
        if best:
            shown = ", ".join(best[:4]) + (" +more" if len(best) > 4 else "")
            return f"changed: {shown}"
        return "absent"
    except Exception:
        return "absent"


# --------------------------------------------------------------------------- #
# the decorator
# --------------------------------------------------------------------------- #


def cached_layouts(func: Callable[..., dict[str, list[list[float]]]]):
    """Persist a deterministic keyword-only layout function's result to disk."""

    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, list[list[float]]]:
        policy = _policy()
        if policy == "off":
            return func(*args, **kwargs)

        if args:
            name = f"{func.__module__}.{func.__qualname__}"
            if name not in _positional_warned:
                _positional_warned.add(name)
                print(
                    f"[layout-cache] disabled for {func.__name__}: called with positional arguments",
                    flush=True,
                )
            return func(*args, **kwargs)

        try:
            bound = signature.bind(**kwargs)
            bound.apply_defaults()
            key_doc = _key_document(func, dict(bound.arguments))
            serialized = _dumps(key_doc)
        except _Uncanonical as exc:
            print(
                f"[layout-cache] disabled: an argument has uncacheable type {exc}; computing fresh",
                flush=True,
            )
            return func(**kwargs)
        except Exception as exc:
            print(
                f"[layout-cache] disabled ({type(exc).__name__}: {exc}); computing fresh",
                flush=True,
            )
            return func(**kwargs)

        key32 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]
        slug = _slug(func)
        try:
            path = _cache_dir() / f"v{SCHEMA}-{slug}-{key32}.npz"
        except Exception as exc:
            print(
                f"[layout-cache] disabled (cache dir unusable: {exc}); computing fresh",
                flush=True,
            )
            return func(**kwargs)

        if policy != "refresh":
            started = time.perf_counter()
            hit = _load(path, key_doc)
            if hit is not None:
                try:
                    os.utime(path, None)  # keeps pruning true-LRU
                except OSError:
                    pass
                print(
                    f"[layout-cache] HIT  {path.name} "
                    f"({len(hit)} layouts, {time.perf_counter() - started:.2f}s)",
                    flush=True,
                )
                return hit
            reason = _miss_reason(path, key_doc, slug)
        else:
            reason = "refresh requested"

        print(
            f"[layout-cache] MISS ({reason}) - computing layouts, "
            "this is cached so restarts will be fast",
            flush=True,
        )
        started = time.perf_counter()
        layouts = func(**kwargs)
        elapsed = time.perf_counter() - started
        if _store(path, key_doc, layouts, elapsed):
            print(f"[layout-cache] wrote {path.name} ({elapsed:.1f}s)", flush=True)
        # Always return the freshly computed object, never a re-read.
        return layouts

    return wrapper
