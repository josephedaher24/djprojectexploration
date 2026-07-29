"""Extract cue-delimited MAEST embeddings from one rendered DJ set."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from djprojectexploration.native_warnings import suppress_native_stderr
from djprojectexploration.tracklists import PROJECT_ROOT, to_project_relpath


DEFAULT_MODEL_FILE = PROJECT_ROOT / "models" / "discogs-maest-30s-pw-519l-2.pb"
DEFAULT_OUTPUT_NODE = "PartitionedCall/Identity_7"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "maest_embeddings"
MAEST_SAMPLE_RATE = 16000
MAEST_MEL_HOP_SAMPLES = 256
MAEST_PATCH_SIZE_FRAMES = 1876
MAEST_MIN_INPUT_DURATION_SEC = MAEST_MEL_HOP_SAMPLES * MAEST_PATCH_SIZE_FRAMES / MAEST_SAMPLE_RATE
SECTION_SAMPLE_RATE = 44100
DEFAULT_ANALYSIS_HEAD_GUARD_SEC = 20.0
DEFAULT_ANALYSIS_TAIL_GUARD_SEC = 30.0
DEFAULT_MIN_ANALYSIS_DURATION_SEC = 30.0
DEFAULT_USE_ANALYSIS_GUARDS = False
DEFAULT_MERGE_SHORT_MAEST_SECTIONS = True


@dataclass(frozen=True)
class RenderedSetCue:
    """One track start in a rendered set tracklist."""

    track_number: int
    track_name: str
    artist: str
    start_seconds: float


@dataclass(frozen=True)
class RenderedSetSection:
    """A cue resolved to a half-open playback and analysis interval."""

    track_number: int
    track_name: str
    artist: str
    start_seconds: float
    end_seconds: float
    analysis_start_seconds: float = math.nan
    analysis_end_seconds: float = math.nan


@dataclass(frozen=True)
class RenderedSetMaestAnalysisWindow:
    """The source-audio interval actually fed to MAEST for one output row."""

    start_seconds: float
    end_seconds: float
    merge_strategy: str
    merged_track_numbers: tuple[int, ...]


def _section_analysis_start(section: RenderedSetSection) -> float:
    value = float(section.analysis_start_seconds)
    return value if math.isfinite(value) else float(section.start_seconds)


def _section_analysis_end(section: RenderedSetSection) -> float:
    value = float(section.analysis_end_seconds)
    return value if math.isfinite(value) else float(section.end_seconds)


def _duration_seconds(start_seconds: float, end_seconds: float) -> float:
    return float(end_seconds) - float(start_seconds)


def _is_final_outro_marker(cue: RenderedSetCue) -> bool:
    return cue.track_name.strip().casefold() == "outro"


def _analysis_interval(
    start_seconds: float,
    end_seconds: float,
    *,
    head_guard_seconds: float,
    tail_guard_seconds: float,
    min_duration_seconds: float,
) -> tuple[float, float]:
    """Return a guarded representative interval inside playback bounds.

    The preferred interval removes the full head/tail guards. If that would make
    the interval shorter than ``min_duration_seconds``, the guards are scaled
    down proportionally. Very short sections fall back to the whole playback
    interval so extraction still succeeds.
    """
    start = float(start_seconds)
    end = float(end_seconds)
    duration = end - start
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Section end_seconds must be greater than start_seconds.")

    head = max(0.0, float(head_guard_seconds))
    tail = max(0.0, float(tail_guard_seconds))
    min_duration = max(0.0, float(min_duration_seconds))

    guarded_start = start + head
    guarded_end = end - tail
    if guarded_end > guarded_start and guarded_end - guarded_start >= min_duration:
        return guarded_start, guarded_end

    if duration >= min_duration and head + tail > 0.0:
        scale = max(0.0, min(1.0, (duration - min_duration) / (head + tail)))
        scaled_start = start + (head * scale)
        scaled_end = end - (tail * scale)
        if scaled_end > scaled_start:
            return scaled_start, scaled_end

    return start, end


def resolve_maest_analysis_windows(
    sections: list[RenderedSetSection],
    *,
    merge_short_sections: bool = DEFAULT_MERGE_SHORT_MAEST_SECTIONS,
    min_duration_seconds: float = MAEST_MIN_INPUT_DURATION_SEC,
) -> list[RenderedSetMaestAnalysisWindow]:
    """Return MAEST extraction windows, expanding very short sections into neighbors."""
    if not sections:
        raise ValueError("At least one rendered-set section is required.")

    min_duration = max(0.0, float(min_duration_seconds))
    windows: list[RenderedSetMaestAnalysisWindow] = []
    for index, section in enumerate(sections):
        base_start = _section_analysis_start(section)
        base_end = _section_analysis_end(section)
        if not math.isfinite(base_start) or not math.isfinite(base_end) or base_end <= base_start:
            raise ValueError(f"Track {section.track_number} has an invalid analysis interval.")
        base_duration = _duration_seconds(base_start, base_end)
        if not merge_short_sections or base_duration >= min_duration:
            windows.append(
                RenderedSetMaestAnalysisWindow(
                    start_seconds=base_start,
                    end_seconds=base_end,
                    merge_strategy="none" if base_duration >= min_duration else "too_short_unmerged",
                    merged_track_numbers=(section.track_number,),
                )
            )
            continue

        start_index = index
        end_index = index

        # Prefer real preceding context for short rendered-set sections. If that
        # still cannot satisfy the MAEST window, expand forward as well.
        while start_index > 0:
            start_index -= 1
            start = _section_analysis_start(sections[start_index])
            end = _section_analysis_end(sections[end_index])
            if _duration_seconds(start, end) >= min_duration:
                break

        while (
            _duration_seconds(
                _section_analysis_start(sections[start_index]),
                _section_analysis_end(sections[end_index]),
            )
            < min_duration
            and end_index < len(sections) - 1
        ):
            end_index += 1

        start = _section_analysis_start(sections[start_index])
        end = _section_analysis_end(sections[end_index])
        if start_index < index and end_index > index:
            strategy = "merged_previous_next"
        elif start_index < index:
            strategy = "merged_previous"
        elif end_index > index:
            strategy = "merged_next"
        else:
            strategy = "too_short_unmerged"
        if _duration_seconds(start, end) < min_duration:
            strategy = f"{strategy}_still_short"

        windows.append(
            RenderedSetMaestAnalysisWindow(
                start_seconds=start,
                end_seconds=end,
                merge_strategy=strategy,
                merged_track_numbers=tuple(s.track_number for s in sections[start_index : end_index + 1]),
            )
        )
    return windows


def load_rendered_set_cues(cue_csv: str | Path) -> list[RenderedSetCue]:
    """Read and validate ``track_number,track_name,artist,start_seconds`` rows."""
    path = Path(cue_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cue CSV not found: {path}")

    required = {"track_number", "track_name", "artist", "start_seconds"}
    cues: list[RenderedSetCue] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"Cue CSV is missing required columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            try:
                track_number = int((row.get("track_number") or "").strip())
            except ValueError as exc:
                raise ValueError(f"Invalid track_number at CSV row {row_number}.") from exc
            track_name = (row.get("track_name") or "").strip()
            artist = (row.get("artist") or "").strip()
            if not track_name:
                raise ValueError(f"track_name is required at CSV row {row_number}.")
            if not artist:
                raise ValueError(f"artist is required at CSV row {row_number}.")
            try:
                start_seconds = float((row.get("start_seconds") or "").strip())
            except ValueError as exc:
                raise ValueError(f"Invalid start_seconds at CSV row {row_number}.") from exc
            if not math.isfinite(start_seconds) or start_seconds < 0:
                raise ValueError(f"start_seconds must be finite and nonnegative at CSV row {row_number}.")
            cues.append(RenderedSetCue(track_number, track_name, artist, start_seconds))

    if not cues:
        raise ValueError(f"Cue CSV contains no tracks: {path}")
    if len({cue.track_number for cue in cues}) != len(cues):
        raise ValueError("track_number values must be unique.")
    for previous, current in zip(cues, cues[1:], strict=False):
        if current.start_seconds <= previous.start_seconds:
            raise ValueError("start_seconds values must be strictly increasing in CSV order.")
    return cues


def resolve_cue_csv_for_audio(audio_file: str | Path, cue_csv: str | Path | None = None) -> Path:
    """Resolve an explicit cue CSV or infer a same-stem ``.csv`` beside the audio file."""
    if cue_csv is not None:
        return Path(cue_csv).expanduser().resolve()

    inferred = Path(audio_file).expanduser().with_suffix(".csv").resolve()
    if not inferred.exists():
        raise FileNotFoundError(
            "Cue CSV was not provided and no same-stem CSV exists. "
            f"Expected: {inferred}"
        )
    return inferred


def resolve_rendered_set_sections(
    cues: list[RenderedSetCue],
    audio_duration_seconds: float,
    *,
    use_analysis_guards: bool = DEFAULT_USE_ANALYSIS_GUARDS,
    analysis_head_guard_seconds: float = DEFAULT_ANALYSIS_HEAD_GUARD_SEC,
    analysis_tail_guard_seconds: float = DEFAULT_ANALYSIS_TAIL_GUARD_SEC,
    min_analysis_duration_seconds: float = DEFAULT_MIN_ANALYSIS_DURATION_SEC,
    drop_final_outro_marker: bool = True,
) -> list[RenderedSetSection]:
    """Infer each cue's end from the next cue, or from the set duration."""
    if not math.isfinite(audio_duration_seconds) or audio_duration_seconds <= 0:
        raise ValueError("Audio duration must be finite and positive.")
    if not cues:
        raise ValueError("At least one cue is required.")
    if cues[-1].start_seconds >= audio_duration_seconds:
        raise ValueError(
            f"Final cue starts at {cues[-1].start_seconds:.3f}s, outside the "
            f"{audio_duration_seconds:.3f}s audio file."
        )

    section_cues = cues[:-1] if drop_final_outro_marker and _is_final_outro_marker(cues[-1]) else cues
    if not section_cues:
        raise ValueError("Cue CSV contains only a final OUTRO marker and no analyzable tracks.")

    sections: list[RenderedSetSection] = []
    for index, cue in enumerate(section_cues):
        end = cues[index + 1].start_seconds if index + 1 < len(cues) else audio_duration_seconds
        if use_analysis_guards:
            analysis_start, analysis_end = _analysis_interval(
                cue.start_seconds,
                end,
                head_guard_seconds=analysis_head_guard_seconds,
                tail_guard_seconds=analysis_tail_guard_seconds,
                min_duration_seconds=min_analysis_duration_seconds,
            )
        else:
            analysis_start, analysis_end = cue.start_seconds, end
        sections.append(
            RenderedSetSection(
                track_number=cue.track_number,
                track_name=cue.track_name,
                artist=cue.artist,
                start_seconds=cue.start_seconds,
                end_seconds=end,
                analysis_start_seconds=analysis_start,
                analysis_end_seconds=analysis_end,
            )
        )
    return sections


def create_rendered_set_embeddings_npz(
    audio_file: str | Path,
    cue_csv: str | Path | None = None,
    *,
    model_file: str | Path = DEFAULT_MODEL_FILE,
    output_node: str = DEFAULT_OUTPUT_NODE,
    output_file: str | Path | None = None,
    suppress_essentia_warnings: bool = True,
    use_analysis_guards: bool = DEFAULT_USE_ANALYSIS_GUARDS,
    analysis_head_guard_seconds: float = DEFAULT_ANALYSIS_HEAD_GUARD_SEC,
    analysis_tail_guard_seconds: float = DEFAULT_ANALYSIS_TAIL_GUARD_SEC,
    min_analysis_duration_seconds: float = DEFAULT_MIN_ANALYSIS_DURATION_SEC,
    drop_final_outro_marker: bool = True,
    merge_short_maest_sections: bool = DEFAULT_MERGE_SHORT_MAEST_SECTIONS,
) -> Path:
    """Decode a set once, embed its cue-delimited sections, and save one NPZ."""
    from essentia.standard import MonoLoader, TensorflowPredictMAEST

    from djprojectexploration.maest_embedding_extractor import extract_embedding_from_audio

    resolved_audio = Path(audio_file).expanduser().resolve()
    resolved_cues = resolve_cue_csv_for_audio(resolved_audio, cue_csv)
    resolved_model = Path(model_file).expanduser().resolve()
    if not resolved_audio.exists():
        raise FileNotFoundError(f"Rendered set audio not found: {resolved_audio}")
    if not resolved_model.exists():
        raise FileNotFoundError(f"MAEST model not found: {resolved_model}")

    cues = load_rendered_set_cues(resolved_cues)
    audio = np.asarray(
        MonoLoader(filename=str(resolved_audio), sampleRate=MAEST_SAMPLE_RATE, resampleQuality=4)(),
        dtype=np.float32,
    ).reshape(-1)
    if audio.size == 0:
        raise ValueError(f"Rendered set audio is empty: {resolved_audio}")
    duration_seconds = audio.size / float(MAEST_SAMPLE_RATE)
    sections = resolve_rendered_set_sections(
        cues,
        duration_seconds,
        use_analysis_guards=use_analysis_guards,
        analysis_head_guard_seconds=analysis_head_guard_seconds,
        analysis_tail_guard_seconds=analysis_tail_guard_seconds,
        min_analysis_duration_seconds=min_analysis_duration_seconds,
        drop_final_outro_marker=drop_final_outro_marker,
    )
    maest_windows = resolve_maest_analysis_windows(
        sections,
        merge_short_sections=merge_short_maest_sections,
    )

    embeddings: list[np.ndarray] = []
    raw_shapes: list[str] = []
    reductions: list[str] = []
    with suppress_native_stderr(suppress_essentia_warnings):
        model = TensorflowPredictMAEST(graphFilename=str(resolved_model), output=output_node)
        for index, (section, window) in enumerate(zip(sections, maest_windows, strict=True), start=1):
            section_audio = _audio_slice(
                audio,
                sample_rate=MAEST_SAMPLE_RATE,
                start_seconds=window.start_seconds,
                end_seconds=window.end_seconds,
            )
            if section_audio.size == 0:
                raise ValueError(f"Track {section.track_number} resolves to an empty audio section.")
            print(
                f"[{index}/{len(sections)}] Extracting rendered-set section: "
                f"{section.artist} - {section.track_name}"
                + (f" [{window.merge_strategy}]" if window.merge_strategy != "none" else ""),
                flush=True,
            )
            embedding, raw_shape, reduction = extract_embedding_from_audio(
                section_audio, resolved_model, output_node, model=model
            )
            embeddings.append(np.asarray(embedding, dtype=np.float32).reshape(-1))
            raw_shapes.append("x".join(str(value) for value in raw_shape))
            reductions.append(reduction)

    matrix = np.vstack(embeddings).astype(np.float32)
    destination = (
        Path(output_file).expanduser().resolve()
        if output_file is not None
        else (DEFAULT_OUTPUT_DIR / f"{resolved_audio.stem}_sections.npz").resolve()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    track_names = np.asarray([section.track_name for section in sections], dtype=np.str_)
    artists = np.asarray([section.artist for section in sections], dtype=np.str_)
    start_seconds = np.asarray([section.start_seconds for section in sections], dtype=np.float64)
    end_seconds = np.asarray([section.end_seconds for section in sections], dtype=np.float64)
    analysis_start_seconds = np.asarray([_section_analysis_start(section) for section in sections], dtype=np.float64)
    analysis_end_seconds = np.asarray([_section_analysis_end(section) for section in sections], dtype=np.float64)
    maest_start_seconds = np.asarray([window.start_seconds for window in maest_windows], dtype=np.float64)
    maest_end_seconds = np.asarray([window.end_seconds for window in maest_windows], dtype=np.float64)
    audio_path = to_project_relpath(resolved_audio)
    np.savez_compressed(
        destination,
        schema_version=np.array(1, dtype=np.int32),
        embedding_type=np.array("maest_rendered_set_sections", dtype=np.str_),
        created_utc=np.array(datetime.now(tz=timezone.utc).isoformat(), dtype=np.str_),
        source_audio_path=np.array(audio_path, dtype=np.str_),
        cue_csv=np.array(to_project_relpath(resolved_cues), dtype=np.str_),
        maest_model_file=np.array(to_project_relpath(resolved_model), dtype=np.str_),
        maest_output_node=np.array(output_node, dtype=np.str_),
        sample_rate=np.array(MAEST_SAMPLE_RATE, dtype=np.int32),
        audio_duration_seconds=np.array(duration_seconds, dtype=np.float64),
        num_tracks=np.array(len(sections), dtype=np.int32),
        embedding_dimension=np.array(matrix.shape[1], dtype=np.int32),
        embeddings=matrix,
        track_numbers=np.asarray([section.track_number for section in sections], dtype=np.int32),
        track_names=track_names,
        titles=track_names,
        artists=artists,
        audio_paths=np.asarray([audio_path] * len(sections), dtype=np.str_),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        section_durations_seconds=end_seconds - start_seconds,
        playback_start_seconds=start_seconds,
        playback_end_seconds=end_seconds,
        analysis_start_seconds=analysis_start_seconds,
        analysis_end_seconds=analysis_end_seconds,
        analysis_duration_seconds=analysis_end_seconds - analysis_start_seconds,
        maest_analysis_start_seconds=maest_start_seconds,
        maest_analysis_end_seconds=maest_end_seconds,
        maest_analysis_duration_seconds=maest_end_seconds - maest_start_seconds,
        maest_merge_strategy=np.asarray([window.merge_strategy for window in maest_windows], dtype=np.str_),
        maest_merged_track_numbers=np.asarray(
            [",".join(str(number) for number in window.merged_track_numbers) for window in maest_windows],
            dtype=np.str_,
        ),
        maest_merged_track_count=np.asarray(
            [len(window.merged_track_numbers) for window in maest_windows],
            dtype=np.int32,
        ),
        config_use_analysis_guards=np.array(bool(use_analysis_guards), dtype=np.bool_),
        config_analysis_head_guard_seconds=np.array(float(analysis_head_guard_seconds), dtype=np.float32),
        config_analysis_tail_guard_seconds=np.array(float(analysis_tail_guard_seconds), dtype=np.float32),
        config_min_analysis_duration_seconds=np.array(float(min_analysis_duration_seconds), dtype=np.float32),
        config_drop_final_outro_marker=np.array(bool(drop_final_outro_marker), dtype=np.bool_),
        config_merge_short_maest_sections=np.array(bool(merge_short_maest_sections), dtype=np.bool_),
        config_maest_min_input_duration_seconds=np.array(float(MAEST_MIN_INPUT_DURATION_SEC), dtype=np.float32),
        raw_prediction_shapes=np.asarray(raw_shapes, dtype=np.str_),
        embedding_reductions=np.asarray(reductions, dtype=np.str_),
    )
    print(f"Saved rendered-set embeddings: {to_project_relpath(destination)}")
    return destination


def _slug(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in normalized.split("-") if part) or "rendered-set"


def _audio_slice(
    audio: np.ndarray,
    *,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
) -> np.ndarray:
    if audio.size <= 0:
        raise ValueError("Cannot slice empty rendered-set audio.")
    start = max(0, min(audio.size - 1, int(round(float(start_seconds) * sample_rate))))
    end = max(start + 1, min(audio.size, int(round(float(end_seconds) * sample_rate))))
    return audio[start:end]


def _add_playback_metadata(path: Path, sections: list[RenderedSetSection], source_audio: Path) -> None:
    """Add original-set playback and analysis coordinates to an existing NPZ artifact."""
    with np.load(path, allow_pickle=False) as bundle:
        payload = {name: bundle[name] for name in bundle.files}
    count = len(sections)
    source = to_project_relpath(source_audio)
    playback_start = np.asarray([section.start_seconds for section in sections], dtype=np.float64)
    playback_end = np.asarray([section.end_seconds for section in sections], dtype=np.float64)
    analysis_start = np.asarray([_section_analysis_start(section) for section in sections], dtype=np.float64)
    analysis_end = np.asarray([_section_analysis_end(section) for section in sections], dtype=np.float64)
    payload.update(
        {
            "rendered_set_source_audio_path": np.array(source, dtype=np.str_),
            "audio_paths": np.asarray([source] * count, dtype=np.str_),
            "playback_start_seconds": playback_start,
            "playback_end_seconds": playback_end,
            "playback_duration_seconds": playback_end - playback_start,
            "analysis_start_seconds": analysis_start,
            "analysis_end_seconds": analysis_end,
            "analysis_duration_seconds": analysis_end - analysis_start,
        }
    )
    np.savez_compressed(path, **payload)


def _add_maest_window_metadata(
    path: Path,
    maest_windows: list[RenderedSetMaestAnalysisWindow],
    *,
    merge_short_maest_sections: bool,
) -> None:
    """Add MAEST-specific merged analysis coordinates to an existing NPZ artifact."""
    with np.load(path, allow_pickle=False) as bundle:
        payload = {name: bundle[name] for name in bundle.files}
    count = len(maest_windows)
    maest_start = np.asarray([window.start_seconds for window in maest_windows], dtype=np.float64)
    maest_end = np.asarray([window.end_seconds for window in maest_windows], dtype=np.float64)
    payload.update(
        {
            "maest_analysis_start_seconds": maest_start,
            "maest_analysis_end_seconds": maest_end,
            "maest_analysis_duration_seconds": maest_end - maest_start,
            "maest_merge_strategy": np.asarray([window.merge_strategy for window in maest_windows], dtype=np.str_),
            "maest_merged_track_numbers": np.asarray(
                [",".join(str(number) for number in window.merged_track_numbers) for window in maest_windows],
                dtype=np.str_,
            ),
            "maest_merged_track_count": np.asarray(
                [len(window.merged_track_numbers) for window in maest_windows],
                dtype=np.int32,
            ),
            "config_merge_short_maest_sections": np.array(
                bool(merge_short_maest_sections),
                dtype=np.bool_,
            ),
            "config_maest_min_input_duration_seconds": np.array(
                float(MAEST_MIN_INPUT_DURATION_SEC),
                dtype=np.float32,
            ),
        }
    )
    if "maest_peak_start_sec" in payload and "maest_peak_end_sec" in payload:
        peak_start = np.asarray(payload["maest_peak_start_sec"], dtype=np.float64)
        peak_end = np.asarray(payload["maest_peak_end_sec"], dtype=np.float64)
        if peak_start.shape[0] == count and peak_end.shape[0] == count:
            payload["maest_peak_source_start_seconds"] = maest_start + peak_start
            payload["maest_peak_source_end_seconds"] = maest_start + peak_end
    np.savez_compressed(path, **payload)


def build_rendered_set_dataset(
    audio_file: str | Path,
    cue_csv: str | Path | None = None,
    *,
    name: str | None = None,
    maest_model_file: str | Path = DEFAULT_MODEL_FILE,
    maest_output_node: str = DEFAULT_OUTPUT_NODE,
    energy_model_file: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
    suppress_essentia_warnings: bool = True,
    use_analysis_guards: bool = DEFAULT_USE_ANALYSIS_GUARDS,
    analysis_head_guard_seconds: float = DEFAULT_ANALYSIS_HEAD_GUARD_SEC,
    analysis_tail_guard_seconds: float = DEFAULT_ANALYSIS_TAIL_GUARD_SEC,
    min_analysis_duration_seconds: float = DEFAULT_MIN_ANALYSIS_DURATION_SEC,
    drop_final_outro_marker: bool = True,
    merge_short_maest_sections: bool = DEFAULT_MERGE_SHORT_MAEST_SECTIONS,
) -> dict[str, Path]:
    """Generate the complete feature suite from cue-delimited temporary sections.

    Section FLAC files exist only for the duration of extraction. Saved artifacts
    are annotated with the original rendered audio path and absolute cue times.
    """
    import soundfile as sf
    from essentia.standard import MonoLoader

    from djprojectexploration.energy_features import (
        DEFAULT_FROZEN_ENERGY_MODEL_FILE,
        create_energy_embedding_npz,
        create_energy_feature_csv,
    )
    from djprojectexploration.playlist_embedding_pipeline import (
        create_chroma_playlist_embeddings_npz,
        create_deam_playlist_embeddings_npz,
        create_groove_playlist_embeddings_npz,
        create_maest_playlist_embeddings_npz,
        create_tempo_playlist_embeddings_npz,
    )
    from djprojectexploration.waveform_features import create_waveform_playlist_features_npz

    root = project_root.expanduser().resolve()
    source_audio = Path(audio_file).expanduser().resolve()
    source_cues = resolve_cue_csv_for_audio(source_audio, cue_csv)
    if not source_audio.exists():
        raise FileNotFoundError(f"Rendered set audio not found: {source_audio}")
    cues = load_rendered_set_cues(source_cues)
    slug = _slug(name or source_audio.stem)
    stem = f"{slug.replace('-', '_')}_tracks"
    resolved_energy_model = (
        Path(energy_model_file).expanduser().resolve()
        if energy_model_file is not None
        else root / "data" / "energy_models" / DEFAULT_FROZEN_ENERGY_MODEL_FILE.name
    )
    if not resolved_energy_model.exists():
        raise FileNotFoundError(f"Default energy model artifact not found: {resolved_energy_model}")

    print("[stage] decode rendered set", flush=True)
    audio = np.asarray(
        MonoLoader(filename=str(source_audio), sampleRate=SECTION_SAMPLE_RATE, resampleQuality=4)(),
        dtype=np.float32,
    ).reshape(-1)
    sections = resolve_rendered_set_sections(
        cues,
        audio.size / float(SECTION_SAMPLE_RATE),
        use_analysis_guards=use_analysis_guards,
        analysis_head_guard_seconds=analysis_head_guard_seconds,
        analysis_tail_guard_seconds=analysis_tail_guard_seconds,
        min_analysis_duration_seconds=min_analysis_duration_seconds,
        drop_final_outro_marker=drop_final_outro_marker,
    )
    maest_windows = resolve_maest_analysis_windows(
        sections,
        merge_short_sections=merge_short_maest_sections,
    )

    output_paths = {
        "waveforms": root / "data" / "waveform_features" / f"{stem}.npz",
        "maest": root / "data" / "maest_embeddings" / f"{stem}.npz",
        "maest_peak30": root / "data" / "maest_embeddings" / f"{stem}_peak30.npz",
        "genre_annotations": root / "data" / "annotations" / f"{stem}__genre_discogs519_maest30pw519l_v1.npz",
        "chroma": root / "data" / "chroma_embeddings" / f"{stem}.npz",
        "key_annotations": root / "data" / "annotations" / f"{stem}__key_chroma_hpcp_v1.npz",
        "tempo": root / "data" / "tempo_embeddings" / f"{stem}.npz",
        "groove": root / "data" / "groove_embeddings" / f"{stem}.npz",
        "deam": root / "data" / "deam_embeddings" / f"{stem}.npz",
        "energy_features": root / "data" / "energy_features" / f"{stem}_energy_features.csv",
        "energy_npz": root / "data" / "energy_embeddings" / f"{slug.replace('-', '_')}_energy_features.npz",
    }

    with tempfile.TemporaryDirectory(prefix="rendered-set-sections-") as temporary:
        temp_root = Path(temporary)
        analysis_root = temp_root / "analysis"
        maest_analysis_root = temp_root / "maest-analysis"
        playback_root = temp_root / "playback"
        analysis_root.mkdir()
        maest_analysis_root.mkdir()
        playback_root.mkdir()
        analysis_csv = temp_root / f"{stem}.analysis.csv"
        maest_analysis_csv = temp_root / f"{stem}.maest-analysis.csv"
        playback_csv = temp_root / f"{stem}.playback.csv"
        analysis_rows: list[dict[str, object]] = []
        maest_analysis_rows: list[dict[str, object]] = []
        playback_rows: list[dict[str, object]] = []
        print("[stage] materialize temporary lossless sections", flush=True)
        for section, maest_window in zip(sections, maest_windows, strict=True):
            playback_file = playback_root / f"{section.track_number:04d}.flac"
            analysis_file = analysis_root / f"{section.track_number:04d}.flac"
            sf.write(
                playback_file,
                _audio_slice(
                    audio,
                    sample_rate=SECTION_SAMPLE_RATE,
                    start_seconds=section.start_seconds,
                    end_seconds=section.end_seconds,
                ),
                SECTION_SAMPLE_RATE,
                format="FLAC",
            )
            sf.write(
                analysis_file,
                _audio_slice(
                    audio,
                    sample_rate=SECTION_SAMPLE_RATE,
                    start_seconds=_section_analysis_start(section),
                    end_seconds=_section_analysis_end(section),
                ),
                SECTION_SAMPLE_RATE,
                format="FLAC",
            )
            common = {
                "track_number": section.track_number,
                "title": section.track_name,
                "artists": section.artist,
                "mix_name": slug,
            }
            playback_rows.append(
                {
                    **common,
                    "filepath": str(playback_file),
                }
            )
            analysis_rows.append(
                {
                    **common,
                    "filepath": str(analysis_file),
                }
            )
            maest_matches_default_analysis = math.isclose(
                maest_window.start_seconds,
                _section_analysis_start(section),
                abs_tol=1e-6,
            ) and math.isclose(
                maest_window.end_seconds,
                _section_analysis_end(section),
                abs_tol=1e-6,
            )
            if maest_matches_default_analysis:
                maest_file = analysis_file
            else:
                maest_file = maest_analysis_root / f"{section.track_number:04d}.flac"
                print(
                    "[merge] MAEST window for "
                    f"{section.track_number}: {maest_window.merge_strategy} "
                    f"({','.join(str(number) for number in maest_window.merged_track_numbers)})",
                    flush=True,
                )
                sf.write(
                    maest_file,
                    _audio_slice(
                        audio,
                        sample_rate=SECTION_SAMPLE_RATE,
                        start_seconds=maest_window.start_seconds,
                        end_seconds=maest_window.end_seconds,
                    ),
                    SECTION_SAMPLE_RATE,
                    format="FLAC",
                )
            maest_analysis_rows.append(
                {
                    **common,
                    "filepath": str(maest_file),
                }
            )
        del audio
        with playback_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["track_number", "title", "artists", "filepath", "mix_name"]
            )
            writer.writeheader()
            writer.writerows(playback_rows)
        with analysis_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["track_number", "title", "artists", "filepath", "mix_name"]
            )
            writer.writeheader()
            writer.writerows(analysis_rows)
        with maest_analysis_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["track_number", "title", "artists", "filepath", "mix_name"]
            )
            writer.writeheader()
            writer.writerows(maest_analysis_rows)

        print("[stage] waveforms", flush=True)
        create_waveform_playlist_features_npz(
            playback_csv,
            output_file=output_paths["waveforms"],
            json_dir=root / "data" / "waveform_features" / slug,
            mix_slug=slug,
            force=True,
        )
        print("[stage] MAEST full", flush=True)
        create_maest_playlist_embeddings_npz(
            maest_analysis_csv, model_file=maest_model_file, output_node=maest_output_node,
            output_file=output_paths["maest"], enrich_genre=True,
            genre_annotation_file=output_paths["genre_annotations"],
            suppress_essentia_warnings=suppress_essentia_warnings,
        )
        print("[stage] MAEST peak30", flush=True)
        create_maest_playlist_embeddings_npz(
            maest_analysis_csv, model_file=maest_model_file, output_node=maest_output_node,
            output_file=output_paths["maest_peak30"], section="peak30", enrich_genre=False,
            suppress_essentia_warnings=suppress_essentia_warnings,
        )
        print("[stage] chroma/key", flush=True)
        create_chroma_playlist_embeddings_npz(
            analysis_csv, output_file=output_paths["chroma"], enrich_key=True,
            key_annotation_file=output_paths["key_annotations"],
        )
        print("[stage] tempo", flush=True)
        create_tempo_playlist_embeddings_npz(
            analysis_csv, output_file=output_paths["tempo"],
            suppress_essentia_warnings=suppress_essentia_warnings,
        )
        print("[stage] groove", flush=True)
        create_groove_playlist_embeddings_npz(
            analysis_csv, output_file=output_paths["groove"], tempo_embeddings=output_paths["tempo"],
            suppress_essentia_warnings=suppress_essentia_warnings,
        )
        print("[stage] DEAM valence/arousal", flush=True)
        create_deam_playlist_embeddings_npz(analysis_csv, output_file=output_paths["deam"])
        print("[stage] energy features", flush=True)
        create_energy_feature_csv(
            analysis_csv, output_file=output_paths["energy_features"], tempo_embeddings=output_paths["tempo"],
            suppress_essentia_warnings=suppress_essentia_warnings,
        )
        print("[stage] default-model energy predictions", flush=True)
        create_energy_embedding_npz(
            [output_paths["energy_features"]],
            output_file=output_paths["energy_npz"],
            name=f"{slug.replace('-', '_')}_energy_features",
            model_file=resolved_energy_model,
            maest_dir=root / "data" / "maest_embeddings",
        )

    for key in (
        "waveforms", "maest", "maest_peak30", "genre_annotations", "chroma",
        "key_annotations", "tempo", "groove", "deam", "energy_npz",
    ):
        _add_playback_metadata(output_paths[key], sections, source_audio)
    for key in ("maest", "maest_peak30", "genre_annotations", "energy_npz"):
        _add_maest_window_metadata(
            output_paths[key],
            maest_windows,
            merge_short_maest_sections=merge_short_maest_sections,
        )

    # Preserve absolute cue playback metadata alongside deterministic audio features.
    energy_rows: list[dict[str, str]] = []
    with output_paths["energy_features"].open("r", encoding="utf-8", newline="") as handle:
        energy_rows = list(csv.DictReader(handle))
    fields = list(energy_rows[0]) if energy_rows else []
    for field in (
        "source_audio_path",
        "playback_start_seconds",
        "playback_end_seconds",
        "playback_duration_seconds",
        "analysis_start_seconds",
        "analysis_end_seconds",
        "analysis_duration_seconds",
    ):
        if field not in fields:
            fields.append(field)
    for row, section in zip(energy_rows, sections, strict=True):
        row["source_audio_path"] = to_project_relpath(source_audio)
        if "audio_path" in row:
            row["audio_path"] = to_project_relpath(source_audio)
        if "filepath" in row:
            row["filepath"] = to_project_relpath(source_audio)
        row["playback_start_seconds"] = str(section.start_seconds)
        row["playback_end_seconds"] = str(section.end_seconds)
        row["playback_duration_seconds"] = str(section.end_seconds - section.start_seconds)
        row["analysis_start_seconds"] = str(_section_analysis_start(section))
        row["analysis_end_seconds"] = str(_section_analysis_end(section))
        row["analysis_duration_seconds"] = str(_section_analysis_end(section) - _section_analysis_start(section))
    with output_paths["energy_features"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(energy_rows)

    with np.load(output_paths["waveforms"], allow_pickle=False) as waveform_bundle:
        waveform_json_paths = [str(value) for value in waveform_bundle["waveform_json_paths"]]
    for section, waveform_json_value in zip(sections, waveform_json_paths, strict=True):
        waveform_json = Path(waveform_json_value)
        if not waveform_json.is_absolute():
            waveform_json = root / waveform_json
        waveform_payload = json.loads(waveform_json.read_text(encoding="utf-8"))
        waveform_payload["source_audio_path"] = to_project_relpath(source_audio)
        waveform_payload["playback_start_seconds"] = section.start_seconds
        waveform_payload["playback_end_seconds"] = section.end_seconds
        waveform_payload["playback_duration_seconds"] = section.end_seconds - section.start_seconds
        waveform_payload["analysis_start_seconds"] = _section_analysis_start(section)
        waveform_payload["analysis_end_seconds"] = _section_analysis_end(section)
        waveform_payload["analysis_duration_seconds"] = _section_analysis_end(section) - _section_analysis_start(section)
        waveform_json.write_text(
            json.dumps(waveform_payload, separators=(",", ":")) + "\n", encoding="utf-8"
        )

    manifest_path = root / "data" / "exports" / f"{slug}_rendered_set_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": slug,
                "source_audio_path": to_project_relpath(source_audio),
                "cue_csv": to_project_relpath(source_cues),
                "energy_model_file": to_project_relpath(resolved_energy_model),
                "temporary_section_files_retained": False,
                "use_analysis_guards": bool(use_analysis_guards),
                "analysis_head_guard_seconds": float(analysis_head_guard_seconds),
                "analysis_tail_guard_seconds": float(analysis_tail_guard_seconds),
                "min_analysis_duration_seconds": float(min_analysis_duration_seconds),
                "drop_final_outro_marker": bool(drop_final_outro_marker),
                "merge_short_maest_sections": bool(merge_short_maest_sections),
                "maest_min_input_duration_seconds": float(MAEST_MIN_INPUT_DURATION_SEC),
                "dropped_final_outro_marker": (
                    {
                        "track_number": cues[-1].track_number,
                        "track_name": cues[-1].track_name,
                        "artist": cues[-1].artist,
                        "start_seconds": cues[-1].start_seconds,
                    }
                    if drop_final_outro_marker and _is_final_outro_marker(cues[-1])
                    else None
                ),
                "tracks": [
                    {
                        "track_number": section.track_number,
                        "track_name": section.track_name,
                        "artist": section.artist,
                        "playback_start_seconds": section.start_seconds,
                        "playback_end_seconds": section.end_seconds,
                        "playback_duration_seconds": section.end_seconds - section.start_seconds,
                        "analysis_start_seconds": _section_analysis_start(section),
                        "analysis_end_seconds": _section_analysis_end(section),
                        "analysis_duration_seconds": _section_analysis_end(section) - _section_analysis_start(section),
                        "maest_analysis_start_seconds": maest_window.start_seconds,
                        "maest_analysis_end_seconds": maest_window.end_seconds,
                        "maest_analysis_duration_seconds": maest_window.end_seconds - maest_window.start_seconds,
                        "maest_merge_strategy": maest_window.merge_strategy,
                        "maest_merged_track_numbers": list(maest_window.merged_track_numbers),
                    }
                    for section, maest_window in zip(sections, maest_windows, strict=True)
                ],
                "outputs": {key: to_project_relpath(path) for key, path in output_paths.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    output_paths["manifest"] = manifest_path
    print(f"Saved rendered-set manifest: {to_project_relpath(manifest_path)}", flush=True)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the complete feature suite for cue-delimited tracks in a rendered set."
    )
    parser.add_argument("audio_file", type=Path, help="Rendered set audio file.")
    parser.add_argument(
        "cue_csv",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "CSV with track_number, track_name, artist, start_seconds. "
            "Defaults to the audio path with a .csv extension."
        ),
    )
    parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--output-node", default=DEFAULT_OUTPUT_NODE)
    parser.add_argument("--output", type=Path, default=None, help="Output path used with --maest-only.")
    parser.add_argument("--name", default=None, help="Dataset slug; defaults to the audio filename.")
    parser.add_argument("--energy-model-file", type=Path, default=None)
    parser.add_argument(
        "--maest-only", action="store_true", help="Generate only the legacy single MAEST section bundle."
    )
    parser.add_argument(
        "--analysis-guards",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_ANALYSIS_GUARDS,
        help=(
            "Use guarded analysis windows for embeddings/features. Disabled by default; "
            "when enabled, guard durations are controlled by the analysis guard options."
        ),
    )
    parser.add_argument(
        "--analysis-head-guard-sec",
        type=float,
        default=DEFAULT_ANALYSIS_HEAD_GUARD_SEC,
        help="Seconds to trim from each cue section start for analysis/embedding extraction.",
    )
    parser.add_argument(
        "--analysis-tail-guard-sec",
        type=float,
        default=DEFAULT_ANALYSIS_TAIL_GUARD_SEC,
        help="Seconds to trim from each cue section end for analysis/embedding extraction.",
    )
    parser.add_argument(
        "--min-analysis-duration-sec",
        type=float,
        default=DEFAULT_MIN_ANALYSIS_DURATION_SEC,
        help="Minimum analysis duration before guards are scaled down. Shorter sections use full playback.",
    )
    parser.add_argument(
        "--keep-final-outro",
        action="store_true",
        help="Keep a final cue named OUTRO as a real track instead of using it only as an end marker.",
    )
    parser.add_argument(
        "--merge-short-maest-sections",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MERGE_SHORT_MAEST_SECTIONS,
        help=(
            "For MAEST only, expand cue sections shorter than the model's minimum "
            "input duration into neighboring real audio. Enabled by default."
        ),
    )
    parser.add_argument("--show-essentia-warnings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.maest_only:
        build_rendered_set_dataset(
            args.audio_file,
            args.cue_csv,
            name=args.name,
            maest_model_file=args.model_file,
            maest_output_node=args.output_node,
            energy_model_file=args.energy_model_file,
            suppress_essentia_warnings=not args.show_essentia_warnings,
            use_analysis_guards=args.analysis_guards,
            analysis_head_guard_seconds=args.analysis_head_guard_sec,
            analysis_tail_guard_seconds=args.analysis_tail_guard_sec,
            min_analysis_duration_seconds=args.min_analysis_duration_sec,
            drop_final_outro_marker=not args.keep_final_outro,
            merge_short_maest_sections=args.merge_short_maest_sections,
        )
        return
    create_rendered_set_embeddings_npz(
        args.audio_file,
        args.cue_csv,
        model_file=args.model_file,
        output_node=args.output_node,
        output_file=args.output,
        suppress_essentia_warnings=not args.show_essentia_warnings,
        use_analysis_guards=args.analysis_guards,
        analysis_head_guard_seconds=args.analysis_head_guard_sec,
        analysis_tail_guard_seconds=args.analysis_tail_guard_sec,
        min_analysis_duration_seconds=args.min_analysis_duration_sec,
        drop_final_outro_marker=not args.keep_final_outro,
        merge_short_maest_sections=args.merge_short_maest_sections,
    )


if __name__ == "__main__":
    main()
