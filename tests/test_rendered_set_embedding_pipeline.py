from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from djprojectexploration.rendered_set_embedding_pipeline import (
    RenderedSetSection,
    _add_playback_metadata,
    load_rendered_set_cues,
    resolve_maest_analysis_windows,
    resolve_cue_csv_for_audio,
    resolve_rendered_set_sections,
)


class RenderedSetCueTests(unittest.TestCase):
    def _write_csv(self, text: str) -> Path:
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        path = Path(temporary_dir.name) / "cues.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_cues_and_infers_section_ends(self) -> None:
        path = self._write_csv(
            "track_number,track_name,artist,start_seconds\n"
            "1,Opening Track,Artist A,0\n"
            "2,Second Track,Artist B,42.5\n"
        )

        cues = load_rendered_set_cues(path)
        sections = resolve_rendered_set_sections(cues, 100.0)

        self.assertEqual([section.track_number for section in sections], [1, 2])
        self.assertEqual([section.start_seconds for section in sections], [0.0, 42.5])
        self.assertEqual([section.end_seconds for section in sections], [42.5, 100.0])
        self.assertEqual([section.analysis_start_seconds for section in sections], [0.0, 42.5])
        self.assertEqual([section.analysis_end_seconds for section in sections], [42.5, 100.0])

    def test_outro_marker_truncates_previous_track_and_is_dropped(self) -> None:
        path = self._write_csv(
            "track_number,track_name,artist,start_seconds\n"
            "1,Opening Track,Artist A,0\n"
            "2,Final Track,Artist B,100\n"
            "40,OUTRO,Unknown,180\n"
        )

        sections = resolve_rendered_set_sections(load_rendered_set_cues(path), 240.0)

        self.assertEqual([section.track_number for section in sections], [1, 2])
        self.assertEqual([section.end_seconds for section in sections], [100.0, 180.0])

    def test_outro_marker_can_be_kept_as_a_track(self) -> None:
        path = self._write_csv(
            "track_number,track_name,artist,start_seconds\n"
            "1,Opening Track,Artist A,0\n"
            "40,OUTRO,Unknown,180\n"
        )

        sections = resolve_rendered_set_sections(
            load_rendered_set_cues(path), 240.0, drop_final_outro_marker=False
        )

        self.assertEqual([section.track_number for section in sections], [1, 40])
        self.assertEqual([section.end_seconds for section in sections], [180.0, 240.0])

    def test_analysis_guards_scale_down_to_minimum_duration(self) -> None:
        path = self._write_csv(
            "track_number,track_name,artist,start_seconds\n"
            "1,Short Track,Artist A,0\n"
        )

        sections = resolve_rendered_set_sections(
            load_rendered_set_cues(path), 45.0, use_analysis_guards=True
        )

        self.assertAlmostEqual(sections[0].analysis_start_seconds, 6.0)
        self.assertAlmostEqual(sections[0].analysis_end_seconds, 36.0)

    def test_maest_windows_leave_long_sections_unmerged(self) -> None:
        sections = [
            RenderedSetSection(1, "Long Track", "Artist A", 0.0, 60.0),
            RenderedSetSection(2, "Long Track 2", "Artist B", 60.0, 120.0),
        ]

        windows = resolve_maest_analysis_windows(sections, min_duration_seconds=30.0)

        self.assertEqual([window.merge_strategy for window in windows], ["none", "none"])
        self.assertEqual([window.start_seconds for window in windows], [0.0, 60.0])
        self.assertEqual([window.end_seconds for window in windows], [60.0, 120.0])

    def test_maest_windows_merge_short_middle_section_backward(self) -> None:
        sections = [
            RenderedSetSection(1, "Previous", "Artist A", 0.0, 40.0),
            RenderedSetSection(2, "Short", "Artist B", 40.0, 50.0),
            RenderedSetSection(3, "Next", "Artist C", 50.0, 100.0),
        ]

        windows = resolve_maest_analysis_windows(sections, min_duration_seconds=30.0)

        self.assertEqual(windows[1].merge_strategy, "merged_previous")
        self.assertEqual(windows[1].start_seconds, 0.0)
        self.assertEqual(windows[1].end_seconds, 50.0)
        self.assertEqual(windows[1].merged_track_numbers, (1, 2))

    def test_maest_windows_merge_short_first_section_forward(self) -> None:
        sections = [
            RenderedSetSection(1, "Intro", "Artist A", 0.0, 10.0),
            RenderedSetSection(2, "Next", "Artist B", 10.0, 60.0),
        ]

        windows = resolve_maest_analysis_windows(sections, min_duration_seconds=30.0)

        self.assertEqual(windows[0].merge_strategy, "merged_next")
        self.assertEqual(windows[0].start_seconds, 0.0)
        self.assertEqual(windows[0].end_seconds, 60.0)
        self.assertEqual(windows[0].merged_track_numbers, (1, 2))

    def test_maest_windows_can_leave_short_sections_unmerged(self) -> None:
        sections = [RenderedSetSection(1, "Short", "Artist A", 0.0, 10.0)]

        windows = resolve_maest_analysis_windows(
            sections, merge_short_sections=False, min_duration_seconds=30.0
        )

        self.assertEqual(windows[0].merge_strategy, "too_short_unmerged")
        self.assertEqual(windows[0].start_seconds, 0.0)
        self.assertEqual(windows[0].end_seconds, 10.0)

    def test_resolves_same_stem_cue_csv_when_omitted(self) -> None:
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        root = Path(temporary_dir.name)
        audio = root / "rendered_mix.mp3"
        cue_csv = root / "rendered_mix.csv"
        audio.write_bytes(b"placeholder")
        cue_csv.write_text(
            "track_number,track_name,artist,start_seconds\n1,First,Artist A,0\n",
            encoding="utf-8",
        )

        self.assertEqual(resolve_cue_csv_for_audio(audio), cue_csv.resolve())

    def test_explicit_cue_csv_overrides_same_stem_resolution(self) -> None:
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        root = Path(temporary_dir.name)
        audio = root / "rendered_mix.mp3"
        cue_csv = root / "different.csv"

        self.assertEqual(resolve_cue_csv_for_audio(audio, cue_csv), cue_csv.resolve())

    def test_missing_inferred_cue_csv_is_clear(self) -> None:
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        audio = Path(temporary_dir.name) / "missing_cues.mp3"

        with self.assertRaisesRegex(FileNotFoundError, "same-stem CSV"):
            resolve_cue_csv_for_audio(audio)

    def test_rejects_non_increasing_start_times(self) -> None:
        path = self._write_csv(
            "track_number,track_name,artist,start_seconds\n"
            "1,First,Artist A,10\n"
            "2,Second,Artist B,10\n"
        )

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            load_rendered_set_cues(path)

    def test_rejects_missing_columns(self) -> None:
        path = self._write_csv("track_number,track_name,start_seconds\n1,First,0\n")

        with self.assertRaisesRegex(ValueError, "artist"):
            load_rendered_set_cues(path)

    def test_rejects_final_cue_outside_audio(self) -> None:
        path = self._write_csv(
            "track_number,track_name,artist,start_seconds\n1,First,Artist A,100\n"
        )

        with self.assertRaisesRegex(ValueError, "outside"):
            resolve_rendered_set_sections(load_rendered_set_cues(path), 100.0)

    def test_playback_metadata_preserves_full_source_path(self) -> None:
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        root = Path(temporary_dir.name)
        bundle = root / "features.npz"
        source = root / "a-rendered-set-with-a-long-name.mp3"
        np.savez_compressed(bundle, embeddings=np.zeros((1, 2), dtype=np.float32))
        section = RenderedSetSection(1, "Track", "Artist", 12.5, 42.0)

        _add_playback_metadata(bundle, [section], source)

        with np.load(bundle, allow_pickle=False) as payload:
            self.assertEqual(str(payload["audio_paths"][0]), str(source.resolve()))
            self.assertEqual(float(payload["playback_start_seconds"][0]), 12.5)
            self.assertEqual(float(payload["analysis_start_seconds"][0]), 12.5)


if __name__ == "__main__":
    unittest.main()
