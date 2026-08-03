from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from djprojectexploration.transition_preview import prepare_live_transition


class LiveTransitionPreparationTests(unittest.TestCase):
    def _tracklist(self, root: Path) -> Path:
        source_a = root / "a.wav"
        source_b = root / "b.wav"
        source_a.touch()
        source_b.touch()
        tracklist = root / "tracks.csv"
        with tracklist.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["track_number", "title", "artists", "mp3_name", "filepath", "bpm", "onset-time"],
            )
            writer.writeheader()
            writer.writerow({"track_number": 1, "title": "A", "artists": "Test", "mp3_name": "a.wav", "filepath": str(source_a), "bpm": 120, "onset-time": 0})
            writer.writerow({"track_number": 2, "title": "B", "artists": "Test", "mp3_name": "b.wav", "filepath": str(source_b), "bpm": 120, "onset-time": 0})
        return tracklist

    @staticmethod
    def _stretch(_audio: np.ndarray, *, target_samples: int, **_kwargs: object) -> np.ndarray:
        return np.zeros((2, target_samples), dtype=np.float32)

    def test_prepared_pair_is_reused_and_pitch_changes_invalidate_it(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            tracklist = self._tracklist(root)
            with (
                patch("djprojectexploration.transition_preview._load_segment_with_padding", return_value=np.zeros((2, 8), dtype=np.float32)),
                patch("djprojectexploration.transition_preview._time_stretch_multichannel", side_effect=self._stretch) as stretch,
            ):
                first = prepare_live_transition(
                    tracklist_csv=tracklist,
                    from_query="1",
                    to_query="2",
                    output_dir=root / "out",
                    overlap_bars=2,
                    guard_bars=4,
                )
                reused = prepare_live_transition(
                    tracklist_csv=tracklist,
                    from_query="1",
                    to_query="2",
                    output_dir=root / "out",
                    overlap_bars=2,
                    guard_bars=4,
                )
                changed = prepare_live_transition(
                    tracklist_csv=tracklist,
                    from_query="1",
                    to_query="2",
                    output_dir=root / "out",
                    overlap_bars=2,
                    guard_bars=4,
                    to_pitch_shift=1,
                )
                retimed = prepare_live_transition(
                    tracklist_csv=tracklist,
                    from_query="1",
                    to_query="2",
                    output_dir=root / "out",
                    overlap_bars=2,
                    guard_bars=4,
                    from_nudge_beats=1,
                )
                fractional_jump = prepare_live_transition(
                    tracklist_csv=tracklist,
                    from_query="1",
                    to_query="2",
                    output_dir=root / "out",
                    overlap_bars=2,
                    guard_bars=4,
                    from_nudge_beats=0.125,
                )

            self.assertEqual(first.transition_id, reused.transition_id)
            self.assertNotEqual(first.transition_id, changed.transition_id)
            self.assertNotEqual(first.transition_id, retimed.transition_id)
            self.assertNotEqual(first.transition_id, fractional_jump.transition_id)
            self.assertEqual(stretch.call_count, 8)
            self.assertAlmostEqual(first.loop_start_seconds, 8.0)
            self.assertAlmostEqual(first.loop_end_seconds, 12.0)
            self.assertTrue(first.from_path.exists())
            self.assertTrue(first.to_path.exists())

    def test_transport_window_includes_padding_and_marks_overlap_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            tracklist = self._tracklist(root)
            with (
                patch("djprojectexploration.transition_preview._load_segment_with_padding", return_value=np.zeros((2, 8), dtype=np.float32)),
                patch("djprojectexploration.transition_preview._time_stretch_multichannel", side_effect=self._stretch) as stretch,
            ):
                prepared = prepare_live_transition(
                    tracklist_csv=tracklist,
                    from_query="1",
                    to_query="2",
                    output_dir=root / "out",
                    overlap_bars=2,
                    front_padding_bars=1,
                    back_padding_bars=3,
                    guard_bars=4,
                )
                changed_padding = prepare_live_transition(
                    tracklist_csv=tracklist,
                    from_query="1",
                    to_query="2",
                    output_dir=root / "out",
                    overlap_bars=2,
                    front_padding_bars=2,
                    back_padding_bars=3,
                    guard_bars=4,
                )

            self.assertAlmostEqual(prepared.transport_start_seconds, 8.0)
            self.assertAlmostEqual(prepared.transition_start_seconds, 10.0)
            self.assertAlmostEqual(prepared.transition_end_seconds, 14.0)
            self.assertAlmostEqual(prepared.transport_end_seconds, 20.0)
            self.assertEqual(prepared.loop_start_seconds, prepared.transport_start_seconds)
            self.assertEqual(prepared.loop_end_seconds, prepared.transport_end_seconds)
            self.assertNotEqual(prepared.transition_id, changed_padding.transition_id)
            self.assertEqual(stretch.call_count, 4)
