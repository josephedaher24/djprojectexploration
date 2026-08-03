from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from djprojectexploration.loudness_matching import (
    DEFAULT_LOUDNESS_MATCH_MODE,
    LoudnessMeasurement,
    load_loudness_catalog,
    lookup_loudness,
)
from djprojectexploration.tracklists import PlaylistTrack
from djprojectexploration.transition_preview import _transition_key


class LoudnessMatchingTests(unittest.TestCase):
    def test_integrated_loudness_uses_bounded_gain_and_missing_data_is_neutral(self) -> None:
        self.assertEqual(LoudnessMeasurement(integrated_lufs=-30).gain_db(), 6.0)
        self.assertEqual(LoudnessMeasurement(integrated_lufs=-3).gain_db(), -6.0)
        self.assertAlmostEqual(LoudnessMeasurement(integrated_lufs=-10).gain_db(), -2.0)
        self.assertEqual(LoudnessMeasurement().gain_db(DEFAULT_LOUDNESS_MATCH_MODE), 0.0)
        self.assertEqual(LoudnessMeasurement(integrated_lufs=-30).gain_db("off"), 0.0)
        self.assertEqual(LoudnessMeasurement(integrated_lufs=-30).gain_db("auditions"), 0.0)

    def test_catalog_prefers_filename_and_falls_back_to_track_number(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            tracklist = root / "tracks.csv"
            tracklist.write_text("track_number,mp3_name\n1,one.wav\n", encoding="utf-8")
            feature_dir = root / "data" / "energy_features"
            feature_dir.mkdir(parents=True)
            with (feature_dir / "tracks_energy_features.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["track_number", "filename", "full_integrated_loudness_lufs"])
                writer.writeheader()
                writer.writerow({"track_number": "1", "filename": "other.wav", "full_integrated_loudness_lufs": "-9.5"})
                writer.writerow({"track_number": "2", "filename": "two.wav", "full_integrated_loudness_lufs": "not-a-number"})

            by_filename, by_number = load_loudness_catalog(tracklist, project_root=root)
            by_number_match = lookup_loudness(by_filename, by_number, filename="missing.wav", track_number=1)
            missing = lookup_loudness(by_filename, by_number, filename="two.wav", track_number=2)

        self.assertEqual(by_number_match.integrated_lufs, -9.5)
        self.assertIsNone(missing.integrated_lufs)

    def test_verified_render_cache_identity_includes_loudness_mode_and_measurement(self) -> None:
        track = PlaylistTrack(1, "One", "Artist", "one.wav", Path("/tmp/one.wav"), "House", "8A", 120.0, 0.0, None)
        other = PlaylistTrack(2, "Two", "Artist", "two.wav", Path("/tmp/two.wav"), "House", "9A", 120.0, 0.0, None)
        common = {
            "from_track": track,
            "to_track": other,
            "overlap_bars": 8,
            "front_padding_bars": 2,
            "back_padding_bars": 2,
            "from_nudge_beats": 0.0,
            "to_nudge_beats": 0.0,
            "from_beatgrid_ms": 0.0,
            "to_beatgrid_ms": 0.0,
            "from_pitch_shift": 0,
            "to_pitch_shift": 0,
            "from_bar": 0.0,
            "to_bar": 0.0,
            "from_cue": None,
            "to_cue": None,
            "from_start_seconds": 0.0,
            "to_start_seconds": 0.0,
            "beats_per_bar": 4,
            "sample_rate": 44100,
            "preset": "auto",
            "volume_mode": "overlap-crossfade",
            "eq_mode": "none",
            "filter_mode": "none",
            "automation": None,
            "effects": {"from": {}, "to": {}},
            "to_loudness_lufs": -10.0,
            "to_match_gain_db": -2.0,
            "quality": "preview",
        }
        matched = _transition_key(**common, loudness_match_mode="auditions_transitions", from_loudness_lufs=-9.0, from_match_gain_db=-3.0)
        off = _transition_key(**common, loudness_match_mode="off", from_loudness_lufs=-9.0, from_match_gain_db=0.0)
        updated_measurement = _transition_key(**common, loudness_match_mode="auditions_transitions", from_loudness_lufs=-8.0, from_match_gain_db=-4.0)

        self.assertNotEqual(matched, off)
        self.assertNotEqual(matched, updated_measurement)
