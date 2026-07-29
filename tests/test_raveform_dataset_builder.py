from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from djprojectexploration.raveform_dataset_builder import (
    discover_raveform_mixes,
    normalize_raveform_mix,
    write_normalized_tracklist,
)


class RaveformDatasetBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        self.ytdlp_root = Path(temporary_dir.name)
        self.raveform_root = self.ytdlp_root / "raveform_mixes"
        self.raveform_root.mkdir()

    def _create_mix(
        self,
        *,
        index: int = 1,
        rows: list[tuple[int, str, str, float, str]] | None = None,
    ) -> tuple[Path, Path]:
        stem = f"{index:03d}_mix-test"
        audio_dir = self.raveform_root / stem
        audio_dir.mkdir()
        csv_path = self.raveform_root / f"{stem}.csv"
        rows = rows or [
            (1, "First", "Artist A", 10.0, "first.mp3"),
            (2, "Layered", "Artist B", 10.001, "layered.mp3"),
            (3, "Next", "Artist C", 200.0, "next.mp3"),
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "track_number",
                    "track_name",
                    "artist",
                    "start_seconds",
                    "audio_file",
                ],
            )
            writer.writeheader()
            for track_number, title, artist, start_seconds, filename in rows:
                audio = audio_dir / filename
                if filename:
                    audio.write_bytes(b"audio")
                    audio_file = f"raveform_mixes/{stem}/{filename}"
                else:
                    audio_file = ""
                writer.writerow(
                    {
                        "track_number": track_number,
                        "track_name": title,
                        "artist": artist,
                        "start_seconds": start_seconds,
                        "audio_file": audio_file,
                    }
                )

        (self.raveform_root / f"{stem}.youtube.csv").write_text(
            "track_number,youtube_url\n1,https://example.test\n",
            encoding="utf-8",
        )
        return csv_path, audio_dir

    def test_discovers_source_csv_and_ignores_youtube_sidecar(self) -> None:
        csv_path, audio_dir = self._create_mix()

        mixes = discover_raveform_mixes(self.raveform_root, start=1, end=1)

        self.assertEqual(len(mixes), 1)
        self.assertEqual(mixes[0].csv_path, csv_path.resolve())
        self.assertEqual(mixes[0].audio_dir, audio_dir.resolve())
        self.assertEqual(mixes[0].dataset_name, "raveform-001-mix-test")

    def test_normalizes_schema_paths_and_layered_groups(self) -> None:
        self._create_mix()
        mix = discover_raveform_mixes(self.raveform_root, start=1, end=1)[0]

        normalized = normalize_raveform_mix(
            mix,
            raveform_root=self.raveform_root,
            group_tolerance_seconds=1.0,
        )

        self.assertEqual(len(normalized.rows), 3)
        self.assertEqual(normalized.transition_groups, 2)
        self.assertEqual(normalized.layered_entries, 1)
        self.assertEqual(
            [row["transition_group"] for row in normalized.rows],
            ["1", "1", "2"],
        )
        self.assertEqual(normalized.rows[0]["title"], "First")
        self.assertEqual(normalized.rows[0]["artists"], "Artist A")
        self.assertTrue(Path(normalized.rows[0]["filepath"]).is_absolute())
        self.assertTrue(Path(normalized.rows[0]["filepath"]).is_file())

    def test_rejects_missing_audio_before_building(self) -> None:
        self._create_mix(
            rows=[
                (1, "First", "Artist A", 10.0, "first.mp3"),
                (2, "Missing", "Artist B", 200.0, ""),
            ]
        )
        mix = discover_raveform_mixes(self.raveform_root, start=1, end=1)[0]

        with self.assertRaisesRegex(FileNotFoundError, "2: Missing"):
            normalize_raveform_mix(mix, raveform_root=self.raveform_root)

    def test_writes_tracklist_accepted_by_canonical_loader(self) -> None:
        self._create_mix()
        mix = discover_raveform_mixes(self.raveform_root, start=1, end=1)[0]
        normalized = normalize_raveform_mix(mix, raveform_root=self.raveform_root)
        output = self.ytdlp_root / "normalized.csv"

        write_normalized_tracklist(normalized, output)

        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["track_number"] for row in rows], ["1", "2", "3"])
        self.assertEqual(rows[1]["mix_start_seconds"], "10.001")
        self.assertEqual(rows[2]["transition_group"], "2")


if __name__ == "__main__":
    unittest.main()
