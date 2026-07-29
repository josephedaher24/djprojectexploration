from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from djprojectexploration.pacmap_settings import TransitionScoringSettings
from djprojectexploration.transition_scoring import (
    DEFAULT_RHYTHM_TEMPO_WEIGHT,
    baseline_transition_score,
    blend_component_distances,
    final_transition_score,
    normalize_top_level_weights,
    rhythm_component_score,
)


class TransitionScoringTests(unittest.TestCase):
    def test_default_rhythm_is_seventy_percent_tempo(self) -> None:
        tempo = np.asarray([0.2, 0.8, 1.0])
        groove = np.asarray([1.0, 0.4, 0.0])

        rhythm = rhythm_component_score(tempo, groove)

        self.assertEqual(DEFAULT_RHYTHM_TEMPO_WEIGHT, 0.70)
        self.assertTrue(np.allclose(rhythm, 0.70 * tempo + 0.30 * groove))

    def test_legacy_preset_converts_tempo_and_groove_weights(self) -> None:
        payload = {
            "weights": {
                "style_weight": 2.0,
                "tempo_weight": 3.0,
                "groove_weight": 1.0,
                "harmony_weight": 4.0,
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            settings = TransitionScoringSettings.from_json(path)

        self.assertAlmostEqual(settings.style_weight, 0.2)
        self.assertAlmostEqual(settings.rhythm_weight, 0.4)
        self.assertAlmostEqual(settings.harmony_weight, 0.4)
        self.assertAlmostEqual(settings.rhythm_tempo_weight, 0.75)

    def test_legacy_zero_rhythm_uses_default_tempo_share(self) -> None:
        payload = {
            "weights": {
                "style_weight": 0.8,
                "tempo_weight": 0.0,
                "groove_weight": 0.0,
                "harmony_weight": 0.2,
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-zero.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            settings = TransitionScoringSettings.from_json(path)

        self.assertAlmostEqual(settings.rhythm_tempo_weight, 0.70)

    def test_top_level_weights_normalize_over_three_components(self) -> None:
        weights = normalize_top_level_weights(4.0, 3.0, 3.0)

        self.assertTrue(np.allclose(weights, (0.4, 0.3, 0.3)))
        self.assertAlmostEqual(sum(weights), 1.0)

        baseline = baseline_transition_score(
            1.0,
            0.5,
            0.0,
            style_weight=4.0,
            rhythm_weight=3.0,
            harmony_weight=3.0,
        )
        self.assertAlmostEqual(float(baseline), 0.55)

    def test_final_score_stays_in_unit_interval(self) -> None:
        baseline = np.linspace(0.0, 1.0, 101)
        energy_fit = np.linspace(1.0, 0.0, 101)

        final = final_transition_score(baseline, energy_fit)

        self.assertGreaterEqual(float(final.min()), 0.0)
        self.assertLessEqual(float(final.max()), 1.0)

    def test_layout_distance_is_three_component_linear_blend(self) -> None:
        style = np.asarray([[0.0, 0.2], [0.2, 0.0]])
        rhythm = np.asarray([[0.0, 0.8], [0.8, 0.0]])
        harmony = np.asarray([[0.0, 0.5], [0.5, 0.0]])

        distance = blend_component_distances(
            style,
            rhythm,
            harmony,
            style_weight=0.5,
            rhythm_weight=0.3,
            harmony_weight=0.2,
        )

        self.assertAlmostEqual(float(distance[0, 1]), 0.44)
        self.assertTrue(np.allclose(np.diag(distance), 0.0))

    def test_main_weight_controls_only_expose_three_components(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        template = (
            project_root
            / "src"
            / "djprojectexploration"
            / "templates"
            / "energy_sequence_builder_mixability_weights.html"
        ).read_text(encoding="utf-8")

        self.assertIn('id="weight-style"', template)
        self.assertIn('id="weight-rhythm"', template)
        self.assertIn('id="weight-harmony"', template)
        self.assertNotIn('id="weight-tempo"', template)
        self.assertNotIn('id="weight-groove"', template)


if __name__ == "__main__":
    unittest.main()
