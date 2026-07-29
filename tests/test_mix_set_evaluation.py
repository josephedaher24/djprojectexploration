from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from djprojectexploration.mix_set_evaluation import (
    DEFAULT_SIMPLEX_STEP,
    adjacent_component_means,
    normalize_distance_matrix_for_evaluation,
    rank_metrics_for_similarity,
    rhythm_component_score,
    simplex_permutation_metrics,
    simplex_weight_grid,
    slug_from_manifest_path,
)


class MixSetEvaluationTests(unittest.TestCase):
    def test_slug_from_rendered_set_manifest_filename(self) -> None:
        self.assertEqual(
            slug_from_manifest_path(Path("data/exports/fourtet2024ade_rendered_set_manifest.json")),
            "fourtet2024ade",
        )

    def test_slug_from_canonical_build_manifest_filename(self) -> None:
        self.assertEqual(
            slug_from_manifest_path(Path("data/exports/raveform-001-example_build_manifest.json")),
            "raveform-001-example",
        )

    def test_default_simplex_grid_has_expected_count_and_sums(self) -> None:
        grid = simplex_weight_grid(DEFAULT_SIMPLEX_STEP)

        self.assertEqual(len(grid), 66)
        sums = grid[["style_weight", "rhythm_weight", "harmony_weight"]].sum(axis=1)
        self.assertTrue(np.allclose(sums, 1.0))

    def test_top_nonself_normalization_anchors_best_nonself_score_at_one(self) -> None:
        similarity = np.asarray(
            [
                [1.0, 0.95, 0.80],
                [0.95, 1.0, 0.50],
                [0.80, 0.50, 1.0],
            ],
            dtype=np.float32,
        )

        distance = normalize_distance_matrix_for_evaluation(
            1.0 - similarity,
            top_nonself_score_anchor=True,
        )
        normalized_similarity = 1.0 - distance

        self.assertAlmostEqual(float(normalized_similarity[0, 1]), 1.0)
        self.assertAlmostEqual(float(normalized_similarity[1, 0]), 1.0)
        self.assertAlmostEqual(float(np.diag(distance).max()), 0.0)

    def test_constant_distance_component_is_valid(self) -> None:
        distance = np.zeros((4, 4), dtype=np.float32)

        normalized = normalize_distance_matrix_for_evaluation(
            distance,
            top_nonself_score_anchor=True,
        )

        self.assertTrue(np.allclose(normalized, 0.0))

    def test_rank_metrics_reward_actual_next_track_position(self) -> None:
        similarity = np.asarray(
            [
                [1.0, 0.9, 0.1],
                [0.2, 1.0, 0.8],
                [0.3, 0.4, 1.0],
            ],
            dtype=np.float64,
        )
        order = np.asarray([0, 1, 2], dtype=np.int64)

        metrics = rank_metrics_for_similarity(similarity, order)

        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["median_rank"], 1.0)

    def test_rhythm_score_combines_tempo_and_groove_after_normalization(self) -> None:
        tempo = np.asarray([[1.0, 0.8], [0.8, 1.0]])
        groove = np.asarray([[1.0, 0.2], [0.2, 1.0]])

        combined = rhythm_component_score(tempo, groove, tempo_weight=0.75)

        self.assertTrue(np.allclose(combined, np.asarray([[1.0, 0.65], [0.65, 1.0]])))

    def test_simplex_permutation_metrics_reports_positive_lift(self) -> None:
        component_scores = [
            np.asarray([[1.0, 0.9, 0.1], [0.2, 1.0, 0.8], [0.3, 0.4, 1.0]]),
            np.asarray([[1.0, 0.6, 0.1], [0.2, 1.0, 0.7], [0.3, 0.4, 1.0]]),
        ]
        order = np.asarray([0, 1, 2], dtype=np.int64)
        real_means = adjacent_component_means(component_scores, order)
        random_means = np.asarray([[0.2, 0.2], [0.4, 0.3], [0.5, 0.4]], dtype=np.float64)

        metrics = simplex_permutation_metrics(
            np.asarray([0.5, 0.5], dtype=np.float64),
            real_means,
            random_means,
        )

        self.assertGreater(metrics["lift"], 0.0)
        self.assertGreater(metrics["z_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
