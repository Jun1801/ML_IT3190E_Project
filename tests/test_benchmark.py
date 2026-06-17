import unittest

import numpy as np

from benchmark import benchmark_models, family_of
from config import ModelConfig


def make_synthetic_regression(n_samples=48, n_features=8, n_targets=12):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n_samples, n_features))
    wavelength = np.linspace(0.0, 1.0, n_targets)
    basis = np.vstack(
        [
            np.ones(n_targets),
            np.sin(2 * np.pi * wavelength),
            np.cos(2 * np.pi * wavelength),
        ]
    )
    coeff = x[:, :3] @ np.array(
        [
            [0.020, 0.005, -0.002],
            [0.003, 0.010, 0.004],
            [-0.002, 0.003, 0.006],
        ]
    )
    y = coeff @ basis + 0.01
    y += 0.0001 * rng.normal(size=y.shape)
    return x, y


class BenchmarkTests(unittest.TestCase):
    def test_family_of_resolves_known_and_unknown_names(self):
        self.assertEqual(family_of("svr"), "kernel_svm")
        self.assertEqual(family_of("Random-Forest"), "trees")
        self.assertEqual(family_of("not_a_model"), "other")

    def test_benchmark_runs_each_model_on_same_folds(self):
        x, y = make_synthetic_regression()
        config = ModelConfig(n_components=3, calibrate_sigma=False)
        result = benchmark_models(
            x,
            y,
            model_names=["ridge", "svr", "knn", "extra_trees"],
            model_config=config,
            n_splits=3,
        )
        self.assertEqual(len(result.rows), 4)
        for row in result.rows:
            self.assertEqual(row.status, "ok", msg=f"{row.model_name}: {row.message}")
            self.assertIn("gaussian_nll", row.metrics)
            self.assertGreaterEqual(row.train_seconds, 0.0)

    def test_missing_optional_dependency_is_skipped_not_fatal(self):
        x, y = make_synthetic_regression()
        config = ModelConfig(n_components=3, calibrate_sigma=False)
        result = benchmark_models(
            x,
            y,
            model_names=["ridge", "lightgbm"],
            model_config=config,
            n_splits=3,
        )
        by_name = {row.model_name: row for row in result.rows}
        self.assertEqual(by_name["ridge"].status, "ok")
        # lightgbm is not installed in the test environment -> skipped, not error.
        self.assertEqual(by_name["lightgbm"].status, "skipped")

    def test_to_frame_and_best_select_lowest_nll(self):
        x, y = make_synthetic_regression()
        config = ModelConfig(n_components=3, calibrate_sigma=False)
        result = benchmark_models(
            x,
            y,
            model_names=["ridge", "extra_trees"],
            model_config=config,
            n_splits=3,
        )
        frame = result.to_frame()
        self.assertEqual(set(["family", "model", "status", "gaussian_nll"]) - set(frame.columns), set())
        best = result.best("gaussian_nll")
        self.assertIsNotNone(best)
        nll_values = [row.metrics["gaussian_nll"] for row in result.ok_rows]
        self.assertAlmostEqual(best.metrics["gaussian_nll"], min(nll_values))


if __name__ == "__main__":
    unittest.main()
