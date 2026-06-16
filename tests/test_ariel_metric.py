import unittest

import numpy as np

from ariel_ml.metrics import (
    ariel_gll_score,
    ariel_naive_reference,
    gaussian_log_likelihood,
)


class ArielMetricTests(unittest.TestCase):
    def test_gaussian_log_likelihood_matches_closed_form(self):
        y = np.array([0.0, 1.0])
        mu = np.array([0.0, 0.0])
        sigma = np.array([1.0, 2.0])
        expected = -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * ((y - mu) / sigma) ** 2
        np.testing.assert_allclose(gaussian_log_likelihood(y, mu, sigma), expected)

    def test_ideal_prediction_scores_one(self):
        rng = np.random.default_rng(0)
        y = 0.01 + 0.001 * rng.normal(size=(5, 20))
        naive_mean, naive_sigma = ariel_naive_reference(y)
        # Perfect mean with the ideal 10 ppm sigma -> score at the upper bound.
        score = ariel_gll_score(
            y, y.copy(), np.full_like(y, 1e-5),
            naive_mean=naive_mean, naive_sigma=naive_sigma,
        )
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_naive_prediction_scores_zero(self):
        rng = np.random.default_rng(1)
        y = 0.01 + 0.001 * rng.normal(size=(5, 20))
        naive_mean, naive_sigma = ariel_naive_reference(y)
        # Predicting the naive baseline itself -> score at the lower bound.
        score = ariel_gll_score(
            y,
            np.full_like(y, naive_mean),
            np.full_like(y, naive_sigma),
            naive_mean=naive_mean,
            naive_sigma=naive_sigma,
        )
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_score_is_clipped_and_monotonic_in_accuracy(self):
        rng = np.random.default_rng(2)
        y = 0.01 + 0.001 * rng.normal(size=(8, 30))
        naive_mean, naive_sigma = ariel_naive_reference(y)
        sigma = np.full_like(y, 5e-4)

        good = ariel_gll_score(y, y + 1e-4, sigma, naive_mean=naive_mean, naive_sigma=naive_sigma)
        bad = ariel_gll_score(y, y + 5e-3, sigma, naive_mean=naive_mean, naive_sigma=naive_sigma)

        self.assertGreater(good, bad)
        for value in (good, bad):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
