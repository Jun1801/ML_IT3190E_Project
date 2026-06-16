from __future__ import annotations

import numpy as np


def gaussian_nll(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    return float(np.mean(0.5 * ((y_true - mu) / sigma) ** 2 + np.log(sigma)))


def gaussian_log_likelihood(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    sigma_floor: float = 1e-15,
) -> np.ndarray:
    """Per-element Gaussian log-likelihood (matches ``scipy.stats.norm.logpdf``).

    ``GLL = -1/2 (log(2*pi) + log(sigma**2) + ((y - mu) / sigma)**2)``
    """
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), sigma_floor)
    return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * ((y_true - mu) / sigma) ** 2


def ariel_naive_reference(y_train: np.ndarray) -> tuple[float, float]:
    """Naive baseline statistics used by the official metric.

    Returns the global mean and standard deviation of the training spectra,
    which define the reference (lower-bound) model that predicts a constant
    mean with constant uncertainty everywhere.
    """
    y_train = np.asarray(y_train, dtype=float)
    return float(np.mean(y_train)), float(np.std(y_train))


def ariel_gll_score(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    naive_mean: float,
    naive_sigma: float,
    sigma_true: float = 1e-5,
    sigma_floor: float = 1e-15,
) -> float:
    """Official Ariel Data Challenge 2025 normalized GLL score, in ``[0, 1]``.

    ``score = (GLL_pred - GLL_ref) / (GLL_ideal - GLL_ref)`` clipped to ``[0, 1]``:

    - ``GLL_pred``  — log-likelihood of the ground truth under ``(mu, sigma)``.
    - ``GLL_ideal`` — perfect prediction ``mu == y_true`` with ``sigma_true``
      (the competition uses 10 ppm, i.e. ``1e-5``).
    - ``GLL_ref``   — naive model predicting ``naive_mean`` with ``naive_sigma``
      everywhere (derived from the training spectra).

    Higher is better: ``0`` matches the naive baseline, ``1`` the ideal model.
    """
    y_true = np.asarray(y_true, dtype=float)
    gll_pred = float(np.sum(gaussian_log_likelihood(y_true, mu, sigma, sigma_floor=sigma_floor)))
    gll_ideal = float(
        np.sum(
            gaussian_log_likelihood(
                y_true, y_true, np.full_like(y_true, sigma_true), sigma_floor=sigma_floor
            )
        )
    )
    gll_ref = float(
        np.sum(
            gaussian_log_likelihood(
                y_true,
                np.full_like(y_true, naive_mean),
                np.full_like(y_true, naive_sigma),
                sigma_floor=sigma_floor,
            )
        )
    )
    denominator = gll_ideal - gll_ref
    if not np.isfinite(denominator) or denominator == 0.0:
        return 0.0
    return float(np.clip((gll_pred - gll_ref) / denominator, 0.0, 1.0))


def rmse_per_target(y_true: np.ndarray, mu: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    rmse = np.sqrt(np.mean((y_true - mu) ** 2, axis=0))
    return np.maximum(rmse, floor)


class SigmaCalibrator:
    def __init__(
        self,
        *,
        lower: float = 1e-4,
        upper: float = 100.0,
        n_grid: int = 200,
        sigma_floor: float = 1e-8,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.n_grid = n_grid
        self.sigma_floor = sigma_floor
        self.scale_: float = 1.0

    def fit(self, y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> "SigmaCalibrator":
        sigma = np.maximum(np.asarray(sigma, dtype=float), self.sigma_floor)
        residual = np.asarray(y_true, dtype=float) - np.asarray(mu, dtype=float)
        normalized_mse = np.mean((residual / sigma) ** 2)
        if np.isfinite(normalized_mse) and normalized_mse > 0.0:
            self.scale_ = float(np.clip(np.sqrt(normalized_mse), self.lower, self.upper))
        else:
            scales = np.geomspace(self.lower, self.upper, self.n_grid)
            losses = [gaussian_nll(y_true, mu, sigma * scale) for scale in scales]
            self.scale_ = float(scales[int(np.argmin(losses))])
        return self

    def transform(self, sigma: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(sigma, dtype=float) * self.scale_, self.sigma_floor)

    def fit_transform(self, y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        self.fit(y_true, mu, sigma)
        return self.transform(sigma)
