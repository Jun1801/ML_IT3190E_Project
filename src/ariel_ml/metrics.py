from __future__ import annotations

import numpy as np


def gaussian_nll(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    return float(np.mean(0.5 * ((y_true - mu) / sigma) ** 2 + np.log(sigma)))


def rmse_per_target(y_true: np.ndarray, mu: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    rmse = np.sqrt(np.mean((y_true - mu) ** 2, axis=0))
    return np.maximum(rmse, floor)


class SigmaCalibrator:
    def __init__(
        self,
        *,
        lower: float = 0.05,
        upper: float = 20.0,
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
        scales = np.geomspace(self.lower, self.upper, self.n_grid)
        losses = [gaussian_nll(y_true, mu, sigma * scale) for scale in scales]
        self.scale_ = float(scales[int(np.argmin(losses))])
        return self

    def transform(self, sigma: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(sigma, dtype=float) * self.scale_, self.sigma_floor)

    def fit_transform(self, y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        self.fit(y_true, mu, sigma)
        return self.transform(sigma)
