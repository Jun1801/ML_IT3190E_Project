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
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), sigma_floor)
    return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * ((y_true - mu) / sigma) ** 2


def ariel_naive_reference(y_train: np.ndarray) -> tuple[float, float]:
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
        per_target: bool = False,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.n_grid = n_grid
        self.sigma_floor = sigma_floor
        self.per_target = per_target
        self.scale_: float | np.ndarray = 1.0

    def fit(
        self,
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        x: np.ndarray | None = None,
    ) -> "SigmaCalibrator":
        sigma = np.maximum(np.asarray(sigma, dtype=float), self.sigma_floor)
        residual = np.asarray(y_true, dtype=float) - np.asarray(mu, dtype=float)
        if self.per_target:
            self.scale_ = self._fit_per_target(residual, sigma)
        else:
            self.scale_ = self._fit_scalar(y_true, mu, sigma, residual)
        return self

    def _fit_scalar(
        self, y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, residual: np.ndarray
    ) -> float:
        normalized_mse = np.mean((residual / sigma) ** 2)
        if np.isfinite(normalized_mse) and normalized_mse > 0.0:
            return float(np.clip(np.sqrt(normalized_mse), self.lower, self.upper))
        scales = np.geomspace(self.lower, self.upper, self.n_grid)
        losses = [gaussian_nll(y_true, mu, sigma * scale) for scale in scales]
        return float(scales[int(np.argmin(losses))])

    def _fit_per_target(self, residual: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        normalized_mse = np.mean((residual / sigma) ** 2, axis=0)
        scale = np.sqrt(normalized_mse)
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
        return np.clip(scale, self.lower, self.upper)

    def transform(self, sigma: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
        return np.maximum(np.asarray(sigma, dtype=float) * self.scale_, self.sigma_floor)

    def fit_transform(self, y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        self.fit(y_true, mu, sigma)
        return self.transform(sigma)


class FeatureConditionedSigmaCalibrator:
    def __init__(
        self,
        *,
        sigma_floor: float = 1e-8,
        lower: float = 1e-4,
        upper: float = 100.0,
        ridge_lambda: float = 1.0,
        multiplier_clip: tuple[float, float] = (0.1, 10.0),
        min_rows: int = 8,
    ) -> None:
        self.sigma_floor = sigma_floor
        self.lower = lower
        self.upper = upper
        self.ridge_lambda = ridge_lambda
        self.multiplier_clip = multiplier_clip
        self.min_rows = min_rows
        self.scale_per_target_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    def fit(
        self,
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        x: np.ndarray | None = None,
    ) -> "FeatureConditionedSigmaCalibrator":
        sigma = np.maximum(np.asarray(sigma, dtype=float), self.sigma_floor)
        residual = np.asarray(y_true, dtype=float) - np.asarray(mu, dtype=float)

        nmse_col = np.mean((residual / sigma) ** 2, axis=0)
        scale = np.sqrt(nmse_col)
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
        self.scale_per_target_ = np.clip(scale, self.lower, self.upper)

        self.weights_ = None
        if x is not None:
            x = np.asarray(x, dtype=float)
            sigma_after = sigma * self.scale_per_target_[np.newaxis, :]
            nmse_row = np.mean((residual / sigma_after) ** 2, axis=1)
            t = np.sqrt(np.where(np.isfinite(nmse_row) & (nmse_row > 0.0), nmse_row, 1.0))
            if x.ndim == 2 and x.shape[0] == t.shape[0] and x.shape[0] >= self.min_rows:
                self.weights_ = self._fit_ridge_log(x, t)
        return self

    def _fit_ridge_log(self, x: np.ndarray, t: np.ndarray) -> np.ndarray | None:
        n_rows, n_features = x.shape
        design = np.hstack([np.ones((n_rows, 1)), x])
        target = np.log(np.clip(t, 1e-6, None))
        gram = design.T @ design
        penalty = self.ridge_lambda * np.eye(n_features + 1)
        penalty[0, 0] = 0.0
        try:
            return np.linalg.solve(gram + penalty, design.T @ target)
        except np.linalg.LinAlgError:
            return None

    def _multiplier(self, x: np.ndarray | None) -> np.ndarray | float:
        if self.weights_ is None or x is None:
            return 1.0
        x = np.asarray(x, dtype=float)
        design = np.hstack([np.ones((x.shape[0], 1)), x])
        multiplier = np.exp(design @ self.weights_)
        return np.clip(multiplier, self.multiplier_clip[0], self.multiplier_clip[1])

    def transform(self, sigma: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
        sigma = np.asarray(sigma, dtype=float)
        scale = self.scale_per_target_ if self.scale_per_target_ is not None else 1.0
        out = sigma * scale
        multiplier = self._multiplier(x)
        if not np.isscalar(multiplier):
            out = out * np.asarray(multiplier).reshape(-1, 1)
        else:
            out = out * multiplier
        return np.maximum(out, self.sigma_floor)
