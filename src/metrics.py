from __future__ import annotations

import numpy as np


def gaussian_nll(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Mean per-element Gaussian negative log-likelihood (full constant included).

    Equals ``-mean(gaussian_log_likelihood(...))``; the ``0.5*log(2*pi)`` term makes the
    reported value the standard Gaussian NLL. (A constant offset does not change which
    scale/model minimises it, so calibration and model selection are unaffected.)
    """
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    return float(np.mean(0.5 * ((y_true - mu) / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2.0 * np.pi)))


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
    """Multiplicative sigma calibration that maximises the Gaussian log-likelihood.

    With ``per_target=False`` (default) a single scalar scale is fitted over all
    values (legacy behaviour). With ``per_target=True`` an independent scale
    ``s_j`` is fitted per target column: since the GLL is additive over elements
    and ``s_j`` only touches column ``j``, the GLL-optimal scale has the closed
    form ``s_j = sqrt(mean_i (residual_ij / sigma_ij)**2)`` (the RMS of the
    normalised residuals), which this computes directly.
    """

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
        x: np.ndarray | None = None,  # ignored; kept for a uniform calibrator interface
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
        # Degenerate columns (zero/non-finite residuals) keep an identity scale.
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
        return np.clip(scale, self.lower, self.upper)

    def transform(self, sigma: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
        return np.maximum(np.asarray(sigma, dtype=float) * self.scale_, self.sigma_floor)

    def fit_transform(self, y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        self.fit(y_true, mu, sigma)
        return self.transform(sigma)


class FeatureConditionedSigmaCalibrator:
    """PHC step 2 — heteroscedastic, feature-conditioned sigma calibration.

    ``sigma_calibrated_ij = s_j * m_i * sigma_ij`` with two GLL-optimal stages:

    1. **Per-wavelength** scale ``s_j = sqrt(mean_i (residual_ij/sigma_ij)**2)``
       (RMS over rows) — the closed-form GLL-optimal scale per column.
    2. **Per-observation** multiplier: the GLL-optimal single multiplier for row
       ``i`` (holding ``s_j`` fixed) is ``t_i = sqrt(mean_j (residual_ij/(s_j*sigma_ij))**2)``
       (RMS over columns). We regress ``log t_i`` on the input features via ridge
       regression, so ``m_i = exp(features_i . w)`` lets noisier observations get
       wider intervals at prediction time.

    Falls back to the per-wavelength scale alone (``m_i = 1``) when no features
    are supplied or there are too few rows to fit a stable regression.
    """

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
        self.weights_: np.ndarray | None = None  # ridge weights (intercept first)

    def fit(
        self,
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        x: np.ndarray | None = None,
    ) -> "FeatureConditionedSigmaCalibrator":
        sigma = np.maximum(np.asarray(sigma, dtype=float), self.sigma_floor)
        residual = np.asarray(y_true, dtype=float) - np.asarray(mu, dtype=float)

        # Stage 1 — per-wavelength scale (RMS over rows).
        nmse_col = np.mean((residual / sigma) ** 2, axis=0)
        scale = np.sqrt(nmse_col)
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
        self.scale_per_target_ = np.clip(scale, self.lower, self.upper)

        # Stage 2 — per-row optimal multiplier, then regress on features.
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
        penalty[0, 0] = 0.0  # do not regularise the intercept
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
