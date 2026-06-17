from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import ModelConfig
from metrics import FeatureConditionedSigmaCalibrator, SigmaCalibrator, rmse_per_target


@dataclass(frozen=True)
class ModelPrediction:
    mu: np.ndarray
    sigma: np.ndarray


class TargetPCARegressor:
    def __init__(
        self,
        estimator_factory: Callable[[], RegressorMixin],
        config: ModelConfig | None = None,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.config = config or ModelConfig()
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.models: list[RegressorMixin] = []
        self.residual_rmse_: np.ndarray | None = None
        if self.config.sigma_feature_conditioned:
            self.sigma_calibrator = FeatureConditionedSigmaCalibrator(sigma_floor=self.config.sigma_floor)
        else:
            self.sigma_calibrator = SigmaCalibrator(
                sigma_floor=self.config.sigma_floor,
                per_target=self.config.sigma_per_target,
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Local lambdas used to build sklearn estimators are not pickleable.
        # Fitted models are stored in self.models, so the factory is not needed
        # for saved inference artifacts.
        state["estimator_factory"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TargetPCARegressor":
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        x_scaled = self._fit_transform_x(x_arr)

        n_components = min(self.config.n_components, y_arr.shape[0], y_arr.shape[1])
        self.pca = PCA(n_components=n_components, random_state=self.config.random_state)
        z = self.pca.fit_transform(y_arr)
        self.models = [self.estimator_factory().fit(x_scaled, z[:, idx]) for idx in range(z.shape[1])]

        if x_val is not None and y_val is not None:
            residual_x = np.asarray(x_val, dtype=float)
            residual_y = np.asarray(y_val, dtype=float)
            residual_pred = self._predict_uncalibrated(residual_x)
            self.residual_rmse_ = rmse_per_target(
                residual_y,
                residual_pred.mu,
                floor=self.config.residual_floor,
            )
            if self.config.calibrate_sigma:
                self.sigma_calibrator.fit(
                    residual_y, residual_pred.mu, residual_pred.sigma, self._transform_x(residual_x)
                )
        return self

    def predict(self, x: np.ndarray) -> ModelPrediction:
        prediction = self._predict_uncalibrated(x)
        x_scaled = self._transform_x(np.asarray(x, dtype=float))
        return ModelPrediction(
            mu=prediction.mu,
            sigma=self.sigma_calibrator.transform(prediction.sigma, x_scaled),
        )

    def _predict_uncalibrated(self, x: np.ndarray) -> ModelPrediction:
        self._require_fitted()
        x_scaled = self._transform_x(np.asarray(x, dtype=float))
        z_mu, z_std = self._predict_pca_space(x_scaled)
        assert self.pca is not None
        mu = self.pca.inverse_transform(z_mu)
        propagated_sigma = self._propagate_pca_sigma(z_std)
        residual_rmse = self._residual_rmse(mu.shape[1])
        sigma = np.sqrt(propagated_sigma**2 + residual_rmse[np.newaxis, :] ** 2)
        sigma = np.maximum(sigma, self.config.sigma_floor)
        return ModelPrediction(mu=mu, sigma=sigma)

    def _predict_pca_space(self, x_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mus: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        for model in self.models:
            mu, std = self._predict_one_model(model, x_scaled)
            mus.append(mu)
            stds.append(std)
        return np.column_stack(mus), np.column_stack(stds)

    def _predict_one_model(self, model: RegressorMixin, x_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            mu, std = model.predict(x_scaled, return_std=True)
            return np.asarray(mu, dtype=float), np.asarray(std, dtype=float)
        except TypeError:
            pass

        mu = np.asarray(model.predict(x_scaled), dtype=float)
        std = self._ensemble_std(model, x_scaled, mu.shape[0])
        return mu, std

    def _ensemble_std(self, model: RegressorMixin, x_scaled: np.ndarray, n_rows: int) -> np.ndarray:
        estimators = getattr(model, "estimators_", None)
        if estimators is None:
            return np.zeros(n_rows, dtype=float)
        estimators = np.asarray(estimators, dtype=object).ravel()
        tree_predictions = np.column_stack([est.predict(x_scaled) for est in estimators])
        return np.std(tree_predictions, axis=1)

    def _propagate_pca_sigma(self, z_std: np.ndarray) -> np.ndarray:
        assert self.pca is not None
        z_var = np.asarray(z_std, dtype=float) ** 2
        y_var = z_var @ (self.pca.components_**2)
        return np.sqrt(np.maximum(y_var, 0.0))

    def _fit_transform_x(self, x: np.ndarray) -> np.ndarray:
        if not self.config.standardize_features:
            return x
        return self.scaler.fit_transform(x)

    def _transform_x(self, x: np.ndarray) -> np.ndarray:
        if not self.config.standardize_features:
            return x
        return self.scaler.transform(x)

    def _residual_rmse(self, n_targets: int) -> np.ndarray:
        if self.residual_rmse_ is None:
            return np.full(n_targets, self.config.residual_floor, dtype=float)
        return self.residual_rmse_

    def _require_fitted(self) -> None:
        if self.pca is None or not self.models:
            raise RuntimeError("Model must be fitted before prediction.")


class ResidualCorrectedRegressor:
    def __init__(
        self,
        base_model: TargetPCARegressor,
        residual_model: TargetPCARegressor,
        *,
        eta: float = 0.3,
        sigma_floor: float = 1e-8,
    ) -> None:
        self.base_model = base_model
        self.residual_model = residual_model
        self.eta = eta
        self.sigma_floor = sigma_floor

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "ResidualCorrectedRegressor":
        self.base_model.fit(x, y, x_val=x_val, y_val=y_val)
        residual = np.asarray(y, dtype=float) - self.base_model.predict(x).mu
        residual_val = None
        if x_val is not None and y_val is not None:
            residual_val = np.asarray(y_val, dtype=float) - self.base_model.predict(x_val).mu
        self.residual_model.fit(x, residual, x_val=x_val, y_val=residual_val)
        return self

    def predict(self, x: np.ndarray) -> ModelPrediction:
        base = self.base_model.predict(x)
        residual = self.residual_model.predict(x)
        mu = base.mu + self.eta * residual.mu
        sigma = np.sqrt(base.sigma**2 + (self.eta * residual.sigma) ** 2)
        return ModelPrediction(mu=mu, sigma=np.maximum(sigma, self.sigma_floor))


class WeightedEnsembleRegressor:
    def __init__(self, models: list[TargetPCARegressor], weights: list[float] | None = None) -> None:
        if not models:
            raise ValueError("WeightedEnsembleRegressor requires at least one model.")
        self.models = models
        raw_weights = np.ones(len(models), dtype=float) if weights is None else np.asarray(weights, dtype=float)
        if raw_weights.shape[0] != len(models):
            raise ValueError("weights length must match models length.")
        self.weights = raw_weights / np.sum(raw_weights)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "WeightedEnsembleRegressor":
        for model in self.models:
            model.fit(x, y, x_val=x_val, y_val=y_val)
        return self

    def predict(self, x: np.ndarray) -> ModelPrediction:
        predictions = [model.predict(x) for model in self.models]
        mu = sum(weight * pred.mu for weight, pred in zip(self.weights, predictions))
        second_moment = sum(weight * (pred.sigma**2 + pred.mu**2) for weight, pred in zip(self.weights, predictions))
        sigma = np.sqrt(np.maximum(second_moment - mu**2, 1e-12))
        return ModelPrediction(mu=mu, sigma=sigma)
