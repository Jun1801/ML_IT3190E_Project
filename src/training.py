from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, train_test_split

from config import ModelConfig
from metrics import ariel_gll_score, ariel_naive_reference, gaussian_nll, rmse_per_target
from models import ModelPrediction, ResidualCorrectedRegressor, TargetPCARegressor, WeightedEnsembleRegressor
from estimators import ModelFactory

HIGHER_IS_BETTER_METRICS: frozenset[str] = frozenset({"ariel_gll_score"})


@dataclass(frozen=True)
class EvaluationResult:
    rmse_mean: float
    rmse_per_target: np.ndarray
    mae_mean: float
    gaussian_nll: float
    ariel_gll_score: float
    sigma_mean: float
    sigma_min: float
    sigma_max: float
    sigma_to_abs_error_ratio: float
    coverage_1sigma: float
    coverage_2sigma: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}rmse_mean": self.rmse_mean,
            f"{prefix}mae_mean": self.mae_mean,
            f"{prefix}gaussian_nll": self.gaussian_nll,
            f"{prefix}ariel_gll_score": self.ariel_gll_score,
            f"{prefix}sigma_mean": self.sigma_mean,
            f"{prefix}sigma_min": self.sigma_min,
            f"{prefix}sigma_max": self.sigma_max,
            f"{prefix}sigma_to_abs_error_ratio": self.sigma_to_abs_error_ratio,
            f"{prefix}coverage_1sigma": self.coverage_1sigma,
            f"{prefix}coverage_2sigma": self.coverage_2sigma,
        }


@dataclass(frozen=True)
class TrainResult:
    model: TargetPCARegressor | ResidualCorrectedRegressor
    prediction: ModelPrediction
    evaluation: EvaluationResult
    train_index: np.ndarray
    validation_index: np.ndarray


@dataclass(frozen=True)
class CrossValidationResult:
    fold_results: list[TrainResult]

    @property
    def mean_metrics(self) -> dict[str, float]:
        if not self.fold_results:
            return {}
        rows = [fold.evaluation.as_dict() for fold in self.fold_results]
        keys = rows[0].keys()
        return {key: float(np.mean([row[key] for row in rows])) for key in keys}


@dataclass(frozen=True)
class SearchCandidateResult:
    model_name: str
    model_config: ModelConfig
    mean_metrics: dict[str, float]


@dataclass(frozen=True)
class HyperparameterSearchResult:
    candidates: list[SearchCandidateResult]
    best_candidate: SearchCandidateResult


def feature_dicts_to_frame(feature_rows: list[dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(feature_rows)
    return frame.reindex(sorted(frame.columns), axis=1).fillna(0.0)


def targets_to_matrix(targets: pd.DataFrame, *, id_column: str = "planet_id") -> tuple[np.ndarray, list[str]]:
    target_columns = [col for col in targets.columns if col != id_column]
    return targets[target_columns].to_numpy(dtype=float), target_columns


def evaluate_prediction(
    y_true: np.ndarray,
    prediction: ModelPrediction,
    *,
    naive_reference: tuple[float, float] | None = None,
    sigma_true: float = 1e-5,
) -> EvaluationResult:
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(prediction.mu, dtype=float)
    sigma = np.maximum(np.asarray(prediction.sigma, dtype=float), 1e-12)
    residual = y_true - mu
    rmse = rmse_per_target(y_true, mu)
    abs_error = np.abs(residual)
    within_1sigma = abs_error <= sigma
    within_2sigma = abs_error <= 2.0 * sigma
    naive_mean, naive_sigma = naive_reference if naive_reference is not None else ariel_naive_reference(y_true)
    return EvaluationResult(
        rmse_mean=float(np.mean(rmse)),
        rmse_per_target=rmse,
        mae_mean=float(np.mean(abs_error)),
        gaussian_nll=gaussian_nll(y_true, mu, sigma),
        ariel_gll_score=ariel_gll_score(
            y_true, mu, sigma, naive_mean=naive_mean, naive_sigma=naive_sigma, sigma_true=sigma_true
        ),
        sigma_mean=float(np.mean(sigma)),
        sigma_min=float(np.min(sigma)),
        sigma_max=float(np.max(sigma)),
        sigma_to_abs_error_ratio=float(np.mean(sigma) / max(float(np.mean(abs_error)), 1e-12)),
        coverage_1sigma=float(np.mean(within_1sigma)),
        coverage_2sigma=float(np.mean(within_2sigma)),
    )


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_name: str = "bayesian_ridge",
    model_config: ModelConfig | None = None,
    validation_fraction: float = 0.2,
    groups: np.ndarray | None = None,
    random_state: int = 42,
    sigma_cal_fraction: float = 0.0,
) -> TrainResult:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    train_idx, val_idx = make_train_validation_split(
        n_samples=x_arr.shape[0],
        validation_fraction=validation_fraction,
        groups=groups,
        random_state=random_state,
    )
    return train_model_on_indices(
        x_arr,
        y_arr,
        train_idx,
        val_idx,
        model_name=model_name,
        model_config=model_config,
        sigma_cal_fraction=sigma_cal_fraction,
        random_state=random_state,
    )


def refit_full_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_name: str = "bayesian_ridge",
    model_config: ModelConfig | None = None,
):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    model = ModelFactory.create(model_name, model_config)
    model.fit(x_arr, y_arr)
    return model


def train_model_on_indices(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    model_name: str = "bayesian_ridge",
    model_config: ModelConfig | None = None,
    sigma_cal_fraction: float = 0.0,
    random_state: int = 42,
) -> TrainResult:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    model = ModelFactory.create(model_name, model_config)

    if sigma_cal_fraction > 0.0 and len(train_idx) > 4:
        n_cal = max(1, int(len(train_idx) * sigma_cal_fraction))
        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(len(train_idx))
        cal_idx = train_idx[shuffled[:n_cal]]
        model_train_idx = train_idx[shuffled[n_cal:]]
        model.fit(
            x_arr[model_train_idx], y_arr[model_train_idx],
            x_val=x_arr[cal_idx], y_val=y_arr[cal_idx],
        )
    else:
        model.fit(
            x_arr[train_idx], y_arr[train_idx],
            x_val=x_arr[val_idx], y_val=y_arr[val_idx],
        )

    prediction = model.predict(x_arr[val_idx])
    naive_ref = ariel_naive_reference(y_arr[train_idx])
    evaluation = evaluate_prediction(y_arr[val_idx], prediction, naive_reference=naive_ref)
    return TrainResult(
        model=model,
        prediction=prediction,
        evaluation=evaluation,
        train_index=np.asarray(train_idx, dtype=int),
        validation_index=np.asarray(val_idx, dtype=int),
    )


def hyperparameter_search(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_names: Iterable[str] = ("bayesian_ridge",),
    n_components_grid: Iterable[int] = (20, 30, 40),
    model_params_grid: list[dict] | None = None,
    base_config: ModelConfig | None = None,
    n_splits: int = 5,
    groups: np.ndarray | None = None,
    random_state: int = 42,
    selection_metric: str = "gaussian_nll",
    maximize: bool | None = None,
    sigma_cal_fraction: float = 0.0,
) -> HyperparameterSearchResult:
    if maximize is None:
        maximize = selection_metric in HIGHER_IS_BETTER_METRICS
    base = base_config or ModelConfig(random_state=random_state)
    params_list: list[dict] = model_params_grid if model_params_grid else [{}]
    candidates: list[SearchCandidateResult] = []
    for model_name in model_names:
        for n_components in n_components_grid:
            for params in params_list:
                config = replace(
                    base,
                    n_components=int(n_components),
                    random_state=random_state,
                    model_params=params,
                )
                cv_result = cross_validate_model(
                    x,
                    y,
                    model_name=model_name,
                    model_config=config,
                    n_splits=n_splits,
                    groups=groups,
                    random_state=random_state,
                    sigma_cal_fraction=sigma_cal_fraction,
                )
                candidates.append(
                    SearchCandidateResult(
                        model_name=model_name,
                        model_config=config,
                        mean_metrics=cv_result.mean_metrics,
                    )
                )
    if not candidates:
        raise ValueError("hyperparameter_search requires at least one candidate.")
    chooser = max if maximize else min
    best = chooser(candidates, key=lambda item: item.mean_metrics[selection_metric])
    return HyperparameterSearchResult(candidates=candidates, best_candidate=best)


def search_n_components(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_name: str = "bayesian_ridge",
    n_components_grid: Iterable[int] = (16, 24, 32),
    base_config: ModelConfig | None = None,
    n_splits: int = 5,
    groups: np.ndarray | None = None,
    random_state: int = 42,
    selection_metric: str = "ariel_gll_score",
    maximize: bool | None = None,
    sigma_cal_fraction: float = 0.0,
) -> HyperparameterSearchResult:
    grid = sorted({int(k) for k in n_components_grid})
    if not grid:
        raise ValueError("n_components_grid must be non-empty.")
    base = base_config or ModelConfig(random_state=random_state)
    if maximize is None:
        maximize = selection_metric in HIGHER_IS_BETTER_METRICS
    k_max = grid[-1]
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    splitter = make_cv_splitter(n_splits=n_splits, groups=groups, random_state=random_state)
    split_groups = None if groups is None else np.asarray(groups)

    per_k_rows: dict[int, list[dict[str, float]]] = {k: [] for k in grid}
    for train_idx, val_idx in splitter.split(x_arr, y_arr, groups=split_groups):
        if sigma_cal_fraction > 0.0 and len(train_idx) > 4:
            n_cal = max(1, int(len(train_idx) * sigma_cal_fraction))
            shuffled = np.random.default_rng(random_state).permutation(len(train_idx))
            cal_idx = train_idx[shuffled[:n_cal]]
            model_train_idx = train_idx[shuffled[n_cal:]]
        else:
            cal_idx = val_idx
            model_train_idx = train_idx

        model = ModelFactory.create(model_name, replace(base, n_components=k_max, random_state=random_state))
        if not hasattr(model, "predict_pca_space"):
            raise TypeError(f"search_n_components requires a TargetPCARegressor model, got {model_name}.")
        model.fit(x_arr[model_train_idx], y_arr[model_train_idx])

        zc_mu, zc_std, xc = model.predict_pca_space(x_arr[cal_idx])
        zv_mu, zv_std, xv = model.predict_pca_space(x_arr[val_idx])
        components, mean_ = model.pca.components_, model.pca.mean_
        naive_ref = ariel_naive_reference(y_arr[train_idx])

        for k in grid:
            comp_k = components[:k]
            mu_cal = zc_mu[:, :k] @ comp_k + mean_
            prop_cal = np.sqrt(np.maximum((zc_std[:, :k] ** 2) @ (comp_k ** 2), 0.0))
            residual_rmse = rmse_per_target(y_arr[cal_idx], mu_cal, floor=base.residual_floor)

            sigma_cal_fit = np.maximum(np.sqrt(prop_cal ** 2 + base.residual_floor ** 2), base.sigma_floor)
            calibrator = model._build_calibrator()
            if base.calibrate_sigma:
                calibrator.fit(y_arr[cal_idx], mu_cal, sigma_cal_fit, xc)

            mu_val = zv_mu[:, :k] @ comp_k + mean_
            prop_val = np.sqrt(np.maximum((zv_std[:, :k] ** 2) @ (comp_k ** 2), 0.0))
            sigma_val = np.maximum(np.sqrt(prop_val ** 2 + residual_rmse ** 2), base.sigma_floor)
            if base.calibrate_sigma:
                sigma_val = calibrator.transform(sigma_val, xv)
            evaluation = evaluate_prediction(
                y_arr[val_idx], ModelPrediction(mu=mu_val, sigma=sigma_val), naive_reference=naive_ref
            )
            per_k_rows[k].append(evaluation.as_dict())

    candidates: list[SearchCandidateResult] = []
    for k in grid:
        rows = per_k_rows[k]
        mean_metrics = {key: float(np.mean([r[key] for r in rows])) for key in rows[0].keys()}
        candidates.append(
            SearchCandidateResult(
                model_name=model_name,
                model_config=replace(base, n_components=k, random_state=random_state),
                mean_metrics=mean_metrics,
            )
        )
    chooser = max if maximize else min
    best = chooser(candidates, key=lambda item: item.mean_metrics[selection_metric])
    return HyperparameterSearchResult(candidates=candidates, best_candidate=best)


def cross_validate_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_name: str = "bayesian_ridge",
    model_config: ModelConfig | None = None,
    n_splits: int = 5,
    groups: np.ndarray | None = None,
    random_state: int = 42,
    sigma_cal_fraction: float = 0.0,
) -> CrossValidationResult:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    splitter = make_cv_splitter(n_splits=n_splits, groups=groups, random_state=random_state)
    split_groups = None if groups is None else np.asarray(groups)
    fold_results = [
        train_model_on_indices(
            x_arr,
            y_arr,
            train_idx,
            val_idx,
            model_name=model_name,
            model_config=model_config,
            sigma_cal_fraction=sigma_cal_fraction,
            random_state=random_state,
        )
        for train_idx, val_idx in splitter.split(x_arr, y_arr, groups=split_groups)
    ]
    return CrossValidationResult(fold_results=fold_results)


def make_train_validation_split(
    *,
    n_samples: int,
    validation_fraction: float,
    groups: np.ndarray | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_samples)
    if groups is None:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=random_state,
            shuffle=True,
        )
        return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)

    groups_arr = np.asarray(groups)
    unique_groups = np.unique(groups_arr)
    train_groups, val_groups = train_test_split(
        unique_groups,
        test_size=validation_fraction,
        random_state=random_state,
        shuffle=True,
    )
    train_mask = np.isin(groups_arr, train_groups)
    val_mask = np.isin(groups_arr, val_groups)
    return indices[train_mask], indices[val_mask]


def make_cv_splitter(n_splits: int, groups: np.ndarray | None, random_state: int):
    if groups is None:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return GroupKFold(n_splits=n_splits)


@dataclass(frozen=True)
class EnsembleBuildResult:
    ensemble: WeightedEnsembleRegressor
    model_names: list[str]
    weights: np.ndarray
    val_scores: np.ndarray


def build_gll_weighted_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_names: Iterable[str],
    model_config: ModelConfig | None = None,
    validation_fraction: float = 0.2,
    groups: np.ndarray | None = None,
    random_state: int = 42,
    temperature: float = 0.1,
    sigma_true: float = 1e-5,
) -> EnsembleBuildResult:
    names = [str(name) for name in model_names]
    if not names:
        raise ValueError("build_gll_weighted_ensemble requires at least one model name.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    train_idx, val_idx = make_train_validation_split(
        n_samples=x_arr.shape[0],
        validation_fraction=validation_fraction,
        groups=groups,
        random_state=random_state,
    )
    naive_mean, naive_sigma = ariel_naive_reference(y_arr[train_idx])

    models: list[TargetPCARegressor | ResidualCorrectedRegressor] = []
    scores: list[float] = []
    for name in names:
        model = ModelFactory.create(name, model_config)
        model.fit(x_arr[train_idx], y_arr[train_idx], x_val=x_arr[val_idx], y_val=y_arr[val_idx])
        prediction = model.predict(x_arr[val_idx])
        scores.append(
            ariel_gll_score(
                y_arr[val_idx],
                prediction.mu,
                prediction.sigma,
                naive_mean=naive_mean,
                naive_sigma=naive_sigma,
                sigma_true=sigma_true,
            )
        )
        models.append(model)

    score_arr = np.asarray(scores, dtype=float)
    logits = (score_arr - np.max(score_arr)) / temperature
    weights = np.exp(logits)
    weights = weights / np.sum(weights)
    ensemble = WeightedEnsembleRegressor(models, weights.tolist())
    return EnsembleBuildResult(
        ensemble=ensemble,
        model_names=names,
        weights=weights,
        val_scores=score_arr,
    )
