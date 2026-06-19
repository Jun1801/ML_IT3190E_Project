from __future__ import annotations

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, Lasso, Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from config import ModelConfig
from models import ModelPrediction, ResidualCorrectedRegressor, TargetPCARegressor, WeightedEnsembleRegressor


def _resolve_device(use_gpu: bool, framework: str) -> str:
    """Map ``ModelConfig.use_gpu`` to the device string of a GBM framework."""
    if framework == "xgboost":
        return "cuda" if use_gpu else "cpu"
    if framework == "lightgbm":
        return "gpu" if use_gpu else "cpu"
    raise ValueError(f"Unknown GBM framework: {framework}")


# ---------------------------------------------------------------------------
# Bayesian family
# ---------------------------------------------------------------------------

class BayesianRidgePCARegressor(TargetPCARegressor):
    def __init__(self, config: ModelConfig | None = None, *, max_iter: int = 1000, tol: float = 1e-6) -> None:
        super().__init__(
            lambda: BayesianRidge(max_iter=max_iter, tol=tol),
            config=config,
        )


class ARDPCARegressor(TargetPCARegressor):
    """Automatic Relevance Determination — Bayesian sibling of BayesianRidge."""

    def __init__(self, config: ModelConfig | None = None, *, max_iter: int = 300, tol: float = 1e-3) -> None:
        super().__init__(lambda: ARDRegression(max_iter=max_iter, tol=tol), config=config)


class GaussianProcessPCARegressor(TargetPCARegressor):
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        length_scale: float = 1.0,
        noise_level: float = 1e-3,
        alpha: float = 1e-6,
    ) -> None:
        cfg = config or ModelConfig()
        super().__init__(
            lambda: self._create_estimator(cfg, length_scale, noise_level, alpha),
            config=cfg,
        )

    def _create_estimator(self, cfg: ModelConfig, length_scale: float, noise_level: float, alpha: float):
        kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            random_state=cfg.random_state,
        )


class NGBoostPCARegressor(TargetPCARegressor):
    """Natural Gradient Boosting — produces a predictive distribution per component."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        n_estimators: int = 300,
        learning_rate: float = 0.01,
    ) -> None:
        cfg = config or ModelConfig()
        super().__init__(
            lambda: self._create_estimator(cfg, n_estimators, learning_rate),
            config=cfg,
        )

    def _create_estimator(self, cfg: ModelConfig, n_estimators: int, learning_rate: float):
        try:
            from ngboost import NGBRegressor
        except ImportError as exc:
            raise ImportError("Install ngboost to use NGBoostPCARegressor.") from exc
        return NGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=cfg.random_state,
            verbose=False,
        )

    def _predict_one_model(self, model: RegressorMixin, x_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dist = model.pred_dist(x_scaled)
        params = getattr(dist, "params", {})
        mu = np.asarray(params.get("loc", model.predict(x_scaled)), dtype=float)
        std = np.asarray(params.get("scale", np.zeros(mu.shape[0])), dtype=float)
        return mu, std


# ---------------------------------------------------------------------------
# Linear family
# ---------------------------------------------------------------------------

class RidgePCARegressor(TargetPCARegressor):
    def __init__(self, config: ModelConfig | None = None, *, alpha: float = 1.0) -> None:
        super().__init__(lambda: Ridge(alpha=alpha), config=config)


class LassoPCARegressor(TargetPCARegressor):
    def __init__(self, config: ModelConfig | None = None, *, alpha: float = 1e-3, max_iter: int = 5000) -> None:
        super().__init__(lambda: Lasso(alpha=alpha, max_iter=max_iter), config=config)


class ElasticNetPCARegressor(TargetPCARegressor):
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        alpha: float = 1e-3,
        l1_ratio: float = 0.5,
        max_iter: int = 5000,
    ) -> None:
        super().__init__(
            lambda: ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter),
            config=config,
        )


# ---------------------------------------------------------------------------
# Kernel / SVM family
# ---------------------------------------------------------------------------

class KernelRidgePCARegressor(TargetPCARegressor):
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        alpha: float = 1.0,
        kernel: str = "rbf",
        gamma: float | None = None,
    ) -> None:
        super().__init__(
            lambda: KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma),
            config=config,
        )


class SVRPCARegressor(TargetPCARegressor):
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 1e-3,
        gamma: str | float = "scale",
    ) -> None:
        super().__init__(
            lambda: SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma),
            config=config,
        )


# ---------------------------------------------------------------------------
# Neighbors family
# ---------------------------------------------------------------------------

class KNNPCARegressor(TargetPCARegressor):
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        n_neighbors: int = 5,
        weights: str = "distance",
    ) -> None:
        super().__init__(
            lambda: KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights),
            config=config,
        )


# ---------------------------------------------------------------------------
# Tree ensemble family
# ---------------------------------------------------------------------------

class RandomForestPCARegressor(TargetPCARegressor):
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        n_estimators: int = 100,
        min_samples_leaf: int = 4,
        n_jobs: int | None = -1,
    ) -> None:
        cfg = config or ModelConfig()
        super().__init__(
            lambda: RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=cfg.random_state,
                n_jobs=n_jobs,
            ),
            config=cfg,
        )


class ExtraTreesPCARegressor(TargetPCARegressor):
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        n_estimators: int = 200,
        min_samples_leaf: int = 2,
        n_jobs: int | None = -1,
    ) -> None:
        cfg = config or ModelConfig()
        super().__init__(
            lambda: ExtraTreesRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=cfg.random_state,
                n_jobs=n_jobs,
            ),
            config=cfg,
        )


class BoostingPCARegressor(TargetPCARegressor):
    """Sklearn boosting baseline used when LightGBM/XGBoost are not installed."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        n_estimators: int = 200,
        learning_rate: float = 0.03,
        max_depth: int = 3,
    ) -> None:
        cfg = config or ModelConfig()
        super().__init__(
            lambda: GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=cfg.random_state,
            ),
            config=cfg,
        )


class HistGradientBoostingPCARegressor(TargetPCARegressor):
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_depth: int | None = None,
    ) -> None:
        cfg = config or ModelConfig()
        super().__init__(
            lambda: HistGradientBoostingRegressor(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=cfg.random_state,
            ),
            config=cfg,
        )


class LightGBMPCARegressor(TargetPCARegressor):
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int | None = -1,
    ) -> None:
        cfg = config or ModelConfig()
        super().__init__(
            lambda: self._create_estimator(
                cfg,
                n_estimators,
                learning_rate,
                num_leaves,
                min_child_samples,
                subsample,
                colsample_bytree,
                n_jobs,
            ),
            config=cfg,
        )

    def _create_estimator(
        self,
        cfg: ModelConfig,
        n_estimators: int,
        learning_rate: float,
        num_leaves: int,
        min_child_samples: int,
        subsample: float,
        colsample_bytree: float,
        n_jobs: int | None,
    ):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("Install lightgbm to use LightGBMPCARegressor.") from exc
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=cfg.random_state,
            n_jobs=n_jobs,
            device=_resolve_device(cfg.use_gpu, "lightgbm"),
            verbosity=-1,
        )


class XGBoostPCARegressor(TargetPCARegressor):
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int | None = -1,
    ) -> None:
        cfg = config or ModelConfig()
        super().__init__(
            lambda: self._create_estimator(
                cfg,
                n_estimators,
                learning_rate,
                max_depth,
                subsample,
                colsample_bytree,
                n_jobs,
            ),
            config=cfg,
        )

    def _create_estimator(
        self,
        cfg: ModelConfig,
        n_estimators: int,
        learning_rate: float,
        max_depth: int,
        subsample: float,
        colsample_bytree: float,
        n_jobs: int | None,
    ):
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("Install xgboost to use XGBoostPCARegressor.") from exc
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method="hist",
            device=_resolve_device(cfg.use_gpu, "xgboost"),
            random_state=cfg.random_state,
            n_jobs=n_jobs,
        )


# ---------------------------------------------------------------------------
# Neural (shallow) family
# ---------------------------------------------------------------------------

class MLPPCARegressor(TargetPCARegressor):
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        alpha: float = 1e-2,
        max_iter: int = 1000,
        learning_rate_init: float = 1e-3,
    ) -> None:
        cfg = config or ModelConfig()
        super().__init__(
            lambda: self._create_estimator(cfg, hidden_layer_sizes, alpha, max_iter, learning_rate_init),
            config=cfg,
        )

    def _create_estimator(self, cfg, hidden_layer_sizes, alpha, max_iter, learning_rate_init):
        # Each PCA component target has its own (small, uncentred) scale; without
        # standardising it the MLP diverges on little data. Wrap it in a target
        # scaler and use early stopping + stronger L2 so it stays well-behaved.
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=cfg.random_state,
        )
        return TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Canonical model name → family grouping. Used for organised comparison
# studies and report tables. Each canonical name is creatable via ModelFactory.
MODEL_FAMILIES: dict[str, tuple[str, ...]] = {
    "linear": ("ridge", "lasso", "elastic_net"),
    "bayesian": ("bayesian_ridge", "ard", "gaussian_process", "ngboost"),
    "kernel_svm": ("svr", "kernel_ridge"),
    "neighbors": ("knn",),
    "trees": (
        "random_forest",
        "extra_trees",
        "boosting",
        "hist_gradient_boosting",
        "lightgbm",
        "xgboost",
    ),
    "neural": ("mlp",),
    "hybrid": ("br_lgbm_residual", "br_boosting_residual"),
}


class ModelFactory:
    @staticmethod
    def families() -> dict[str, tuple[str, ...]]:
        """Return the canonical model name grouping by family."""
        return dict(MODEL_FAMILIES)

    @staticmethod
    def list_models() -> list[str]:
        """Flat list of every canonical model name across all families."""
        return [name for names in MODEL_FAMILIES.values() for name in names]

    @staticmethod
    def create(name: str, config: ModelConfig | None = None) -> "TargetPCARegressor | ResidualCorrectedRegressor":
        cfg = config or ModelConfig()
        params = cfg.model_params  # forwarded to the underlying estimator constructor
        normalized = name.strip().lower().replace("-", "_")
        if normalized in {"bayesian_ridge", "br"}:
            return BayesianRidgePCARegressor(cfg, **params)
        if normalized == "ridge":
            return RidgePCARegressor(cfg, **params)
        if normalized == "lasso":
            return LassoPCARegressor(cfg, **params)
        if normalized in {"elastic_net", "elasticnet"}:
            return ElasticNetPCARegressor(cfg, **params)
        if normalized in {"ard", "ard_regression"}:
            return ARDPCARegressor(cfg, **params)
        if normalized in {"gaussian_process", "gpr"}:
            return GaussianProcessPCARegressor(cfg, **params)
        if normalized in {"ngboost", "ngb"}:
            return NGBoostPCARegressor(cfg, **params)
        if normalized in {"svr", "svm"}:
            return SVRPCARegressor(cfg, **params)
        if normalized in {"knn", "kneighbors"}:
            return KNNPCARegressor(cfg, **params)
        if normalized in {"kernel_ridge", "krr"}:
            return KernelRidgePCARegressor(cfg, **params)
        if normalized in {"random_forest", "rf"}:
            return RandomForestPCARegressor(cfg, **params)
        if normalized in {"extra_trees", "extratrees", "random_forest_like"}:
            return ExtraTreesPCARegressor(cfg, **params)
        if normalized in {"boosting", "gradient_boosting", "lightgbm_like"}:
            return BoostingPCARegressor(cfg, **params)
        if normalized in {"hist_gradient_boosting", "hist_gb", "histgb"}:
            return HistGradientBoostingPCARegressor(cfg, **params)
        if normalized in {"mlp", "neural_network"}:
            return MLPPCARegressor(cfg, **params)
        if normalized in {"lightgbm", "lgbm"}:
            return LightGBMPCARegressor(cfg, **params)
        if normalized in {"xgboost", "xgb"}:
            return XGBoostPCARegressor(cfg, **params)
        if normalized in {"bayesian_ridge_lgbm_residual", "br_lgbm_residual"}:
            residual_params = dict(params)
            eta = residual_params.pop("eta", 0.3)
            return ResidualCorrectedRegressor(
                BayesianRidgePCARegressor(cfg),
                LightGBMPCARegressor(cfg, **residual_params),
                eta=eta,
                sigma_floor=cfg.sigma_floor,
            )
        if normalized in {"bayesian_ridge_boosting_residual", "br_boosting_residual"}:
            residual_params = dict(params)
            eta = residual_params.pop("eta", 0.3)
            return ResidualCorrectedRegressor(
                BayesianRidgePCARegressor(cfg),
                BoostingPCARegressor(cfg, **residual_params),
                eta=eta,
                sigma_floor=cfg.sigma_floor,
            )
        raise ValueError(f"Unknown model name: {name}")
