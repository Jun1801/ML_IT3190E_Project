import unittest
import tempfile
from pathlib import Path

import joblib
import numpy as np

from config import ModelConfig
from metrics import FeatureConditionedSigmaCalibrator, SigmaCalibrator, gaussian_nll
from models import ResidualCorrectedRegressor
from estimators import (
    MODEL_FAMILIES,
    _resolve_device,
    BayesianRidgePCARegressor,
    BoostingPCARegressor,
    ExtraTreesPCARegressor,
    KernelRidgePCARegressor,
    ModelFactory,
    RidgePCARegressor,
)


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


class MetricsTests(unittest.TestCase):
    def test_sigma_calibrator_improves_over_tiny_sigma(self):
        y = np.array([[1.0, 2.0], [1.1, 1.9]])
        mu = np.array([[1.05, 2.05], [1.05, 1.95]])
        sigma = np.full_like(y, 0.001)

        before = gaussian_nll(y, mu, sigma)
        calibrated = SigmaCalibrator().fit_transform(y, mu, sigma)
        after = gaussian_nll(y, mu, calibrated)

        self.assertLess(after, before)

    def test_sigma_calibrator_shrinks_overwide_sigma(self):
        y = np.array([[1.0, 2.0], [1.1, 1.9]])
        mu = np.array([[1.01, 2.01], [1.09, 1.91]])
        sigma = np.full_like(y, 0.1)

        calibrated = SigmaCalibrator().fit_transform(y, mu, sigma)

        self.assertLess(float(np.mean(calibrated)), 0.02)

    def test_per_target_calibration_beats_scalar_when_miscalibration_differs(self):
        rng = np.random.default_rng(0)
        n = 400
        # Column 0 is over-confident (sigma too small); column 1 under-confident.
        y = np.column_stack([rng.normal(0.0, 0.1, n), rng.normal(0.0, 0.001, n)])
        mu = np.zeros_like(y)
        sigma = np.column_stack([np.full(n, 0.01), np.full(n, 0.1)])

        scalar = SigmaCalibrator(per_target=False).fit(y, mu, sigma)
        per_target = SigmaCalibrator(per_target=True).fit(y, mu, sigma)

        nll_scalar = gaussian_nll(y, mu, scalar.transform(sigma))
        nll_per_target = gaussian_nll(y, mu, per_target.transform(sigma))

        self.assertLess(nll_per_target, nll_scalar)
        self.assertEqual(np.shape(per_target.scale_), (2,))
        self.assertTrue(np.isscalar(scalar.scale_))

    def test_feature_conditioned_beats_per_target_when_noise_depends_on_feature(self):
        rng = np.random.default_rng(0)
        n_rows, n_cols = 300, 12
        feature = rng.uniform(-1.0, 1.0, n_rows)
        row_scale = np.exp(0.9 * feature)  # per-row noise multiplier driven by feature 0
        col_sigma = 0.01 * (1.0 + np.arange(n_cols))  # heteroscedastic columns
        noise = rng.normal(size=(n_rows, n_cols)) * (row_scale[:, None] * col_sigma[None, :])
        mu = np.zeros((n_rows, n_cols))
        y = mu + noise
        sigma = np.tile(col_sigma, (n_rows, 1))  # right columns, but blind to per-row scale
        x = np.column_stack([feature, rng.normal(size=n_rows)])  # feature 0 informative, 1 noise

        per_target = SigmaCalibrator(per_target=True).fit(y, mu, sigma)
        feature_cond = FeatureConditionedSigmaCalibrator().fit(y, mu, sigma, x)

        nll_per_target = gaussian_nll(y, mu, per_target.transform(sigma))
        nll_feature_cond = gaussian_nll(y, mu, feature_cond.transform(sigma, x))

        self.assertLess(nll_feature_cond, nll_per_target)

    def test_feature_conditioned_falls_back_without_features(self):
        x, y = make_synthetic_regression(n_samples=40, n_targets=8)
        mu = np.zeros_like(y)
        sigma = np.full_like(y, 0.05)
        calibrator = FeatureConditionedSigmaCalibrator().fit(y, mu, sigma, None)

        self.assertIsNone(calibrator.weights_)
        # With no feature regression the multiplier is identity -> pure per-wavelength scale.
        calibrated = calibrator.transform(sigma, None)
        self.assertEqual(calibrated.shape, sigma.shape)
        self.assertTrue(np.all(calibrated > 0))


class ModelTests(unittest.TestCase):
    def test_bayesian_ridge_pca_fit_predict_shapes_and_positive_sigma(self):
        x, y = make_synthetic_regression()
        model = BayesianRidgePCARegressor(ModelConfig(n_components=3))

        model.fit(x[:36], y[:36], x_val=x[36:], y_val=y[36:])
        prediction = model.predict(x[36:])

        self.assertEqual(prediction.mu.shape, y[36:].shape)
        self.assertEqual(prediction.sigma.shape, y[36:].shape)
        self.assertTrue(np.all(prediction.sigma > 0))

    def test_model_factory_creates_all_plan_models(self):
        names = [
            "bayesian_ridge",
            "ridge",
            "kernel_ridge",
            "extra_trees",
            "boosting",
            "br_boosting_residual",
        ]
        for name in names:
            model = ModelFactory.create(name, ModelConfig(n_components=2))
            self.assertIsNotNone(model)

    def test_baseline_models_train_on_small_synthetic_data(self):
        x, y = make_synthetic_regression(n_samples=30, n_targets=8)
        config = ModelConfig(n_components=2, calibrate_sigma=False)
        models = [
            RidgePCARegressor(config),
            KernelRidgePCARegressor(config, gamma=0.5),
            ExtraTreesPCARegressor(config, n_estimators=5, n_jobs=1),
            BoostingPCARegressor(config, n_estimators=5),
        ]

        for model in models:
            model.fit(x, y)
            prediction = model.predict(x[:4])
            self.assertEqual(prediction.mu.shape, (4, y.shape[1]))
            self.assertTrue(np.all(prediction.sigma > 0))

    def test_factory_builds_every_sklearn_family_model(self):
        x, y = make_synthetic_regression(n_samples=40, n_targets=10)
        config = ModelConfig(n_components=3, calibrate_sigma=False)
        # ngboost / lightgbm / xgboost are optional deps not installed here.
        optional = {"ngboost", "lightgbm", "xgboost", "br_lgbm_residual"}
        names = [name for name in ModelFactory.list_models() if name not in optional]
        self.assertIn("svr", names)
        self.assertIn("knn", names)
        for name in names:
            model = ModelFactory.create(name, config)
            model.fit(x[:32], y[:32])
            prediction = model.predict(x[32:])
            self.assertEqual(prediction.mu.shape, y[32:].shape, msg=name)
            self.assertTrue(np.all(prediction.sigma > 0), msg=name)

    def test_resolve_device_maps_gpu_flag_per_framework(self):
        self.assertEqual(_resolve_device(True, "xgboost"), "cuda")
        self.assertEqual(_resolve_device(False, "xgboost"), "cpu")
        self.assertEqual(_resolve_device(True, "lightgbm"), "gpu")
        self.assertEqual(_resolve_device(False, "lightgbm"), "cpu")
        with self.assertRaises(ValueError):
            _resolve_device(True, "sklearn")

    def test_use_gpu_is_ignored_by_cpu_only_models(self):
        # use_gpu must not break models that have no GPU backend.
        x, y = make_synthetic_regression(n_samples=24, n_targets=8)
        model = ModelFactory.create("ridge", ModelConfig(n_components=2, calibrate_sigma=False, use_gpu=True))
        model.fit(x, y)
        self.assertEqual(model.predict(x[:3]).mu.shape, (3, y.shape[1]))

    def test_model_families_cover_listed_models(self):
        flat = ModelFactory.list_models()
        self.assertEqual(sorted(flat), sorted(set(flat)))  # no duplicates
        self.assertEqual(
            set(flat),
            {name for names in MODEL_FAMILIES.values() for name in names},
        )

    def test_residual_corrected_model_trains_and_predicts(self):
        x, y = make_synthetic_regression(n_samples=36, n_targets=8)
        config = ModelConfig(n_components=2, calibrate_sigma=False)
        model = ResidualCorrectedRegressor(
            BayesianRidgePCARegressor(config),
            BoostingPCARegressor(config, n_estimators=5),
            eta=0.2,
        )

        model.fit(x[:28], y[:28], x_val=x[28:], y_val=y[28:])
        prediction = model.predict(x[28:])

        self.assertEqual(prediction.mu.shape, y[28:].shape)
        self.assertTrue(np.all(prediction.sigma > 0))

    def test_fitted_model_can_be_saved_with_joblib(self):
        x, y = make_synthetic_regression(n_samples=24, n_targets=8)
        model = RidgePCARegressor(ModelConfig(n_components=2, calibrate_sigma=False))
        model.fit(x, y)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.joblib"
            joblib.dump({"model": model}, path)
            loaded = joblib.load(path)["model"]

        prediction = loaded.predict(x[:3])
        self.assertEqual(prediction.mu.shape, (3, y.shape[1]))


if __name__ == "__main__":
    unittest.main()
