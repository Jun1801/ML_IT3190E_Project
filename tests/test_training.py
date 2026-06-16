import unittest

import numpy as np
import pandas as pd

from ariel_ml.config import ModelConfig
from ariel_ml.models import ModelPrediction
from ariel_ml.training import (
    cross_validate_model,
    evaluate_prediction,
    feature_dicts_to_frame,
    hyperparameter_search,
    make_train_validation_split,
    refit_full_model,
    targets_to_matrix,
    train_model,
)


def synthetic_data(n_samples=36, n_features=6, n_targets=10):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(n_samples, n_features))
    y = 0.01 + x[:, [0]] * np.linspace(0.001, 0.004, n_targets)
    y += x[:, [1]] * np.linspace(-0.002, 0.002, n_targets)
    return x, y


class TrainingTests(unittest.TestCase):
    def test_evaluate_prediction_returns_expected_metrics(self):
        y = np.array([[1.0, 2.0], [1.5, 2.5]])
        pred = ModelPrediction(mu=y.copy(), sigma=np.full_like(y, 0.1))

        result = evaluate_prediction(y, pred)

        self.assertAlmostEqual(result.rmse_mean, 1e-8)
        self.assertAlmostEqual(result.coverage_1sigma, 1.0)
        self.assertLess(result.gaussian_nll, 0.0)

    def test_train_model_returns_validation_result(self):
        x, y = synthetic_data()

        result = train_model(
            x,
            y,
            model_name="bayesian_ridge",
            model_config=ModelConfig(n_components=2),
            validation_fraction=0.25,
            random_state=3,
        )

        self.assertEqual(result.prediction.mu.shape[1], y.shape[1])
        self.assertGreater(result.validation_index.size, 0)
        self.assertTrue(np.all(result.prediction.sigma > 0))

    def test_group_split_keeps_groups_disjoint(self):
        groups = np.repeat(np.arange(6), 3)
        train_idx, val_idx = make_train_validation_split(
            n_samples=groups.size,
            validation_fraction=0.33,
            groups=groups,
            random_state=4,
        )

        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        self.assertTrue(train_groups.isdisjoint(val_groups))

    def test_cross_validate_model_aggregates_metrics(self):
        x, y = synthetic_data(n_samples=24)
        groups = np.repeat(np.arange(8), 3)

        result = cross_validate_model(
            x,
            y,
            model_name="ridge",
            model_config=ModelConfig(n_components=2, calibrate_sigma=False),
            n_splits=4,
            groups=groups,
        )

        self.assertEqual(len(result.fold_results), 4)
        self.assertIn("rmse_mean", result.mean_metrics)

    def test_hyperparameter_search_and_refit_full_model(self):
        x, y = synthetic_data(n_samples=24)
        result = hyperparameter_search(
            x,
            y,
            model_names=("ridge",),
            n_components_grid=(1, 2),
            base_config=ModelConfig(calibrate_sigma=False),
            n_splits=3,
        )

        self.assertEqual(len(result.candidates), 2)
        model = refit_full_model(
            x,
            y,
            model_name=result.best_candidate.model_name,
            model_config=result.best_candidate.model_config,
        )
        prediction = model.predict(x[:3])
        self.assertEqual(prediction.mu.shape, (3, y.shape[1]))

    def test_search_by_ariel_gll_score_picks_highest(self):
        x, y = synthetic_data(n_samples=30)
        result = hyperparameter_search(
            x,
            y,
            model_names=("ridge", "knn"),
            n_components_grid=(1, 2),
            base_config=ModelConfig(calibrate_sigma=False),
            n_splits=3,
            selection_metric="ariel_gll_score",
        )
        scores = [c.mean_metrics["ariel_gll_score"] for c in result.candidates]
        # Higher-is-better metric: the best candidate must be the maximum.
        self.assertAlmostEqual(
            result.best_candidate.mean_metrics["ariel_gll_score"], max(scores)
        )

    def test_search_by_nll_still_minimises(self):
        x, y = synthetic_data(n_samples=30)
        result = hyperparameter_search(
            x,
            y,
            model_names=("ridge",),
            n_components_grid=(1, 2, 3),
            base_config=ModelConfig(calibrate_sigma=False),
            n_splits=3,
            selection_metric="gaussian_nll",
        )
        nll = [c.mean_metrics["gaussian_nll"] for c in result.candidates]
        self.assertAlmostEqual(result.best_candidate.mean_metrics["gaussian_nll"], min(nll))

    def test_hyperparameter_search_with_model_params_grid(self):
        x, y = synthetic_data(n_samples=24)
        result = hyperparameter_search(
            x,
            y,
            model_names=("ridge",),
            n_components_grid=(2,),
            model_params_grid=[{"alpha": 0.1}, {"alpha": 10.0}],
            base_config=ModelConfig(calibrate_sigma=False),
            n_splits=3,
        )

        self.assertEqual(len(result.candidates), 2)
        alphas = [c.model_config.model_params.get("alpha") for c in result.candidates]
        self.assertIn(0.1, alphas)
        self.assertIn(10.0, alphas)

    def test_model_params_forwarded_to_ridge(self):
        x, y = synthetic_data()
        from ariel_ml.models import ModelFactory

        model = ModelFactory.create("ridge", ModelConfig(n_components=2, model_params={"alpha": 50.0}))
        model.fit(x, y)
        prediction = model.predict(x[:4])
        self.assertEqual(prediction.mu.shape, (4, y.shape[1]))

    def test_model_params_forwarded_to_residual_eta(self):
        x, y = synthetic_data()
        from ariel_ml.models import ModelFactory, ResidualCorrectedRegressor

        model = ModelFactory.create(
            "br_boosting_residual",
            ModelConfig(n_components=2, calibrate_sigma=False, model_params={"eta": 0.5}),
        )
        self.assertIsInstance(model, ResidualCorrectedRegressor)
        self.assertAlmostEqual(model.eta, 0.5)

    def test_feature_and_target_helpers(self):
        frame = feature_dicts_to_frame([{"b": 2.0, "a": 1.0}, {"a": 3.0}])
        self.assertEqual(list(frame.columns), ["a", "b"])
        self.assertEqual(frame.loc[1, "b"], 0.0)

        targets = pd.DataFrame({"planet_id": [1, 2], "wl_1": [0.1, 0.2], "wl_2": [0.3, 0.4]})
        matrix, columns = targets_to_matrix(targets)
        self.assertEqual(columns, ["wl_1", "wl_2"])
        self.assertEqual(matrix.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
