import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ariel_ml.config import FeatureConfig, PreprocessConfig
from ariel_ml.dataset_builder import ArielDatasetBuilder, align_features_and_targets
from ariel_ml.io import RawObservation
from ariel_ml.pipeline import ArielPreprocessFeaturePipeline
from ariel_ml.preprocessing import CalibrationBundle
from ariel_ml.submission import (
    build_submission_frame,
    infer_submission_schema,
    save_submission,
)
from ariel_ml.models import ModelPrediction


class FakeRepository:
    def list_planet_ids(self, split):
        return ["1", "2"]

    def list_observation_ids(self, split, planet_id):
        return [0, 1]

    def load_observation(self, split, planet_id, observation_id=0):
        n_time = 50
        depths = np.array([0.01, 0.02, 0.03]) + int(planet_id) * 0.001
        airs = np.ones((n_time, 2, depths.size))
        fgs = np.ones((n_time, 2, 2))
        airs[20:30, :, :] *= 1.0 - depths
        fgs[20:30, :, :] *= 0.98
        return RawObservation(
            planet_id=planet_id,
            observation_id=observation_id,
            airs_signal=airs,
            fgs_signal=fgs,
            airs_calibration=CalibrationBundle(),
            fgs_calibration=CalibrationBundle(),
            star_info={"planet_id": planet_id, "Rs": 1.0 + int(planet_id)},
        )

    def get_adc_params(self, instrument, planet_id=None):
        return 1.0, 0.0


class DatasetBuilderTests(unittest.TestCase):
    def test_build_split_features_aggregates_observations(self):
        pipeline = ArielPreprocessFeaturePipeline(
            PreprocessConfig(target_time_bins=None, apply_cds=False, detrend_degree=0),
            FeatureConfig(spectral_bin_sizes=(3,)),
        )
        builder = ArielDatasetBuilder(repository=FakeRepository(), pipeline=pipeline)

        result = builder.build_split_features("train")

        self.assertEqual(result.failures, [])
        self.assertEqual(result.features.shape[0], 2)
        self.assertIn("airs_depth_mean", result.features.columns)
        self.assertNotIn("observation_id", result.features.columns)

    def test_align_features_and_targets_returns_training_arrays(self):
        features = pd.DataFrame({"planet_id": ["1", "2"], "f1": [0.1, 0.2], "f2": [1.0, 2.0]})
        targets = pd.DataFrame({"planet_id": [1, 2], "wl_1": [0.01, 0.02], "wl_2": [0.03, 0.04]})

        x, y, groups, target_columns = align_features_and_targets(features, targets)

        self.assertEqual(list(x.columns), ["f1", "f2"])
        self.assertEqual(y.shape, (2, 2))
        self.assertEqual(list(groups), ["1", "2"])
        self.assertEqual(target_columns, ["wl_1", "wl_2"])


class SubmissionTests(unittest.TestCase):
    def test_build_submission_from_sample_schema(self):
        sample = pd.DataFrame(columns=["planet_id", "wl_1", "wl_2", "sigma_1", "sigma_2"])
        schema = infer_submission_schema(sample_submission=sample)
        prediction = ModelPrediction(
            mu=np.array([[0.1, 0.2], [0.3, 0.4]]),
            sigma=np.array([[0.01, 0.02], [0.03, 0.04]]),
        )

        frame = build_submission_frame(["a", "b"], prediction, schema)

        self.assertEqual(list(frame.columns), list(sample.columns))
        self.assertEqual(frame.loc[1, "sigma_2"], 0.04)

    def test_save_submission_creates_parent_directory(self):
        schema = infer_submission_schema(target_columns=["wl_1"])
        prediction = ModelPrediction(mu=np.array([[0.1]]), sigma=np.array([[0.01]]))
        frame = build_submission_frame(["1"], prediction, schema)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "submission.csv"
            save_submission(frame, output)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
