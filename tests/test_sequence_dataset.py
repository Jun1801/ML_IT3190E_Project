import unittest

import numpy as np

from config import PreprocessConfig
from data_io import RawObservation
from pipeline import ArielPreprocessFeaturePipeline
from preprocessing import CalibrationBundle
from sequence_dataset import build_sequence_dataset, observation_sequence


class FakeRepository:
    def __init__(self, n_time=40, n_wavelength=24):
        self.n_time = n_time
        self.n_wavelength = n_wavelength

    def list_planet_ids(self, split):
        return ["1", "2", "3"]

    def list_observation_ids(self, split, planet_id):
        return [0]

    def load_observation(self, split, planet_id, observation_id=0):
        rng = np.random.default_rng(int(planet_id))
        airs = 1.0 + 0.01 * rng.normal(size=(self.n_time, 2, self.n_wavelength))
        fgs = 1.0 + 0.01 * rng.normal(size=(self.n_time, 2, 2))
        airs[15:25] *= 0.99
        fgs[15:25] *= 0.99
        return RawObservation(
            planet_id=planet_id,
            observation_id=observation_id,
            airs_signal=airs,
            fgs_signal=fgs,
            airs_calibration=CalibrationBundle(),
            fgs_calibration=CalibrationBundle(),
            star_info=None,
        )

    def get_adc_params(self, instrument, planet_id=None):
        return 1.0, 0.0


def make_pipeline():
    return ArielPreprocessFeaturePipeline(
        PreprocessConfig(target_time_bins=None, apply_cds=False, detrend_degree=0)
    )


class SequenceDatasetTests(unittest.TestCase):
    def test_builds_stacked_tensor_with_binned_channels(self):
        dataset = build_sequence_dataset(
            FakeRepository(), make_pipeline(), "train", wavelength_bins=8
        )
        self.assertEqual(dataset.x.ndim, 3)
        self.assertEqual(dataset.x.shape[0], 3)
        self.assertEqual(dataset.x.shape[2], 8 + 1)
        self.assertEqual(dataset.planet_ids, ["1", "2", "3"])
        self.assertFalse(np.isnan(dataset.x).any())

    def test_wavelength_bins_none_keeps_all_channels(self):
        repo = FakeRepository(n_wavelength=24)
        seq = observation_sequence(
            make_pipeline().processed_light_curves(repo.load_observation("train", "1")),
            wavelength_bins=None,
        )
        self.assertEqual(seq.shape[1], 24 + 1)

    def test_limit_restricts_planets(self):
        dataset = build_sequence_dataset(
            FakeRepository(), make_pipeline(), "train", limit=2, wavelength_bins=4
        )
        self.assertEqual(dataset.x.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
