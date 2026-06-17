import unittest
import warnings

import numpy as np

from config import FeatureConfig, PreprocessConfig
from features import ArielFeatureBuilder
from pipeline import ArielPreprocessFeaturePipeline
from preprocessing import (
    DetectorCalibrator,
    LightCurveExtractor,
    LightCurves,
    TransitBoundaryDetector,
    TransitBounds,
)


class PreprocessingTests(unittest.TestCase):
    def test_calibrator_applies_adc_and_temporal_binning(self):
        config = PreprocessConfig(target_time_bins=2, apply_cds=False)
        calibrator = DetectorCalibrator(config)
        raw = np.array([[1.0], [2.0], [3.0], [4.0]])

        result = calibrator.calibrate(raw, gain=2.0, offset=1.0)

        np.testing.assert_allclose(result.signal[:, 0], np.array([1.0, 5.0]))

    def test_light_curve_extractor_collapses_expected_axes(self):
        airs = np.ones((5, 2, 3))
        fgs = np.ones((5, 4, 4))

        curves = LightCurveExtractor().extract(airs, fgs)

        self.assertEqual(curves.airs.shape, (5, 3))
        self.assertEqual(curves.fgs.shape, (5,))
        np.testing.assert_allclose(curves.airs, 2.0)
        np.testing.assert_allclose(curves.fgs, 16.0)

    def test_transit_detector_finds_central_dip(self):
        curve = np.ones(80)
        curve[35:45] = 0.97

        bounds = TransitBoundaryDetector(PreprocessConfig(apply_cds=False)).detect(curve)

        self.assertLessEqual(bounds.ingress_end, 36)
        self.assertGreaterEqual(bounds.egress_start, 44)
        self.assertTrue(bounds.in_mask[40])

    def test_hot_pixel_mask_ignores_all_nan_pixels_without_warning(self):
        signal = np.ones((5, 2, 2), dtype=float)
        signal[:, 0, 0] = np.nan
        calibrator = DetectorCalibrator(PreprocessConfig(apply_cds=False))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            masked, hot_count = calibrator._mask_hot_pixels(signal)

        self.assertEqual(hot_count, 0)
        self.assertTrue(np.isnan(masked[:, 0, 0]).all())
        self.assertFalse(caught)


class FeatureTests(unittest.TestCase):
    def test_feature_builder_recovers_depth_features(self):
        n_time = 60
        depths = np.array([0.01, 0.02, 0.03, 0.04])
        airs = np.ones((n_time, depths.size))
        fgs = np.ones(n_time)
        airs[25:35, :] *= 1.0 - depths
        fgs[25:35] *= 0.98
        bounds = TransitBounds(start=23, ingress_end=25, egress_start=35, end=37, n_points=n_time)

        features = ArielFeatureBuilder(FeatureConfig(spectral_bin_sizes=(2,))).build(
            LightCurves(airs=airs, fgs=fgs, airs_white=airs.mean(axis=1)),
            bounds,
            star_info={"planet_id": 1, "Rs": 1.2, "Ts": 5500.0},
        )

        self.assertAlmostEqual(features["airs_depth_mean"], 0.025, places=6)
        self.assertAlmostEqual(features["fgs_depth_mean"], 0.02, places=6)
        self.assertAlmostEqual(features["airs_depth_wl_002"], 0.03, places=6)
        self.assertIn("airs_spectral_bin2_000_mean", features)
        self.assertIn("star_log_Rs", features)

    def test_end_to_end_pipeline_accepts_synthetic_arrays_without_cds(self):
        n_time = 60
        depths = np.array([0.01, 0.02, 0.03])
        airs = np.ones((n_time, 2, depths.size))
        fgs = np.ones((n_time, 2, 2))
        airs[25:35, :, :] *= 1.0 - depths
        fgs[25:35, :, :] *= 0.98

        pipeline = ArielPreprocessFeaturePipeline(
            PreprocessConfig(target_time_bins=None, apply_cds=False, detrend_degree=0),
            FeatureConfig(spectral_bin_sizes=(3,)),
        )
        features = pipeline.run_arrays(airs_signal=airs, fgs_signal=fgs)

        self.assertIn("airs_depth_mean", features)
        self.assertGreater(features["airs_depth_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
