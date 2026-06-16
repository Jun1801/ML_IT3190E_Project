from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ariel_ml.config import FeatureConfig, PreprocessConfig
from ariel_ml.features import ArielFeatureBuilder
from ariel_ml.preprocessing import (
    CalibrationBundle,
    DetectorCalibrator,
    LightCurveExtractor,
    LightCurveTransformer,
    TransitBoundaryDetector,
)


class ArielPreprocessFeaturePipeline:
    def __init__(
        self,
        preprocess_config: PreprocessConfig | None = None,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.calibrator = DetectorCalibrator(self.preprocess_config)
        self.light_curve_extractor = LightCurveExtractor()
        self.boundary_detector = TransitBoundaryDetector(self.preprocess_config)
        self.transformer = LightCurveTransformer(self.preprocess_config)
        self.feature_builder = ArielFeatureBuilder(feature_config or FeatureConfig())

    def run_arrays(
        self,
        *,
        airs_signal,
        fgs_signal,
        airs_calibration: CalibrationBundle | None = None,
        fgs_calibration: CalibrationBundle | None = None,
        airs_adc: tuple[float, float] = (1.0, 0.0),
        fgs_adc: tuple[float, float] = (1.0, 0.0),
        star_info: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        processed_curves, bounds, metrics = self._process_to_light_curves(
            airs_signal=airs_signal,
            fgs_signal=fgs_signal,
            airs_calibration=airs_calibration,
            fgs_calibration=fgs_calibration,
            airs_adc=airs_adc,
            fgs_adc=fgs_adc,
        )
        return self.feature_builder.build(
            processed_curves,
            bounds,
            calibration_metrics=metrics,
            star_info=star_info,
        )

    def _process_to_light_curves(
        self,
        *,
        airs_signal,
        fgs_signal,
        airs_calibration: CalibrationBundle | None,
        fgs_calibration: CalibrationBundle | None,
        airs_adc: tuple[float, float],
        fgs_adc: tuple[float, float],
    ):
        airs = self.calibrator.calibrate(airs_signal, airs_calibration, gain=airs_adc[0], offset=airs_adc[1])
        fgs = self.calibrator.calibrate(fgs_signal, fgs_calibration, gain=fgs_adc[0], offset=fgs_adc[1])
        light_curves = self.light_curve_extractor.extract(airs.signal, fgs.signal)
        bounds = self.boundary_detector.detect(light_curves.fgs)

        normalized_airs = self.transformer.normalize(light_curves.airs, bounds)
        normalized_fgs = self.transformer.normalize(light_curves.fgs, bounds)
        detrended_airs = self.transformer.detrend(normalized_airs, bounds)
        detrended_fgs = self.transformer.detrend(normalized_fgs, bounds)
        processed_curves = self.light_curve_extractor.extract(detrended_airs, detrended_fgs)
        return processed_curves, bounds, {"airs": airs.metrics, "fgs": fgs.metrics}

    def run_observation(self, observation, *, airs_adc=(1.0, 0.0), fgs_adc=(1.0, 0.0)) -> dict[str, float]:
        return self.run_arrays(
            airs_signal=observation.airs_signal,
            fgs_signal=observation.fgs_signal,
            airs_calibration=observation.airs_calibration,
            fgs_calibration=observation.fgs_calibration,
            airs_adc=airs_adc,
            fgs_adc=fgs_adc,
            star_info=observation.star_info,
        )

    def processed_light_curves(self, observation, *, airs_adc=(1.0, 0.0), fgs_adc=(1.0, 0.0)):
        """Return the normalised, detrended ``LightCurves`` for one observation.

        Exposes the per-observation light curves (AIRS ``[time, wavelength]`` and
        FGS ``[time]``) that deep sequence models consume, without building the
        tabular feature dict.
        """
        curves, _bounds, _metrics = self._process_to_light_curves(
            airs_signal=observation.airs_signal,
            fgs_signal=observation.fgs_signal,
            airs_calibration=observation.airs_calibration,
            fgs_calibration=observation.fgs_calibration,
            airs_adc=airs_adc,
            fgs_adc=fgs_adc,
        )
        return curves
