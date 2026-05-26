from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ariel_ml.config import FeatureConfig
from ariel_ml.preprocessing import CalibrationMetrics, LightCurves, TransitBounds


class ArielFeatureBuilder:
    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def build(
        self,
        light_curves: LightCurves,
        bounds: TransitBounds,
        *,
        calibration_metrics: Mapping[str, CalibrationMetrics] | None = None,
        star_info: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        airs_depth = self._depth_by_wavelength(light_curves.airs, bounds)
        fgs_depth = self._depth_scalar(light_curves.fgs, bounds)
        airs_white = np.asarray(light_curves.airs_white, dtype=float)

        features: dict[str, float] = {}
        features.update(self._depth_summary_features("airs", airs_depth, light_curves.airs, bounds))
        features.update(self._depth_summary_features("fgs", np.asarray([fgs_depth]), light_curves.fgs, bounds))
        features.update(self._spectral_features(airs_depth))
        features.update(self._shape_features("fgs", light_curves.fgs, bounds))
        features.update(self._shape_features("airs_white", airs_white, bounds))
        features.update(self._noise_features("airs", light_curves.airs, airs_depth, bounds))
        features.update(self._noise_features("fgs", light_curves.fgs, np.asarray([fgs_depth]), bounds))

        if calibration_metrics:
            for prefix, metrics in calibration_metrics.items():
                features.update(metrics.as_features(prefix))

        if star_info:
            features.update(self._stellar_features(star_info, float(np.nanmean(airs_depth))))

        return self._finite_features(features)

    def _depth_by_wavelength(self, light_curve: np.ndarray, bounds: TransitBounds) -> np.ndarray:
        arr = np.asarray(light_curve, dtype=float)
        if arr.ndim != 2:
            raise ValueError("AIRS light curve must have shape [time, wavelength].")
        oot = arr[bounds.oot_mask, :]
        in_transit = arr[bounds.in_mask, :]
        return 1.0 - self._safe_mean(in_transit, axis=0) / self._safe_mean(oot, axis=0)

    def _depth_scalar(self, light_curve: np.ndarray, bounds: TransitBounds) -> float:
        arr = np.asarray(light_curve, dtype=float).reshape(-1)
        return float(1.0 - self._safe_mean(arr[bounds.in_mask]) / self._safe_mean(arr[bounds.oot_mask]))

    def _depth_summary_features(
        self,
        prefix: str,
        depth: np.ndarray,
        light_curve: np.ndarray,
        bounds: TransitBounds,
    ) -> dict[str, float]:
        depth = np.asarray(depth, dtype=float).reshape(-1)
        features = {
            f"{prefix}_depth_mean": float(np.nanmean(depth)),
            f"{prefix}_depth_median": float(np.nanmedian(depth)),
            f"{prefix}_depth_min": float(np.nanmin(depth)),
            f"{prefix}_depth_p05": float(np.nanpercentile(depth, 5)),
            f"{prefix}_depth_p10": float(np.nanpercentile(depth, 10)),
            f"{prefix}_depth_std": float(np.nanstd(depth)),
        }
        features[f"{prefix}_depth_weighted_by_noise"] = self._noise_weighted_depth(light_curve, depth, bounds)
        features[f"{prefix}_depth_mid_transit"] = self._mid_transit_depth(light_curve, bounds)

        if self.config.include_per_wavelength_depths and prefix == "airs":
            for idx, value in enumerate(depth):
                features[f"airs_depth_wl_{idx:03d}"] = float(value)
        return features

    def _spectral_features(self, depth: np.ndarray) -> dict[str, float]:
        depth = np.asarray(depth, dtype=float).reshape(-1)
        features: dict[str, float] = {}
        x = np.arange(depth.shape[0], dtype=float)
        for bin_size in self.config.spectral_bin_sizes:
            for bin_idx, start in enumerate(range(0, depth.shape[0], bin_size)):
                segment = depth[start : start + bin_size]
                if segment.size == 0:
                    continue
                prefix = f"airs_spectral_bin{bin_size}_{bin_idx:03d}"
                features[f"{prefix}_mean"] = float(np.nanmean(segment))
                features[f"{prefix}_median"] = float(np.nanmedian(segment))
                features[f"{prefix}_std"] = float(np.nanstd(segment))
                features[f"{prefix}_slope"] = self._slope(x[start : start + segment.size], segment)
                features[f"{prefix}_curvature"] = self._curvature(segment)
        return features

    def _shape_features(self, prefix: str, curve: np.ndarray, bounds: TransitBounds) -> dict[str, float]:
        arr = np.asarray(curve, dtype=float)
        if arr.ndim == 2:
            arr = np.nanmean(arr, axis=1)
        else:
            arr = arr.reshape(-1)

        in_curve = arr[bounds.in_mask]
        left_oot = arr[: bounds.start]
        right_oot = arr[bounds.end :]
        mid = len(in_curve) // 2
        first_half = in_curve[:mid] if mid > 0 else in_curve
        second_half = in_curve[mid:] if mid > 0 else in_curve

        features = {
            f"{prefix}_ingress_slope": self._safe_slope_between(arr, bounds.start, bounds.ingress_end),
            f"{prefix}_egress_slope": self._safe_slope_between(arr, bounds.egress_start, bounds.end),
            f"{prefix}_transit_duration": float(bounds.end - bounds.start),
            f"{prefix}_mid_transit_flux": float(np.nanmedian(in_curve)),
            f"{prefix}_symmetry": float(abs(np.nanmean(first_half) - np.nanmean(second_half))),
            f"{prefix}_curvature_bottom": self._curvature(in_curve),
            f"{prefix}_oot_left_trend": self._slope(np.arange(left_oot.size), left_oot),
            f"{prefix}_oot_right_trend": self._slope(np.arange(right_oot.size), right_oot),
            f"{prefix}_baseline_drift": float(np.nanmean(right_oot) - np.nanmean(left_oot))
            if left_oot.size and right_oot.size
            else 0.0,
        }
        return features

    def _noise_features(
        self,
        prefix: str,
        light_curve: np.ndarray,
        depth: np.ndarray,
        bounds: TransitBounds,
    ) -> dict[str, float]:
        arr = np.asarray(light_curve, dtype=float)
        oot = arr[bounds.oot_mask] if arr.ndim == 1 else arr[bounds.oot_mask, :]
        in_transit = arr[bounds.in_mask] if arr.ndim == 1 else arr[bounds.in_mask, :]

        oot_std = np.nanstd(oot, axis=0)
        in_std = np.nanstd(in_transit, axis=0)
        snr = np.asarray(depth).reshape(-1) / np.maximum(np.asarray(oot_std).reshape(-1), self.config.epsilon)

        features = {
            f"{prefix}_oot_std_mean": float(np.nanmean(oot_std)),
            f"{prefix}_oot_std_max": float(np.nanmax(oot_std)),
            f"{prefix}_in_transit_std_mean": float(np.nanmean(in_std)),
            f"{prefix}_snr_depth_mean": float(np.nanmean(snr)),
            f"{prefix}_snr_depth_median": float(np.nanmedian(snr)),
        }
        if self.config.include_per_wavelength_noise and prefix == "airs":
            for idx, value in enumerate(np.asarray(oot_std).reshape(-1)):
                features[f"airs_oot_std_wl_{idx:03d}"] = float(value)
        return features

    def _stellar_features(self, star_info: Mapping[str, Any], depth_mean: float) -> dict[str, float]:
        features: dict[str, float] = {}
        for key, value in star_info.items():
            if key == "planet_id":
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            features[f"star_{key}"] = numeric
            if key in {"Rs", "Ms", "Ts"} and numeric > 0:
                features[f"star_log_{key}"] = float(np.log(numeric))

        for key in ("Rs", "Ts", "logg"):
            feature_key = f"star_{key}"
            if feature_key in features:
                features[f"interaction_depth_mean_x_{key}"] = depth_mean * features[feature_key]
        return features

    def _noise_weighted_depth(self, light_curve: np.ndarray, depth: np.ndarray, bounds: TransitBounds) -> float:
        arr = np.asarray(light_curve, dtype=float)
        oot = arr[bounds.oot_mask] if arr.ndim == 1 else arr[bounds.oot_mask, :]
        noise = np.asarray(np.nanstd(oot, axis=0)).reshape(-1)
        weights = 1.0 / np.maximum(noise, self.config.epsilon)
        depth = np.asarray(depth, dtype=float).reshape(-1)
        return float(np.nansum(depth * weights) / np.nansum(weights))

    def _mid_transit_depth(self, light_curve: np.ndarray, bounds: TransitBounds) -> float:
        arr = np.asarray(light_curve, dtype=float)
        mid_idx = (bounds.ingress_end + bounds.egress_start) // 2
        oot_mean = self._safe_mean(arr[bounds.oot_mask], axis=0)
        mid_flux = arr[mid_idx]
        return float(np.nanmean(1.0 - mid_flux / oot_mean))

    def _safe_mean(self, values: np.ndarray, axis: int | None = None) -> np.ndarray:
        mean = np.nanmean(values, axis=axis)
        return np.where(np.abs(mean) < self.config.epsilon, self.config.epsilon, mean)

    def _safe_slope_between(self, curve: np.ndarray, left: int, right: int) -> float:
        if right <= left or left < 0 or right >= curve.size:
            return 0.0
        return float((curve[right] - curve[left]) / max(1, right - left))

    def _slope(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(mask) < 2:
            return 0.0
        return float(np.polyfit(x[mask], y[mask], deg=1)[0])

    def _curvature(self, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        if y.size < 3:
            return 0.0
        return float(np.nanmean(np.diff(y, n=2)))

    def _finite_features(self, features: dict[str, float]) -> dict[str, float]:
        clean: dict[str, float] = {}
        for key, value in features.items():
            numeric = float(value)
            clean[key] = numeric if np.isfinite(numeric) else 0.0
        return clean
