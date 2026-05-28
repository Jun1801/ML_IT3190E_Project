from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ariel_ml.config import PreprocessConfig


@dataclass(frozen=True)
class CalibrationBundle:
    dead: np.ndarray | None = None
    dark: np.ndarray | None = None
    flat: np.ndarray | None = None
    linear_corr: np.ndarray | None = None
    read: np.ndarray | None = None


@dataclass(frozen=True)
class CalibrationMetrics:
    bad_pixel_count: int = 0
    hot_pixel_count: int = 0
    flat_field_variance: float = 0.0
    dark_current_level: float = 0.0
    read_noise_level: float = 0.0
    cds_noise_proxy: float = 0.0

    def as_features(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_bad_pixel_count": float(self.bad_pixel_count),
            f"{prefix}_hot_pixel_count": float(self.hot_pixel_count),
            f"{prefix}_flat_field_variance": float(self.flat_field_variance),
            f"{prefix}_dark_current_level": float(self.dark_current_level),
            f"{prefix}_read_noise_level": float(self.read_noise_level),
            f"{prefix}_cds_noise_proxy": float(self.cds_noise_proxy),
        }


@dataclass(frozen=True)
class CalibrationResult:
    signal: np.ndarray
    metrics: CalibrationMetrics


@dataclass(frozen=True)
class LightCurves:
    airs: np.ndarray
    fgs: np.ndarray
    airs_white: np.ndarray


@dataclass(frozen=True)
class TransitBounds:
    start: int
    ingress_end: int
    egress_start: int
    end: int
    n_points: int

    @property
    def in_slice(self) -> slice:
        return slice(self.ingress_end, self.egress_start)

    @property
    def full_slice(self) -> slice:
        return slice(self.start, self.end)

    @property
    def oot_mask(self) -> np.ndarray:
        mask = np.ones(self.n_points, dtype=bool)
        mask[self.full_slice] = False
        return mask

    @property
    def in_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_points, dtype=bool)
        mask[self.in_slice] = True
        return mask

    @classmethod
    def centered(cls, n_points: int, fraction: float = 0.25) -> "TransitBounds":
        width = max(3, int(round(n_points * fraction)))
        start = max(0, (n_points - width) // 2)
        end = min(n_points, start + width)
        margin = max(1, width // 5)
        return cls(
            start=max(0, start - margin),
            ingress_end=start,
            egress_start=end,
            end=min(n_points, end + margin),
            n_points=n_points,
        )


class DetectorCalibrator:
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def calibrate(
        self,
        signal: np.ndarray,
        bundle: CalibrationBundle | None = None,
        *,
        gain: float = 1.0,
        offset: float = 0.0,
    ) -> CalibrationResult:
        bundle = bundle or CalibrationBundle()
        calibrated = self._as_float_array(signal)
        calibrated = (calibrated - offset) * gain

        if self.config.bin_before_spatial_calibration:
            calibrated, cds_noise_proxy = self._apply_cds_and_binning(calibrated)
        else:
            cds_noise_proxy = 0.0

        calibrated, bad_pixel_count = self._mask_dead_pixels(calibrated, bundle.dead)
        calibrated, hot_pixel_count = self._mask_hot_pixels(calibrated)

        if bundle.dark is not None:
            dark = self._broadcast_to_signal(bundle.dark, calibrated)
            calibrated = calibrated - dark
            dark_current_level = float(np.nanmean(dark))
        else:
            dark_current_level = 0.0
        read_noise_level = float(np.nanmean(bundle.read)) if bundle.read is not None else 0.0

        if self.config.apply_linearity and bundle.linear_corr is not None:
            calibrated = self._apply_linearity_correction(calibrated, bundle.linear_corr)

        if bundle.flat is not None:
            flat = self._broadcast_to_signal(bundle.flat, calibrated)
            calibrated = calibrated / np.where(np.abs(flat) < self.config.epsilon, 1.0, flat)
            flat_field_variance = float(np.nanvar(flat))
        else:
            flat_field_variance = 0.0

        calibrated = self._fill_nan(calibrated)

        if not self.config.bin_before_spatial_calibration:
            calibrated, cds_noise_proxy = self._apply_cds_and_binning(calibrated)

        return CalibrationResult(
            signal=calibrated,
            metrics=CalibrationMetrics(
                bad_pixel_count=bad_pixel_count,
                hot_pixel_count=hot_pixel_count,
                flat_field_variance=flat_field_variance,
                dark_current_level=dark_current_level,
                read_noise_level=read_noise_level,
                cds_noise_proxy=cds_noise_proxy,
            ),
        )

    def temporal_bin(self, signal: np.ndarray, n_bins: int, mode: str = "mean") -> np.ndarray:
        signal = self._as_float_array(signal)
        if signal.shape[0] <= n_bins:
            return signal.copy()
        chunks = np.array_split(signal, n_bins, axis=0)
        reducer = np.nanmedian if mode == "median" else np.nanmean
        return np.stack([reducer(chunk, axis=0) for chunk in chunks], axis=0)

    def _apply_cds_and_binning(self, signal: np.ndarray) -> tuple[np.ndarray, float]:
        processed = signal
        if self.config.apply_cds:
            processed = self._correlated_double_sample(processed)
            cds_noise_proxy = float(np.nanstd(processed - np.nanmedian(processed, axis=0)))
        else:
            cds_noise_proxy = 0.0
        if self.config.target_time_bins is not None:
            processed = self.temporal_bin(
                processed,
                n_bins=self.config.target_time_bins,
                mode=self.config.bin_mode,
            )
        return processed, cds_noise_proxy

    def _correlated_double_sample(self, signal: np.ndarray) -> np.ndarray:
        if signal.shape[0] < 2:
            return signal
        if self.config.cds_mode == "consecutive":
            return np.diff(signal, axis=0)
        even_count = (signal.shape[0] // 2) * 2
        paired = signal[:even_count]
        return paired[1::2] - paired[0::2]

    def _mask_dead_pixels(
        self,
        signal: np.ndarray,
        dead: np.ndarray | None,
    ) -> tuple[np.ndarray, int]:
        if dead is None:
            return signal, 0
        dead_mask = self._broadcast_to_signal(dead.astype(bool), signal).astype(bool)
        masked = signal.copy()
        masked[dead_mask] = np.nan
        return masked, int(np.count_nonzero(dead_mask))

    def _mask_hot_pixels(self, signal: np.ndarray) -> tuple[np.ndarray, int]:
        if signal.shape[0] < 3:
            return signal, 0
        valid = np.isfinite(signal)
        count = np.sum(valid, axis=0, keepdims=True)
        clean = np.where(valid, signal, 0.0)
        total = np.sum(clean, axis=0, keepdims=True)
        mean = np.divide(total, count, out=np.zeros_like(total, dtype=float), where=count > 0)
        diff = np.where(valid, signal - mean, 0.0)
        sumsq = np.sum(diff**2, axis=0, keepdims=True)
        variance = np.divide(sumsq, count, out=np.zeros_like(sumsq, dtype=float), where=count > 1)
        std = np.where(count > 1, np.sqrt(np.maximum(variance, 0.0)), np.inf)
        threshold = self.config.hot_pixel_sigma * np.where(std < self.config.epsilon, np.inf, std)
        hot_mask = valid & (np.abs(signal - mean) > threshold)
        masked = signal.copy()
        masked[hot_mask] = np.nan
        return masked, int(np.count_nonzero(hot_mask))

    def _apply_linearity_correction(self, signal: np.ndarray, linear_corr: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(linear_corr, dtype=float)
        if (
            coeffs.ndim == signal.ndim
            and coeffs.shape[1:] == signal.shape[1:]
            and 1 <= coeffs.shape[0] <= 8
        ):
            corrected = signal.copy()
            with np.errstate(over="ignore", invalid="ignore"):
                for idx, coeff in enumerate(coeffs, start=2):
                    corrected = corrected + self._broadcast_to_signal(coeff, signal) * np.power(signal, idx)
            return np.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0)
        if coeffs.ndim == signal.ndim + 1 and 1 <= coeffs.shape[0] <= 8:
            corrected = signal.copy()
            with np.errstate(over="ignore", invalid="ignore"):
                for idx, coeff in enumerate(coeffs, start=2):
                    corrected = corrected + self._broadcast_to_signal(coeff, signal) * np.power(signal, idx)
            return np.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0)
        return signal + self._broadcast_to_signal(coeffs, signal)

    def _broadcast_to_signal(self, value: np.ndarray, signal: np.ndarray) -> np.ndarray:
        arr = np.asarray(value)
        if arr.shape == signal.shape:
            return arr
        if arr.shape == signal.shape[1:]:
            arr = arr[np.newaxis, ...]
        return np.broadcast_to(arr, signal.shape)

    def _fill_nan(self, signal: np.ndarray) -> np.ndarray:
        if not np.isnan(signal).any():
            return signal
        fill_value = np.nanmedian(signal)
        if np.isnan(fill_value):
            fill_value = 0.0
        return np.nan_to_num(signal, nan=float(fill_value))

    def _as_float_array(self, signal: np.ndarray) -> np.ndarray:
        return np.asarray(signal, dtype=float)


class LightCurveExtractor:
    def extract(self, airs_signal: np.ndarray, fgs_signal: np.ndarray) -> LightCurves:
        airs = self.extract_airs(airs_signal)
        fgs = self.extract_fgs(fgs_signal)
        return LightCurves(airs=airs, fgs=fgs, airs_white=np.nanmean(airs, axis=1))

    def extract_airs(self, signal: np.ndarray) -> np.ndarray:
        arr = np.asarray(signal, dtype=float)
        if arr.ndim == 1:
            return arr[:, np.newaxis]
        if arr.ndim == 2:
            return arr
        spatial_axes = tuple(range(1, arr.ndim - 1))
        return np.nansum(arr, axis=spatial_axes)

    def extract_fgs(self, signal: np.ndarray) -> np.ndarray:
        arr = np.asarray(signal, dtype=float)
        if arr.ndim == 1:
            return arr
        return np.nansum(arr, axis=tuple(range(1, arr.ndim)))


class LightCurveTransformer:
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def smooth(self, curve: np.ndarray, window: int | None = None) -> np.ndarray:
        arr = np.asarray(curve, dtype=float)
        window = self._odd_window(window or self.config.smooth_window)
        if window <= 1:
            return arr.copy()
        kernel = np.ones(window, dtype=float) / window
        if arr.ndim == 1:
            return np.convolve(arr, kernel, mode="same")
        return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, arr)

    def normalize(self, curve: np.ndarray, bounds: TransitBounds) -> np.ndarray:
        arr = np.asarray(curve, dtype=float)
        oot = self._select(arr, bounds.oot_mask)
        baseline = np.nanmedian(oot, axis=0)
        baseline = np.where(np.abs(baseline) < self.config.epsilon, 1.0, baseline)
        return arr / baseline

    def detrend(self, curve: np.ndarray, bounds: TransitBounds, degree: int | None = None) -> np.ndarray:
        arr = np.asarray(curve, dtype=float)
        degree = self.config.detrend_degree if degree is None else degree
        if degree <= 0:
            return arr.copy()
        if arr.ndim == 1:
            return self._detrend_one(arr, bounds, degree)
        columns = [self._detrend_one(arr[:, idx], bounds, degree) for idx in range(arr.shape[1])]
        return np.stack(columns, axis=1)

    def _detrend_one(self, curve: np.ndarray, bounds: TransitBounds, degree: int) -> np.ndarray:
        x = np.arange(curve.shape[0], dtype=float)
        mask = bounds.oot_mask & np.isfinite(curve)
        if np.count_nonzero(mask) <= degree:
            return curve.copy()
        coeffs = np.polyfit(x[mask], curve[mask], deg=degree)
        baseline = np.polyval(coeffs, x)
        baseline = np.where(np.abs(baseline) < self.config.epsilon, 1.0, baseline)
        return curve / baseline

    def _select(self, arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr[mask]
        return arr[mask, :]

    def _odd_window(self, window: int) -> int:
        window = max(1, int(window))
        return window if window % 2 == 1 else window + 1


class TransitBoundaryDetector:
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self.transformer = LightCurveTransformer(self.config)

    def detect(self, fgs_curve: np.ndarray) -> TransitBounds:
        curve = np.asarray(fgs_curve, dtype=float).reshape(-1)
        n_points = curve.shape[0]
        if n_points < self.config.min_transit_points * 3:
            return TransitBounds.centered(n_points, self.config.fallback_transit_fraction)

        smooth = self.transformer.smooth(curve)
        derivative_bounds = self._detect_from_derivatives(smooth)
        if derivative_bounds is not None:
            return derivative_bounds

        median = float(np.nanmedian(smooth))
        min_value = float(np.nanmin(smooth))
        dip = median - min_value
        if not np.isfinite(dip) or dip <= self.config.epsilon:
            return TransitBounds.centered(n_points, self.config.fallback_transit_fraction)

        threshold = median - 0.5 * dip
        below = smooth <= threshold
        start, end = self._longest_true_run(below)
        if end - start < self.config.min_transit_points:
            return TransitBounds.centered(n_points, self.config.fallback_transit_fraction)

        width = end - start
        margin = max(1, width // 5)
        return TransitBounds(
            start=max(0, start - margin),
            ingress_end=start,
            egress_start=end,
            end=min(n_points, end + margin),
            n_points=n_points,
        )

    def _detect_from_derivatives(self, smooth: np.ndarray) -> TransitBounds | None:
        n_points = smooth.shape[0]
        if n_points < self.config.min_transit_points * 3:
            return None
        finite = np.isfinite(smooth)
        if np.count_nonzero(finite) < self.config.min_transit_points * 3:
            return None

        d1 = np.diff(smooth)
        d2 = np.diff(d1)
        if d1.size < 4 or not np.isfinite(d1).any():
            return None

        center = int(np.nanargmin(smooth))
        min_side = max(2, self.config.min_transit_points)
        if center < min_side or center > n_points - min_side - 1:
            return None

        left = d1[:center]
        right = d1[center:]
        if left.size < min_side or right.size < min_side:
            return None

        ingress = int(np.nanargmin(left))
        egress = center + int(np.nanargmax(right))
        if egress <= ingress:
            return None

        gradient_scale = float(np.nanstd(d1))
        if gradient_scale <= self.config.epsilon:
            return None
        if abs(float(d1[ingress])) < gradient_scale or abs(float(d1[egress])) < gradient_scale:
            return None

        width = max(egress - ingress, self.config.min_transit_points)
        margin = max(1, width // 5)
        start = max(0, ingress - margin)
        end = min(n_points, egress + margin + 1)

        if end - start < self.config.min_transit_points:
            return None

        return TransitBounds(
            start=start,
            ingress_end=max(start + 1, ingress + 1),
            egress_start=max(ingress + 2, egress),
            end=end,
            n_points=n_points,
        )

    def _longest_true_run(self, mask: np.ndarray) -> tuple[int, int]:
        best_start = 0
        best_end = 0
        current_start: int | None = None
        for idx, value in enumerate(mask):
            if value and current_start is None:
                current_start = idx
            if (not value or idx == len(mask) - 1) and current_start is not None:
                current_end = idx + 1 if value and idx == len(mask) - 1 else idx
                if current_end - current_start > best_end - best_start:
                    best_start, best_end = current_start, current_end
                current_start = None
        return best_start, best_end
