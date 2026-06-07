from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    data_root: Path = Path("data")
    train_dir: str = "train"
    test_dir: str = "test"
    adc_info_file: str = "adc_info.csv"
    train_targets_file: str = "train.csv"
    train_star_info_file: str = "train_star_info.csv"
    test_star_info_file: str = "test_star_info.csv"
    wavelengths_file: str = "wavelengths.csv"


@dataclass(frozen=True)
class PreprocessConfig:
    target_time_bins: int | None = 300
    bin_mode: str = "mean"
    hot_pixel_sigma: float = 8.0
    bin_before_spatial_calibration: bool = True
    apply_linearity: bool = False
    apply_cds: bool = True
    cds_mode: str = "pairwise"
    smooth_window: int = 5
    detrend_degree: int = 2
    fallback_transit_fraction: float = 0.25
    min_transit_points: int = 3
    epsilon: float = 1e-8


@dataclass(frozen=True)
class FeatureConfig:
    spectral_bin_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    include_per_wavelength_depths: bool = True
    include_per_wavelength_noise: bool = False
    epsilon: float = 1e-8


@dataclass(frozen=True)
class ModelConfig:
    n_components: int = 30
    standardize_features: bool = True
    residual_floor: float = 1e-8
    sigma_floor: float = 1e-8
    calibrate_sigma: bool = True
    random_state: int = 42
    model_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeepModelConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_size: int = 128
    dropout: float = 0.1
    random_state: int = 42
    device: str = "auto"
    sigma_floor: float = 1e-6
