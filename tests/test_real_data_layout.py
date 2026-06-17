from pathlib import Path

import numpy as np
import pandas as pd

from config import DatasetConfig
from data_io import ArielDataRepository


def test_repository_reads_real_sample_shapes_when_present():
    root = Path("data")
    train_dir = root / "train"
    if not train_dir.exists():
        return

    planet_ids = [path.name for path in train_dir.iterdir() if path.is_dir()]
    if not planet_ids:
        return

    repo = ArielDataRepository(DatasetConfig(data_root=root))
    observation = repo.load_observation("train", planet_ids[0], 0)

    assert observation.airs_signal.ndim == 3
    assert observation.airs_signal.shape[1:] == (32, 356)
    assert observation.fgs_signal.ndim == 3
    assert observation.fgs_signal.shape[1:] == (32, 32)
    assert observation.airs_calibration.linear_corr is None or observation.airs_calibration.linear_corr.shape[1:] == (32, 356)
    assert observation.fgs_calibration.linear_corr is None or observation.fgs_calibration.linear_corr.shape[1:] == (32, 32)
    assert observation.airs_calibration.read is None or observation.airs_calibration.read.shape == (32, 356)
    assert observation.fgs_calibration.read is None or observation.fgs_calibration.read.shape == (32, 32)


def test_repository_reads_axis_info_features_when_present():
    root = Path("data")
    if not (root / "axis_info.parquet").exists():
        return

    repo = ArielDataRepository(DatasetConfig(data_root=root))
    features = repo.axis_info_features()

    assert "axis_AIRS_CH0_integration_time_mean" in features
    assert "axis_AIRS_CH0_axis2_um_min" in features


def test_repository_reads_instrument_specific_adc_columns(tmp_path):
    pd.DataFrame(
        {
            "FGS1_adc_offset": [-1000.0],
            "FGS1_adc_gain": [0.4],
            "AIRS-CH0_adc_offset": [-900.0],
            "AIRS-CH0_adc_gain": [0.5],
        }
    ).to_csv(tmp_path / "adc_info.csv", index=False)

    repo = ArielDataRepository(DatasetConfig(data_root=tmp_path))

    assert repo.get_adc_params("AIRS-CH0") == (0.5, -900.0)
    assert repo.get_adc_params("FGS1") == (0.4, -1000.0)


def test_repository_accepts_string_data_root(tmp_path):
    repo = ArielDataRepository(DatasetConfig(data_root=str(tmp_path)))

    assert repo.root == tmp_path
