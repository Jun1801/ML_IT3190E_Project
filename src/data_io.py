from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DatasetConfig
from preprocessing import CalibrationBundle


@dataclass(frozen=True)
class RawObservation:
    planet_id: str
    observation_id: int
    airs_signal: np.ndarray
    fgs_signal: np.ndarray
    airs_calibration: CalibrationBundle
    fgs_calibration: CalibrationBundle
    star_info: dict[str, Any] | None = None


class ArielDataRepository:
    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()
        self.root = Path(self.config.data_root)

    def list_planet_ids(self, split: str) -> list[str]:
        split_dir = self.root / split
        if not split_dir.exists():
            return []
        return sorted(path.name for path in split_dir.iterdir() if path.is_dir())

    def list_observation_ids(self, split: str, planet_id: str) -> list[int]:
        planet_dir = self.root / split / str(planet_id)
        ids: list[int] = []
        for path in planet_dir.glob("AIRS-CH0_signal_*.parquet"):
            suffix = path.stem.rsplit("_", maxsplit=1)[-1]
            if suffix.isdigit():
                ids.append(int(suffix))
        return sorted(ids)

    def load_observation(self, split: str, planet_id: str, observation_id: int = 0) -> RawObservation:
        planet_dir = self.root / split / str(planet_id)
        return RawObservation(
            planet_id=str(planet_id),
            observation_id=observation_id,
            airs_signal=self._read_signal_array(
                planet_dir / f"AIRS-CH0_signal_{observation_id}.parquet",
                "AIRS-CH0",
            ),
            fgs_signal=self._read_signal_array(
                planet_dir / f"FGS1_signal_{observation_id}.parquet",
                "FGS1",
            ),
            airs_calibration=self._load_calibration(
                planet_dir / f"AIRS-CH0_calibration_{observation_id}",
                "AIRS-CH0",
            ),
            fgs_calibration=self._load_calibration(
                planet_dir / f"FGS1_calibration_{observation_id}",
                "FGS1",
            ),
            star_info=self.get_star_info(split, planet_id),
        )

    def get_adc_params(self, instrument: str, planet_id: str | None = None) -> tuple[float, float]:
        path = self.root / self.config.adc_info_file
        if not path.exists():
            return 1.0, 0.0
        adc = pd.read_csv(path)
        selected = adc
        if "instrument" in adc.columns:
            selected = selected[adc["instrument"].astype(str) == instrument]
        if planet_id is not None and "planet_id" in selected.columns:
            selected = selected[selected["planet_id"].astype(str) == str(planet_id)]
        if selected.empty:
            return 1.0, 0.0
        row = selected.iloc[0]
        gain_column = f"{instrument}_adc_gain"
        offset_column = f"{instrument}_adc_offset"
        gain = float(row[gain_column]) if gain_column in row else float(row["gain"]) if "gain" in row else 1.0
        offset = (
            float(row[offset_column])
            if offset_column in row
            else float(row["offset"])
            if "offset" in row
            else 0.0
        )
        return gain, offset

    def get_star_info(self, split: str, planet_id: str) -> dict[str, Any] | None:
        file_name = (
            self.config.train_star_info_file
            if split == self.config.train_dir
            else self.config.test_star_info_file
        )
        path = self.root / file_name
        if not path.exists():
            return None
        info = pd.read_csv(path)
        row = info[info["planet_id"].astype(str) == str(planet_id)]
        if row.empty:
            numeric_planet = pd.to_numeric(info["planet_id"], errors="coerce")
            try:
                planet_value = float(planet_id)
            except ValueError:
                planet_value = np.nan
            row = info[numeric_planet == planet_value]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def load_targets(self) -> pd.DataFrame:
        return pd.read_csv(self.root / self.config.train_targets_file)

    def load_wavelengths(self) -> pd.DataFrame:
        return pd.read_csv(self.root / self.config.wavelengths_file)

    def load_axis_info(self) -> pd.DataFrame | None:
        path = self.root / "axis_info.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def axis_info_features(self) -> dict[str, float]:
        axis = self.load_axis_info()
        if axis is None:
            return {}
        features: dict[str, float] = {}
        for column in axis.columns:
            values = pd.to_numeric(axis[column], errors="coerce")
            if values.notna().any():
                key = column.replace("-", "_").replace(" ", "_")
                features[f"axis_{key}_mean"] = float(values.mean())
                features[f"axis_{key}_std"] = float(values.std(ddof=0))
                features[f"axis_{key}_min"] = float(values.min())
                features[f"axis_{key}_max"] = float(values.max())
        return features

    def _load_calibration(self, directory: Path, instrument: str) -> CalibrationBundle:
        return CalibrationBundle(
            dead=self._read_optional_parquet_array(directory / "dead.parquet"),
            dark=self._read_optional_parquet_array(directory / "dark.parquet"),
            flat=self._read_optional_parquet_array(directory / "flat.parquet"),
            linear_corr=self._read_optional_linear_corr(directory / "linear_corr.parquet", instrument),
            read=self._read_optional_parquet_array(directory / "read.parquet"),
        )

    def _read_optional_parquet_array(self, path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        return self._read_parquet_array(path)

    def _read_parquet_array(self, path: Path) -> np.ndarray:
        frame = pd.read_parquet(path)
        return frame.to_numpy()

    def _read_signal_array(self, path: Path, instrument: str) -> np.ndarray:
        arr = self._read_parquet_array(path)
        shape = self._detector_shape(instrument)
        if arr.ndim == 2 and arr.shape[1] == shape[0] * shape[1]:
            return arr.reshape(arr.shape[0], shape[0], shape[1])
        return arr

    def _read_optional_linear_corr(self, path: Path, instrument: str) -> np.ndarray | None:
        if not path.exists():
            return None
        arr = self._read_parquet_array(path)
        shape = self._detector_shape(instrument)
        if arr.ndim == 2 and arr.shape[1] == shape[1] and arr.shape[0] % shape[0] == 0:
            return arr.reshape(arr.shape[0] // shape[0], shape[0], shape[1])
        return arr

    def _detector_shape(self, instrument: str) -> tuple[int, int]:
        if instrument == "AIRS-CH0":
            return 32, 356
        if instrument == "FGS1":
            return 32, 32
        raise ValueError(f"Unknown instrument: {instrument}")
