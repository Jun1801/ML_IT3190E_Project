from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ariel_ml.config import DatasetConfig
from ariel_ml.preprocessing import CalibrationBundle


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
        self.root = self.config.data_root

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
            airs_signal=self._read_parquet_array(planet_dir / f"AIRS-CH0_signal_{observation_id}.parquet"),
            fgs_signal=self._read_parquet_array(planet_dir / f"FGS1_signal_{observation_id}.parquet"),
            airs_calibration=self._load_calibration(planet_dir / f"AIRS-CH0_calibration_{observation_id}"),
            fgs_calibration=self._load_calibration(planet_dir / f"FGS1_calibration_{observation_id}"),
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
        gain = float(row["gain"]) if "gain" in row else 1.0
        offset = float(row["offset"]) if "offset" in row else 0.0
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
            return None
        return row.iloc[0].to_dict()

    def load_targets(self) -> pd.DataFrame:
        return pd.read_csv(self.root / self.config.train_targets_file)

    def load_wavelengths(self) -> pd.DataFrame:
        return pd.read_csv(self.root / self.config.wavelengths_file)

    def _load_calibration(self, directory: Path) -> CalibrationBundle:
        return CalibrationBundle(
            dead=self._read_optional_parquet_array(directory / "dead.parquet"),
            dark=self._read_optional_parquet_array(directory / "dark.parquet"),
            flat=self._read_optional_parquet_array(directory / "flat.parquet"),
            linear_corr=self._read_optional_parquet_array(directory / "linear_corr.parquet"),
        )

    def _read_optional_parquet_array(self, path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        return self._read_parquet_array(path)

    def _read_parquet_array(self, path: Path) -> np.ndarray:
        frame = pd.read_parquet(path)
        return frame.to_numpy()
