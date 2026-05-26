from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ariel_ml.io import ArielDataRepository, RawObservation
from ariel_ml.pipeline import ArielPreprocessFeaturePipeline


class ObservationRepository(Protocol):
    def list_planet_ids(self, split: str) -> list[str]:
        ...

    def list_observation_ids(self, split: str, planet_id: str) -> list[int]:
        ...

    def load_observation(self, split: str, planet_id: str, observation_id: int = 0) -> RawObservation:
        ...

    def get_adc_params(self, instrument: str, planet_id: str | None = None) -> tuple[float, float]:
        ...

    def axis_info_features(self) -> dict[str, float]:
        ...


@dataclass(frozen=True)
class FeatureBuildResult:
    features: pd.DataFrame
    failures: list[str]


class ArielDatasetBuilder:
    def __init__(
        self,
        repository: ObservationRepository | None = None,
        pipeline: ArielPreprocessFeaturePipeline | None = None,
    ) -> None:
        self.repository = repository or ArielDataRepository()
        self.pipeline = pipeline or ArielPreprocessFeaturePipeline()
        self._axis_features = self._load_axis_features()

    def build_split_features(
        self,
        split: str,
        *,
        planet_ids: list[str] | None = None,
        observation_ids: list[int] | None = None,
        aggregate_observations: bool = True,
        limit: int | None = None,
        on_error: str = "raise",
    ) -> FeatureBuildResult:
        selected_planets = planet_ids or self.repository.list_planet_ids(split)
        if limit is not None:
            selected_planets = selected_planets[:limit]

        rows: list[dict[str, float | int | str]] = []
        failures: list[str] = []
        for planet_id in selected_planets:
            ids = observation_ids or self.repository.list_observation_ids(split, planet_id) or [0]
            for observation_id in ids:
                try:
                    rows.append(self._build_observation_row(split, planet_id, observation_id))
                except Exception as exc:
                    message = f"{split}/{planet_id}/obs_{observation_id}: {exc}"
                    if on_error == "skip":
                        failures.append(message)
                        continue
                    raise RuntimeError(message) from exc

        frame = pd.DataFrame(rows)
        if frame.empty:
            return FeatureBuildResult(features=frame, failures=failures)
        if aggregate_observations:
            frame = self.aggregate_observation_features(frame)
        return FeatureBuildResult(features=frame.fillna(0.0), failures=failures)

    def build_feature_csv(self, split: str, output_path: Path | str, **kwargs) -> FeatureBuildResult:
        result = self.build_split_features(split, **kwargs)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.features.to_csv(output, index=False)
        return result

    def aggregate_observation_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if "planet_id" not in features.columns:
            raise ValueError("features must contain a planet_id column.")
        excluded = {"planet_id", "observation_id"}
        feature_columns = [col for col in features.columns if col not in excluded]
        aggregated = features.groupby("planet_id", sort=True)[feature_columns].mean(numeric_only=True)
        return aggregated.reset_index()

    def _build_observation_row(
        self,
        split: str,
        planet_id: str,
        observation_id: int,
    ) -> dict[str, float | int | str]:
        observation = self.repository.load_observation(split, planet_id, observation_id)
        features = self.pipeline.run_observation(
            observation,
            airs_adc=self.repository.get_adc_params("AIRS-CH0", planet_id),
            fgs_adc=self.repository.get_adc_params("FGS1", planet_id),
        )
        return {
            "planet_id": str(planet_id),
            "observation_id": int(observation_id),
            **self._axis_features,
            **features,
        }

    def _load_axis_features(self) -> dict[str, float]:
        axis_loader = getattr(self.repository, "axis_info_features", None)
        if axis_loader is None:
            return {}
        return axis_loader()


def align_features_and_targets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    id_column: str = "planet_id",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    merged = features.copy()
    merged[id_column] = merged[id_column].astype(str)
    target_frame = targets.copy()
    target_frame[id_column] = target_frame[id_column].astype(str)
    merged = merged.merge(target_frame, on=id_column, how="inner", suffixes=("", "_target"))

    target_columns = [col for col in targets.columns if col != id_column]
    metadata_columns = {id_column, "observation_id", *target_columns}
    feature_columns = [
        col
        for col in merged.columns
        if col not in metadata_columns and pd.api.types.is_numeric_dtype(merged[col])
    ]
    x = merged[feature_columns].fillna(0.0)
    y = merged[target_columns].to_numpy(dtype=float)
    groups = merged[id_column].to_numpy()
    return x, y, groups, target_columns
