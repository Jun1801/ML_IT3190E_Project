"""Build fixed-shape light-curve sequence tensors for deep sequence models.

Deep models in :mod:`ariel_ml.deep_models` consume ``X`` of shape
``[samples, time, channels]``. This module turns raw observations into that
tensor by reusing the calibration / extraction / detrending pipeline and
stacking the FGS white-light curve with the (optionally wavelength-binned)
AIRS light curves as channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ariel_ml.dataset_builder import ObservationRepository
from ariel_ml.pipeline import ArielPreprocessFeaturePipeline


@dataclass(frozen=True)
class SequenceDataset:
    x: np.ndarray            # [n_samples, time, channels]
    planet_ids: list[str]
    failures: list[str]


def _bin_columns(matrix: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean-pool the column (wavelength) axis of ``[time, width]`` into ``n_bins``."""
    width = matrix.shape[1]
    if n_bins >= width:
        return matrix
    edges = np.linspace(0, width, n_bins + 1, dtype=int)
    return np.column_stack([matrix[:, edges[i]:edges[i + 1]].mean(axis=1) for i in range(n_bins)])


def observation_sequence(curves, *, wavelength_bins: int | None) -> np.ndarray:
    """Stack one observation's curves into ``[time, channels]`` (FGS + AIRS)."""
    airs = np.nan_to_num(np.asarray(curves.airs, dtype=float))  # [time, wavelength]
    fgs = np.nan_to_num(np.asarray(curves.fgs, dtype=float)).reshape(-1, 1)  # [time, 1]
    if wavelength_bins is not None:
        airs = _bin_columns(airs, wavelength_bins)
    return np.concatenate([fgs, airs], axis=1)


def build_sequence_dataset(
    repository: ObservationRepository,
    pipeline: ArielPreprocessFeaturePipeline,
    split: str,
    *,
    planet_ids: list[str] | None = None,
    limit: int | None = None,
    wavelength_bins: int | None = 64,
    observation_id: int = 0,
    on_error: str = "skip",
) -> SequenceDataset:
    """Build one sequence per planet (a single observation) aligned to planet_ids.

    All sequences are trimmed to the shortest common time length so they stack
    into a single ``[n, time, channels]`` array. Set ``wavelength_bins=None`` to
    keep every AIRS wavelength as a channel.
    """
    selected = planet_ids or repository.list_planet_ids(split)
    if limit is not None:
        selected = selected[:limit]

    sequences: list[np.ndarray] = []
    kept_ids: list[str] = []
    failures: list[str] = []
    for planet_id in selected:
        try:
            observation = repository.load_observation(split, planet_id, observation_id)
            curves = pipeline.processed_light_curves(
                observation,
                airs_adc=repository.get_adc_params("AIRS-CH0", planet_id),
                fgs_adc=repository.get_adc_params("FGS1", planet_id),
            )
            sequences.append(observation_sequence(curves, wavelength_bins=wavelength_bins))
            kept_ids.append(str(planet_id))
        except Exception as exc:  # noqa: BLE001 - skip a bad planet without aborting the build
            message = f"{split}/{planet_id}: {exc}"
            if on_error == "skip":
                failures.append(message)
                continue
            raise RuntimeError(message) from exc

    if not sequences:
        return SequenceDataset(x=np.empty((0, 0, 0), dtype=float), planet_ids=[], failures=failures)

    min_time = min(seq.shape[0] for seq in sequences)
    x = np.stack([seq[:min_time] for seq in sequences], axis=0)
    return SequenceDataset(x=x, planet_ids=kept_ids, failures=failures)
