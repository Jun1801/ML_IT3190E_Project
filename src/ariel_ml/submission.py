from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ariel_ml.models import ModelPrediction, TargetPCARegressor


@dataclass(frozen=True)
class SubmissionSchema:
    id_column: str
    mu_columns: list[str]
    sigma_columns: list[str]


def infer_submission_schema(
    *,
    sample_submission: pd.DataFrame | None = None,
    target_columns: list[str] | None = None,
    n_targets: int | None = None,
    id_column: str = "planet_id",
) -> SubmissionSchema:
    if sample_submission is not None:
        id_col = sample_submission.columns[0]
        value_columns = list(sample_submission.columns[1:])
        if len(value_columns) % 2 != 0:
            raise ValueError("sample_submission must contain equal mu and sigma column counts.")
        half = len(value_columns) // 2
        return SubmissionSchema(
            id_column=id_col,
            mu_columns=value_columns[:half],
            sigma_columns=value_columns[half:],
        )

    if target_columns is not None:
        mu_columns = list(target_columns)
        sigma_columns = [f"sigma_{idx + 1}" for idx in range(len(mu_columns))]
        return SubmissionSchema(id_column=id_column, mu_columns=mu_columns, sigma_columns=sigma_columns)

    if n_targets is None:
        raise ValueError("Provide sample_submission, target_columns, or n_targets.")
    mu_columns = [f"wl_{idx + 1}" for idx in range(n_targets)]
    sigma_columns = [f"sigma_{idx + 1}" for idx in range(n_targets)]
    return SubmissionSchema(id_column=id_column, mu_columns=mu_columns, sigma_columns=sigma_columns)


def build_submission_frame(
    planet_ids: np.ndarray | list[str],
    prediction: ModelPrediction,
    schema: SubmissionSchema,
) -> pd.DataFrame:
    mu = np.asarray(prediction.mu, dtype=float)
    sigma = np.maximum(np.asarray(prediction.sigma, dtype=float), 1e-12)
    if mu.shape != sigma.shape:
        raise ValueError("mu and sigma must have the same shape.")
    if mu.shape[1] != len(schema.mu_columns) or sigma.shape[1] != len(schema.sigma_columns):
        raise ValueError("Prediction target count does not match submission schema.")

    frame = pd.DataFrame({schema.id_column: list(planet_ids)})
    for idx, column in enumerate(schema.mu_columns):
        frame[column] = mu[:, idx]
    for idx, column in enumerate(schema.sigma_columns):
        frame[column] = sigma[:, idx]
    return frame


def predict_submission(
    model: TargetPCARegressor,
    features: pd.DataFrame,
    *,
    feature_columns: list[str],
    schema: SubmissionSchema,
) -> pd.DataFrame:
    x = features[feature_columns].fillna(0.0).to_numpy(dtype=float)
    prediction = model.predict(x)
    return build_submission_frame(features[schema.id_column].to_numpy(), prediction, schema)


def save_submission(frame: pd.DataFrame, output_path: Path | str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
