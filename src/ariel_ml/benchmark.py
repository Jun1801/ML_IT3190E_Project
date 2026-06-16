from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ariel_ml.config import ModelConfig
from ariel_ml.models import MODEL_FAMILIES, ModelFactory
from ariel_ml.training import cross_validate_model

# Reverse lookup: canonical model name -> family.
_MODEL_TO_FAMILY: dict[str, str] = {
    name: family for family, names in MODEL_FAMILIES.items() for name in names
}

# Metric columns surfaced in the comparison table, in display order.
REPORT_METRICS: tuple[str, ...] = (
    "gaussian_nll",
    "ariel_gll_score",
    "rmse_mean",
    "mae_mean",
    "coverage_1sigma",
    "coverage_2sigma",
    "sigma_mean",
)


def family_of(model_name: str) -> str:
    return _MODEL_TO_FAMILY.get(model_name.strip().lower().replace("-", "_"), "other")


@dataclass(frozen=True)
class BenchmarkRow:
    model_name: str
    family: str
    status: str  # "ok" | "skipped" | "error"
    metrics: dict[str, float] = field(default_factory=dict)
    train_seconds: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class BenchmarkResult:
    rows: list[BenchmarkRow]

    @property
    def ok_rows(self) -> list[BenchmarkRow]:
        return [row for row in self.rows if row.status == "ok"]

    def best(self, metric: str = "gaussian_nll", *, maximize: bool = False) -> BenchmarkRow | None:
        """Best row by ``metric``. Set ``maximize=True`` for higher-is-better
        metrics such as ``ariel_gll_score``."""
        candidates = [row for row in self.ok_rows if metric in row.metrics]
        if not candidates:
            return None
        chooser = max if maximize else min
        return chooser(candidates, key=lambda row: row.metrics[metric])

    def to_frame(self, metrics: tuple[str, ...] = REPORT_METRICS) -> pd.DataFrame:
        records = []
        for row in self.rows:
            record = {
                "family": row.family,
                "model": row.model_name,
                "status": row.status,
            }
            for metric in metrics:
                record[metric] = row.metrics.get(metric, np.nan)
            record["train_seconds"] = row.train_seconds
            record["message"] = row.message
            records.append(record)
        frame = pd.DataFrame.from_records(records)
        sort_metric = metrics[0] if metrics else "gaussian_nll"
        # Successful rows first (sorted by the primary metric), then the rest.
        frame["_ok"] = (frame["status"] != "ok").astype(int)
        frame = frame.sort_values(
            ["family", "_ok", sort_metric], na_position="last"
        ).drop(columns="_ok")
        return frame.reset_index(drop=True)


def benchmark_models(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_names: list[str] | None = None,
    model_config: ModelConfig | None = None,
    n_splits: int = 5,
    groups: np.ndarray | None = None,
    random_state: int = 42,
    sigma_cal_fraction: float = 0.0,
) -> BenchmarkResult:
    """Cross-validate every model on identical folds and collect comparison metrics.

    Models whose optional dependency is missing are reported with status
    ``"skipped"``; any other failure is captured as ``"error"`` so a single bad
    model never aborts the whole sweep. Folds are deterministic in
    ``random_state`` + ``groups``, so every model is scored on the same splits.
    """
    names = list(model_names) if model_names is not None else ModelFactory.list_models()
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    rows: list[BenchmarkRow] = []
    for name in names:
        family = family_of(name)
        start = time.perf_counter()
        try:
            cv_result = cross_validate_model(
                x_arr,
                y_arr,
                model_name=name,
                model_config=model_config,
                n_splits=n_splits,
                groups=groups,
                random_state=random_state,
                sigma_cal_fraction=sigma_cal_fraction,
            )
            elapsed = time.perf_counter() - start
            rows.append(
                BenchmarkRow(
                    model_name=name,
                    family=family,
                    status="ok",
                    metrics=cv_result.mean_metrics,
                    train_seconds=elapsed,
                )
            )
        except ImportError as exc:
            rows.append(
                BenchmarkRow(
                    model_name=name,
                    family=family,
                    status="skipped",
                    train_seconds=time.perf_counter() - start,
                    message=str(exc),
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad model must not abort the sweep
            rows.append(
                BenchmarkRow(
                    model_name=name,
                    family=family,
                    status="error",
                    train_seconds=time.perf_counter() - start,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
    return BenchmarkResult(rows=rows)
