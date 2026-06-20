# Repository Guidelines

## Project Structure & Module Organization

This repository implements a full ML pipeline for the Ariel Data Challenge 2025.
All library modules live flat in `src/` — no sub-package, no `ariel_ml.` prefix.
`src/` is added to `sys.path` by every script and by pytest (`pythonpath = ["src"]`).

```text
src/                     flat module directory (each file is directly importable)
  config.py              configuration dataclasses (DatasetConfig, PreprocessConfig, FeatureConfig, ModelConfig, DeepModelConfig)
  preprocessing.py       detector calibration, light curve extraction, transit detection
  features.py            physics-based feature engineering
  pipeline.py            calibration → feature orchestration (ArielPreprocessFeaturePipeline)
  data_io.py             Kaggle parquet repository (ArielDataRepository, RawObservation)
  dataset_builder.py     data loading and feature DataFrame building
  sequence_dataset.py    [samples, time, channels] tensor builder for deep models
  metrics.py             RMSE, Gaussian NLL, Ariel GLL score, SigmaCalibrator
  models.py              TargetPCARegressor (base), ResidualCorrectedRegressor, MeanShiftedRegressor, WeightedEnsembleRegressor
  estimators.py          17 concrete *PCARegressor classes + MeanShiftedRegressor factory + ModelFactory + MODEL_FAMILIES
  deep_models.py         TorchSequenceRegressor + CNN1D/LSTM/GRU/TCN/Transformer/AutoencoderMLP
  training.py            CV, hyperparameter search, train/val split, evaluation
  benchmark.py           benchmark_models() — BenchmarkResult, BenchmarkRow, family_of()
  submission.py          submission generation
scripts/                 CLI entry points
  build_features.py      extract features from raw Kaggle data → CSV
  train.py               train model, optionally hyperparameter-search, refit on all data
  benchmark.py           benchmark all model families on shared CV folds
tests/                   pytest unit tests (synthetic data; no Kaggle data required)
notebooks/               exploratory notebooks (import from src/, not inline code)
  prepare_sequence.ipynb CPU: precompute light-curve tensors → precomputed/
  run_deep_learning.ipynb GPU: train/benchmark deep sequence models
plans/                   research notes and model rationale (Vietnamese)
benchmark/
  result.csv             committed benchmark comparison table
  benchmark.csv          aggregated benchmark results (all model families)
precomputed/             committed npz tensors for deep learning experiments
data/                    local Kaggle data — git-ignored
outputs/                 generated models, features, submissions — git-ignored
```

## Import Convention

All imports use the flat module name — never the old `ariel_ml.` prefix:

```python
from config import ModelConfig, DeepModelConfig
from models import TargetPCARegressor, ModelPrediction
from estimators import ModelFactory, MODEL_FAMILIES
from training import cross_validate_model
from data_io import ArielDataRepository      # NOT from io import (stdlib conflict)
```

## Build, Test, and Development Commands

```powershell
# Install
uv venv && uv pip install -e .
uv pip install -r requirements-optional.txt   # lightgbm, xgboost, torch

# Run all tests (synthetic data, no Kaggle data needed)
python -m pytest

# Build features from raw Kaggle data (smoke → full)
python scripts/build_features.py --data-root data --split train --output outputs/features_train.csv --limit 5
python scripts/build_features.py --data-root data --split train --output outputs/features_train.csv

# Train
python scripts/train.py --features outputs/features_train.csv --targets data/train.csv --model bayesian_ridge --n-components 30

# Benchmark all model families
python scripts/benchmark.py --features outputs/features_train.csv --targets data/train.csv --cv 5
```

## Code Organization Rules

- `models.py` = base classes ONLY: `TargetPCARegressor`, `ResidualCorrectedRegressor`, `MeanShiftedRegressor`, `WeightedEnsembleRegressor`, `ModelPrediction`.
- `estimators.py` = concrete estimators + factory: 17 `*PCARegressor` subclasses, `MeanShiftedRegressor`-based `ms_*` factory entries, `_resolve_device`, `MODEL_FAMILIES`, `ModelFactory`.
- `data_io.py` is named `data_io` (not `io`) to avoid shadowing Python's stdlib `io` module.
- Never add `src/ariel_ml/` back. If new modules are needed, add them flat to `src/`.

## Coding Style & Naming Conventions

Python 3.11+. PEP 8, 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
Import production code from notebooks rather than duplicating it.

## Testing Guidelines

Add focused `pytest` coverage for every non-trivial logic change. Use small synthetic arrays.
Test files follow `test_<module>.py` naming. Tests must not require Kaggle data.

## Commit & Pull Request Guidelines

Use short imperative commit subjects (`Add transit depth feature`, `Fix sigma calibration bug`).
PRs should include a concise summary, commands run to validate, and affected paths.

## Security & Configuration

Do not commit `.env`, raw challenge data, generated submissions, model checkpoints, or large output artifacts.
`data/` and `outputs/` are git-ignored. `features_train.csv` / `features_test.csv` at the repo root are also gitignored — put them in `outputs/` instead.

## Agent-Specific Instructions

- Keep changes narrow and tied to the requested task.
- For non-trivial code changes, add or update tests first, then implement the smallest passing change.
- Use `notebooks/` only for exploratory analysis; import production code from `src/`.
- Large binary artifacts (model weights, npz tensors) should go in `precomputed/` or `outputs/` — only commit small precomputed tensors needed for notebook reproducibility.
