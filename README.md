# Ariel ML

Baseline preprocessing, feature engineering, training, and submission pipeline for the Ariel Data Challenge 2025.

The current approach follows `plans/ariel_2025_model_plan.md`: detector calibration, light-curve extraction, transit boundary detection, physics-based features, Target PCA, Bayesian Ridge, and calibrated uncertainty.

## Repository Layout

```text
plans/          Research notes and data description
src/ariel_ml/   Reusable package code
scripts/        CLI entry points for features and training
notebooks/      End-to-end runnable notebook
tests/          Synthetic unit tests
```

Expected local Kaggle data layout:

```text
data/
  train.csv
  train_star_info.csv
  test_star_info.csv
  adc_info.csv
  wavelengths.csv
  sample_submission.csv
  train/<planet_id>/...
  test/<planet_id>/...
```

`data/` is intentionally ignored by Git.

## Setup

Use a virtual environment. With `uv`:

```powershell
uv venv
uv pip install -e .
```

If using an existing Python environment:

```powershell
python -m pip install -e .
```

Optional libraries for the full comparative model set:

```powershell
python -m pip install -r requirements-optional.txt
```

## Run Tests

```powershell
python -m pytest
```

The tests use synthetic arrays and do not require Kaggle data.

## Build Features

Smoke test on a few planets first:

```powershell
python scripts\build_features.py --data-root data --split train --output outputs\features_train.csv --limit 5
```

Full train feature build:

```powershell
python scripts\build_features.py --data-root data --split train --output outputs\features_train.csv
```

Build test features:

```powershell
python scripts\build_features.py --data-root data --split test --output outputs\features_test.csv
```

## Train

```powershell
python scripts\train.py --features outputs\features_train.csv --targets data\train.csv --output-dir outputs\model --model bayesian_ridge --n-components 30
```

Run grouped cross-validation:

```powershell
python scripts\train.py --features outputs\features_train.csv --targets data\train.csv --cv 5
```

Run model/PCA search, then refit the best candidate on all available training rows:

```powershell
python scripts\train.py --features outputs\features_train.csv --targets data\train.csv --search --cv 5
```

Available tabular model names include `bayesian_ridge`, `ridge`, `kernel_ridge`, `extra_trees`, `boosting`, `lightgbm`, `xgboost`, and `br_lgbm_residual`. `lightgbm`, `xgboost`, and deep learning baselines require optional dependencies.

## Notebooks

- [notebooks/prepare_sequence.ipynb](notebooks/prepare_sequence.ipynb) — precompute light-curve sequence tensors (CPU, run once, commit to `precomputed/`)
- [notebooks/run_deep_learning.ipynb](notebooks/run_deep_learning.ipynb) — train and benchmark deep sequence models (GPU recommended) on the precomputed tensors

For the tabular pipeline (feature extraction → Bayesian Ridge → submission), use the CLI scripts above.

## Current Limitations

- Raw Kaggle parquet shapes have not been validated in this workspace because `data/` is not present.
- LightGBM/XGBoost adapters are optional and require installing `requirements-optional.txt`.
- The transit detector is a baseline heuristic and should be inspected with plots on real light curves.
