# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Ariel ML** is a machine learning pipeline for the Ariel Data Challenge 2025, a multi-target probabilistic regression task predicting atmospheric spectra (283 wavelength values) from noisy telescope detector data, including uncertainty estimates.

Core approach: **Bayesian Ridge Regression + Physics-based Feature Engineering + Target PCA + Sigma Calibration**

## Repository Structure

```
src/                       # All library modules (importable directly via sys.path; no sub-package)
  config.py                # Configuration dataclasses (DatasetConfig, PreprocessConfig, FeatureConfig, ModelConfig, DeepModelConfig)
  preprocessing.py         # Detector calibration, light curve extraction, transit detection
  features.py              # Physics-based feature engineering
  pipeline.py              # End-to-end calibration → feature orchestration
  data_io.py               # Kaggle parquet data repository (ArielDataRepository, RawObservation)
  dataset_builder.py       # Data loading and feature building from raw observations
  sequence_dataset.py      # Build [samples, time, channels] light-curve tensors for deep models
  metrics.py               # Evaluation metrics (RMSE, Gaussian NLL, official Ariel GLL score, sigma calibration)
  models.py                # TargetPCARegressor base class + ResidualCorrectedRegressor + MeanShiftedRegressor + WeightedEnsembleRegressor
  estimators.py            # 17 concrete *PCARegressor subclasses + MeanShiftedRegressor factory + ModelFactory + MODEL_FAMILIES
  deep_models.py           # Deep learning baselines (CNN1D, LSTM, GRU, TCN, Transformer, Autoencoder)
  training.py              # Training pipeline with CV, hyperparameter search, evaluation
  benchmark.py             # benchmark_models() — BenchmarkResult, BenchmarkRow, family_of()
  submission.py            # Submission generation
scripts/
  build_features.py        # CLI: Extract features from raw data
  train.py                 # CLI: Train model and hyperparameter search
  benchmark.py             # CLI: Benchmark all model families
  download_sample_data.ps1 # PowerShell helper for data download
tests/                     # pytest unit tests (uses synthetic data, no Kaggle data required)
notebooks/
  prepare_sequence.ipynb   # CPU: precompute light-curve tensors → precomputed/
  run_deep_learning.ipynb  # GPU: train/benchmark deep sequence models on precomputed tensors
plans/                     # Research notes and model rationale (Vietnamese)
  ariel_2025_model_plan.md # Detailed pipeline design with formulas and experiments
benchmark/
  result.csv               # Committed benchmark results table
  benchmark.csv            # Aggregated benchmark results (all model families)
```

## Setup & Dependencies

**Python Version:** 3.11+

**Core Dependencies:**
- joblib, numpy, pandas, pyarrow, scikit-learn

**Optional (full model set):**
- lightgbm, torch, xgboost

**Installation:**

```powershell
# With uv (recommended)
uv venv
uv pip install -e .

# Or standard pip
python -m pip install -e .
python -m pip install -r requirements-optional.txt  # For all models
```

## Common Commands

### Run Tests
```powershell
python -m pytest                    # All tests
python -m pytest tests/test_models.py::ModelTests::test_bayesian_ridge_pca_fit_predict_shapes_and_positive_sigma  # Single test
python -m pytest -v                 # Verbose output
```

### Build Features (from raw Kaggle data)
```powershell
# Smoke test on 5 planets
python scripts/build_features.py --data-root data --split train --output outputs/features_train.csv --limit 5

# Full train feature extraction
python scripts/build_features.py --data-root data --split train --output outputs/features_train.csv

# Build test features
python scripts/build_features.py --data-root data --split test --output outputs/features_test.csv

# Custom time binning (default 300)
python scripts/build_features.py --data-root data --split train --output outputs/features_train.csv --time-bins 500
```

### Train Models
```powershell
# Simple training with specified model
python scripts/train.py --features outputs/features_train.csv --targets data/train.csv --output-dir outputs/model --model bayesian_ridge --n-components 30

# Run 5-fold grouped cross-validation (respects planet groupings)
python scripts/train.py --features outputs/features_train.csv --targets data/train.csv --cv 5

# Hyperparameter search over models and PCA components, then refit on all data
python scripts/train.py --features outputs/features_train.csv --targets data/train.csv --search --cv 5 `
  --model-candidates "bayesian_ridge,ridge,kernel_ridge,extra_trees,boosting,lightgbm" `
  --n-components-grid "20,30,40"

# Available models (grouped into families via ModelFactory.families()):
#   linear:     ridge, lasso, elastic_net
#   bayesian:   bayesian_ridge, ard, gaussian_process, ngboost
#   kernel_svm: svr, kernel_ridge
#   neighbors:  knn
#   trees:      random_forest, extra_trees, boosting, hist_gradient_boosting, lightgbm, xgboost
#   neural:     mlp
#   hybrid:     br_lgbm_residual, br_boosting_residual
#   mean_shift: ms_bayesian_ridge, ms_ridge, ms_extra_trees  (MeanShiftedRegressor wrapper)
```

### Benchmark All Model Families
```powershell
# Cross-validate every model on identical folds and emit a comparison table (CSV + stdout).
# Models with a missing optional dep (lightgbm/xgboost/ngboost) are reported as "skipped", not fatal.
python scripts/benchmark.py --features outputs/features_train.csv --targets data/train.csv --cv 5 --output outputs/benchmark.csv

# Restrict to a subset of models
python scripts/benchmark.py --features outputs/features_train.csv --targets data/train.csv --models "bayesian_ridge,svr,random_forest,gaussian_process"
```

## Architecture & Data Flow

### High-Level Pipeline

**Data Input:**
- Raw AIRS detector signals (time × spatial × wavelength)
- FGS1 white-light signals (time × spatial)
- Calibration bundles (dead pixels, dark frames, flat fields)
- ADC info (gain/offset per instrument)
- Star metadata (radius, mass, temperature, logg)
- Train targets (planet_id → spectrum of 283 values)

**Stage 1: Detector Calibration** (`DetectorCalibrator`)
1. ADC correction: `(raw - offset) × gain`
2. Bad pixel masking (dead pixel replacement via median)
3. Dark current subtraction
4. Flat-field correction
5. Correlated double sampling (CDS) to reduce read noise
6. Temporal binning (default 300 bins)

**Stage 2: Light Curve Extraction** (`LightCurveExtractor`)
- AIRS: Sum spatial dimensions → `[time, wavelength]` light curves
- FGS: Sum all spatial dimensions → `[time]` white-light curve
- Output: `LightCurves` dataclass with airs, fgs, airs_white

**Stage 3: Transit Boundary Detection** (`TransitBoundaryDetector`)
- Uses FGS white-light curve (higher SNR)
- Detects ingress, in-transit, egress regions via derivative analysis
- Outputs `TransitBounds` (start, ingress_end, egress_start, end indices)

**Stage 4: Normalization & Detrending** (`LightCurveTransformer`)
- Normalize by out-of-transit median
- Polynomial detrending (default degree 2)
- Light smoothing (default window 5)

**Stage 5: Feature Engineering** (`ArielFeatureBuilder`)
Physics-based features extracted per observation:
- **Depth features:** Transit depth per wavelength, mean/median/percentile variants
- **Multi-scale spectral:** Binned wavelengths (sizes 1, 2, 4, 8, 16, 32, 64) with local statistics
- **Shape features:** Ingress/egress slopes, duration, symmetry, baseline drift (from FGS + white curve)
- **Noise features:** Out-of-transit & in-transit std, SNR, calibration metrics
- **Stellar features:** Star radius, mass, temperature, logg (with log and interaction terms)

Output: `dict[str, float]` of ~50–150 features per observation

**Stage 6: Model Training** (`TargetPCARegressor` wrappers)
1. Feature standardization (StandardScaler)
2. Target PCA (default 30 components, preserves spectrum smoothness)
3. Train regressor per PCA component (e.g., Bayesian Ridge)
4. Predict PCA coefficients with uncertainty
5. Inverse PCA transform to 283-wavelength spectrum
6. Combine Bayesian Ridge uncertainty + validation residual RMSE
7. Calibrate sigma scale on validation Gaussian NLL

**Stage 7: Submission**
- Outputs predictions + calibrated uncertainties
- Format: CSV with planet_id, wavelength_001 to wavelength_283, and sigma_001 to sigma_283

### Key Classes & Design Patterns

**Configuration Objects** (`config.py`):
- `DatasetConfig`: Data root paths and file names for Kaggle data layout
- `PreprocessConfig`: Detector calibration & light curve normalization parameters
- `FeatureConfig`: Feature extraction options
- `ModelConfig`: PCA components, standardization, sigma floor, random state
- `DeepModelConfig`: Epochs, batch size, learning rate, device

**Core Abstractions:**
- `ArielPreprocessFeaturePipeline` (`pipeline.py`): Orchestrates calibration → light curves → features
- `ArielDatasetBuilder` (`dataset_builder.py`): Loads raw observations, builds feature DataFrames
- `TargetPCARegressor` (`models.py`): Base class for all tabular models — fit/predict/sigma pipeline
- `MeanShiftedRegressor` (`models.py`): Decomposes spectrum into per-planet mean depth + wavelength shape, models each separately
- `ModelFactory` (`estimators.py`): Factory for creating model instances by name; 17 concrete *PCARegressor classes + MeanShiftedRegressor-based `ms_*` variants

**Data Structures:**
- `CalibrationBundle`: dead, dark, flat, read, linear_corr arrays
- `CalibrationMetrics`: Extracted QA metrics for features
- `LightCurves`: AIRS (time × wavelength) + FGS (time) + white (time)
- `TransitBounds`: Transit region indices and convenience masks/slices
- `ModelPrediction`: (mu: mean spectrum, sigma: uncertainty)
- `EvaluationResult`: RMSE, MAE, Gaussian NLL, official Ariel GLL score, coverage metrics
- Official metric (`ariel_gll_score`): normalized GLL `(GLL_pred - GLL_ref)/(GLL_ideal - GLL_ref)` clipped to [0,1]; ideal sigma = 10 ppm (1e-5), reference = naive train mean/std. Higher is better (opposite direction to gaussian_nll).
- `TrainResult`: Model + prediction + evaluation + train/val indices

**Training Utilities** (`training.py`):
- `train_model`: Single train/val split (delegates to `train_model_on_indices`)
- `train_model_on_indices`: Core train/eval step on explicit index arrays (supports sigma_cal_fraction)
- `cross_validate_model`: K-fold CV with GroupKFold (respects planet groups)
- `hyperparameter_search`: Grid search over models × n_components × model_params
- `search_n_components`: Fast exact n_components sweep for `TargetPCARegressor` — fits once at max k, reuses nested PCA prefix
- `refit_full_model`: Refit best candidate on all data
- `build_gll_weighted_ensemble`: PHC step 3 — scores each family on a held-out split, returns `WeightedEnsembleRegressor` with `softmax(score/temperature)` weights
- `make_train_validation_split` / `make_cv_splitter`: Splitters respecting GroupKFold when planet groups are present
- `feature_dicts_to_frame` / `targets_to_matrix`: Helpers to convert feature dicts and target DataFrames to numpy arrays
- `CrossValidationResult`: Holds `fold_results: list[TrainResult]`; `.mean_metrics` averages over folds
- `SearchCandidateResult` / `HyperparameterSearchResult`: Structured output from grid search with `best_candidate`
- `EnsembleBuildResult`: Output of `build_gll_weighted_ensemble` — `ensemble`, `model_names`, `weights`, `val_scores`

## Model Comparison Strategy

**Primary Model:** Bayesian Ridge + Target PCA
- Rationale: Probabilistic, fits small datasets, provides natural uncertainty, interpretable

**Comparative ML Models:**
1. **Ridge:** Linear baseline, no uncertainty
2. **Kernel Ridge:** Nonlinear via RBF/polynomial kernels
3. **ExtraTrees:** Tree-based, robust to outliers, feature importance
4. **LightGBM/XGBoost:** Gradient boosting, strong on tabular features
5. **ResidualCorrectedRegressor:** Bayesian Ridge + LightGBM residual correction (weighted blend)

**Deep Learning Baselines** (`deep_models.py`):
- **CNN1D:** 1D convolutions on light curve time dimension
- **LSTM/GRU:** Sequence models
- **TCN:** Temporal convolutional network with dilated convolutions
- **Transformer:** Self-attention on time steps
- **Autoencoder + MLP:** Learn latent representation then regress

## Key Implementation Details

### Handling Uncertain Data
- Validation sets used to estimate residual RMSE per wavelength
- Sigma = sqrt(Bayesian_var + residual_RMSE²)
- `SigmaCalibrator` optimizes scale factor on Gaussian NLL
- **PHC — Physics-conditioned Heteroscedastic Calibration (key proposed method), 3 steps:**
  - **Step 1 — per-wavelength scale** (`ModelConfig.sigma_per_target=True` / `--sigma-per-target`): fits an independent `s_j` per wavelength. GLL is additive over elements and `s_j` only affects column `j`, so the GLL-optimal scale has the closed form `s_j = sqrt(mean_i (residual_ij/sigma_ij)²)` (RMS of normalised residuals) — maximises the official Ariel GLL without iteration.
  - **Step 2 — feature-conditioned multiplier** (`ModelConfig.sigma_feature_conditioned=True` / `--sigma-feature-conditioned`, class `FeatureConditionedSigmaCalibrator`): `σ_ij = s_j · m_i · σ_ij`. The per-row GLL-optimal multiplier `t_i = sqrt(mean_j (residual_ij/(s_j σ_ij))²)` also has a closed form; we regress `log t_i` on the (standardised) features via ridge, so noisier observations get wider intervals.
  - **Step 3 — GLL-weighted family mixture** (`training.build_gll_weighted_ensemble`): fits one model per family, scores each on a held-out split with `ariel_gll_score`, sets mixture weights `= softmax(score/temperature)`, and combines via `WeightedEnsembleRegressor` (Gaussian mixture of means + second moments).
  - Default stays the legacy global scalar calibration.

### Cross-Validation
- Uses `GroupKFold` to prevent leakage (same planet in train/val)
- Groups are inferred from planet_id in target CSV

### Feature Alignment
- `align_features_and_targets()` matches feature rows to targets by planet_id
- Handles missing planets or observations gracefully
- Returns X, y, groups, target_column_names

### Joblib Serialization
- Models saved with `joblib.dump(artifact_dict, path)` → restores with `joblib.load(path)`
- Fitted sklearn components (scaler, pca, regressors) fully serialized
- Lambdas in factories not pickled; only fitted models stored

## Testing Strategy

Tests use **synthetic data only** — no Kaggle dataset required.

- `test_preprocessing_features.py`: Detector calibration, light curve extraction, detrending
- `test_models.py`: Model fitting, prediction shapes, sigma positivity, joblib serialization
- `test_training.py`: CV, hyperparameter search, train/val split
- `test_dataset_builder_submission.py`: Feature alignment, submission format
- `test_deep_models.py`: Deep learning model instantiation
- `test_real_data_layout.py`: Expected Kaggle parquet directory structure

Run single test: `pytest tests/test_models.py::ModelTests::test_bayesian_ridge_pca_fit_predict_shapes_and_positive_sigma`

## Expected Data Layout

When raw Kaggle data is available, structure should be:
```
data/
  train.csv                    # planet_id, wavelength_001 to wavelength_283
  train_star_info.csv          # planet_id, Rs, Ms, Ts, logg, period, etc.
  test_star_info.csv
  adc_info.csv                 # instrument, planet_id, gain, offset
  wavelengths.csv              # wavelength values for each channel
  sample_submission.csv
  train/
    <planet_id>/
      AIRS-CH0_signal_0.parquet       # shape [time, height, width, channels]
      AIRS-CH0_calibration_0/         # directory with dead.parquet, dark.parquet, flat.parquet, etc.
      FGS1_signal_0.parquet
      FGS1_calibration_0/
  test/
    <planet_id>/
      (same structure)
```

`data/` is intentionally git-ignored.

## Configuration & Customization

All preprocessing, feature, and model hyperparameters are configurable via dataclasses in `config.py`:

```python
# All modules are importable directly — no package prefix needed.
# Scripts add src/ to sys.path; pytest uses pythonpath = ["src"].
from config import PreprocessConfig, FeatureConfig, ModelConfig
from pipeline import ArielPreprocessFeaturePipeline

# Custom preprocessing
preprocess = PreprocessConfig(
    target_time_bins=500,        # More fine-grained time sampling
    detrend_degree=3,            # Stronger detrending
    smooth_window=7,
)

# Custom features
features = FeatureConfig(
    spectral_bin_sizes=(1, 2, 4, 8, 16, 32),  # Skip size 64
    include_per_wavelength_noise=True,
)

# Custom model
model_cfg = ModelConfig(
    n_components=40,
    calibrate_sigma=True,
)

pipeline = ArielPreprocessFeaturePipeline(preprocess, features)
```

## Research Documentation

**Main reference:** `plans/ariel_2025_model_plan.md` (1279 lines, Vietnamese)
- Detailed rationale for each pipeline stage
- Mathematical formulas for calibration, detrending, feature extraction
- Comparative model descriptions with pros/cons
- Proposed experiment plan (Exp 0–6)
- Pseudocode for feature extraction and Bayesian Ridge training

**Key insights:**
- Top Kaggle solutions rely on physics-based signal processing, not just ML
- Transit depth is the single most informative feature
- Multi-scale spectral features exploit wavelength continuity
- Bayesian Ridge + uncertainty calibration aligns well with metric (Gaussian NLL)

## Repository Guidelines

From `AGENTS.md`:
- Keep all library modules in `src/` (flat, no sub-package); `src/` is on sys.path
- `models.py` = base classes only (`TargetPCARegressor`, `ResidualCorrectedRegressor`, `MeanShiftedRegressor`, `WeightedEnsembleRegressor`, `ModelPrediction`)
- `estimators.py` = 17 concrete `*PCARegressor` subclasses + `MeanShiftedRegressor`-based `ms_*` factory entries + `ModelFactory` + `MODEL_FAMILIES`
- Use `tests/` for deterministic unit tests (synthetic data preferred)
- Use `notebooks/` only for exploratory analysis; import production code from `src/`
- Keep large artifacts (models, submissions, raw data) in `outputs/` and `data/` (both git-ignored)
- Do not commit `.env`, model checkpoints, or challenge data
- Follow PEP 8, use `snake_case` for functions/variables, `PascalCase` for classes
- Add focused `pytest` coverage for non-trivial logic changes

### Coding Approach (`.agents/rules/coding_rule.md`)
- **TDD for non-trivial changes:** Write a failing test first (RED), make the smallest change to pass (GREEN), then refactor.
- **Surgical edits:** Touch only what the task requires. Don't improve adjacent unrelated code.
- **Simplicity:** No features, abstractions, or error handling beyond what was asked. If 200 lines could be 50, rewrite.

### Shadow File Technique (`.agents/rules/shadow_file.md`)
For large or multi-location file edits, use the shadow file technique to avoid indentation errors and context loss:
1. Create `filename.ext.shadow` as a new file (do not copy the original).
2. Write the entire final state of the file to it in chunks if needed.
3. Verify the shadow file is complete and syntactically correct.
4. Delete the original and rename the shadow file to the original name.

### Package Management
Always use `uv` — never bare `pip install`:
```powershell
uv pip install <package>
uv pip install -e .
```

## Notable Limitations

1. Raw Kaggle parquet shapes have not been validated in this workspace (data/ not present)
2. LightGBM/XGBoost adapters require optional dependencies
3. Transit detector is a baseline heuristic; real light curves should be visually inspected
4. Deep learning models likely overfit on small planet count; use as comparison only
5. Sigma calibration assumes validation set is representative of test set distribution
