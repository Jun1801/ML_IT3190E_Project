# Ariel Notebook Config & Run Guide

File này mô tả cách set config khi chạy notebook hoặc CLI cho pipeline Ariel 2025. Ưu tiên chạy theo tầng: smoke nhỏ, debug vừa, baseline nghiêm túc, rồi full train.

## 1. Đường dẫn chuẩn

Trên local:

```python
from pathlib import Path

PROJECT_ROOT = Path(".")
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURE_DIR = OUTPUT_DIR / "features"
MODEL_DIR = OUTPUT_DIR / "models"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
```

Trên Kaggle, nếu repo nằm trong `/kaggle/working/ML_IT3190E_Project`:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path("/kaggle/working/ML_IT3190E_Project")
DATA_ROOT = Path("/kaggle/input/ariel-data-challenge-2025")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURE_DIR = OUTPUT_DIR / "features"
MODEL_DIR = OUTPUT_DIR / "models"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
```

## 2. Config build features

`time_bins` là số điểm thời gian sau khi downsample raw signal. Nó không phải số planet và không phải số PCA components.

```python
from ariel_ml.config import DatasetConfig, FeatureConfig, PreprocessConfig
from ariel_ml.dataset_builder import ArielDatasetBuilder
from ariel_ml.io import ArielDataRepository
from ariel_ml.pipeline import ArielPreprocessFeaturePipeline

dataset_config = DatasetConfig(data_root=DATA_ROOT)
preprocess_config = PreprocessConfig(
    target_time_bins=64,              # 20 smoke, 64 debug, 128 baseline, 256+ nếu đủ RAM
    bin_mode="mean",
    bin_before_spatial_calibration=True,
    apply_cds=True,
    apply_linearity=False,            # để False trước vì nhanh hơn trên smoke/CPU
    smooth_window=5,
    detrend_degree=2,
)
feature_config = FeatureConfig(
    spectral_bin_sizes=(1, 2, 4, 8, 16, 32, 64),
    include_per_wavelength_depths=True,
    include_per_wavelength_noise=False,
)

repository = ArielDataRepository(dataset_config)
pipeline = ArielPreprocessFeaturePipeline(preprocess_config, feature_config)
builder = ArielDatasetBuilder(repository=repository, pipeline=pipeline)

train_result = builder.build_feature_csv(
    split="train",
    output_path=OUTPUT_DIR / "features_train_64.csv",
    limit=50,                         # None để chạy full
    aggregate_observations=True,
    on_error="raise",                 # đổi thành "skip" nếu muốn bỏ planet lỗi
)
```

## 3. Mức chạy khuyến nghị

| Mức | `limit` | `time_bins` | Mục tiêu |
|---|---:|---:|---|
| Smoke | 5 | 20 | Kiểm tra đọc data thật, build feature, train, save model |
| Debug | 50 | 64 | Bắt lỗi shape, planet lạ, feature NaN/Inf |
| Baseline | 200-500 | 128 | So sánh model ổn định |
| Full | None | 128 hoặc 256 | Train model cuối |

Không nên bắt đầu bằng full data. Nếu smoke chỉ có `5 rows` thì đó là do `limit=5`, nghĩa là 5 planet sau khi aggregate observation.

## 4. Config model

`n_components` là số chiều PCA của target spectrum 283 chiều, không phải số planet. Nó bị giới hạn bởi số sample train, nên nếu chỉ có 5 planet thì PCA thực tế tối đa cũng chỉ 5.

```python
from ariel_ml.config import ModelConfig

model_config = ModelConfig(
    n_components=30,                  # 10-20 cho ít data, 30-50 cho baseline/full
    standardize_features=True,
    calibrate_sigma=True,
    residual_floor=1e-8,
    sigma_floor=1e-8,
    random_state=42,
)
```

Thứ tự model nên chạy:

1. `ridge`: rất nhanh, dùng để kiểm tra pipeline.
2. `bayesian_ridge`: baseline chính theo plan, có uncertainty tốt hơn.
3. `kernel_ridge`: thử khi feature đã ổn, có thể chậm hơn.
4. `extra_trees`: bắt phi tuyến, chạy nặng hơn.
5. `boosting`: baseline boosting dùng sklearn, không cần package ngoài.
6. `lightgbm` hoặc `xgboost`: chỉ chạy khi đã cài optional dependency.
7. `bayesian_ridge_boosting_residual`: residual correction không cần LightGBM.
8. `bayesian_ridge_lgbm_residual`: residual correction mạnh hơn, cần LightGBM.

## 5. Training trong notebook

```python
import joblib
import json
import pandas as pd

from ariel_ml.dataset_builder import align_features_and_targets
from ariel_ml.training import train_model, refit_full_model

features = pd.read_csv(OUTPUT_DIR / "features_train_64.csv")
targets = pd.read_csv(DATA_ROOT / "train.csv")

x_frame, y, groups, target_columns = align_features_and_targets(features, targets)

result = train_model(
    x_frame.to_numpy(dtype=float),
    y,
    model_name="bayesian_ridge",
    model_config=model_config,
    validation_fraction=0.2,
    groups=groups,
    random_state=42,
)

final_model = refit_full_model(
    x_frame.to_numpy(dtype=float),
    y,
    model_name="bayesian_ridge",
    model_config=model_config,
)

artifact = {
    "model": final_model,
    "validation_model": result.model,
    "feature_columns": list(x_frame.columns),
    "target_columns": target_columns,
    "metrics": result.evaluation.as_dict(),
    "model_name": "bayesian_ridge",
    "model_config": model_config,
}

model_path = MODEL_DIR / "bayesian_ridge_model.joblib"
joblib.dump(artifact, model_path)
(MODEL_DIR / "bayesian_ridge_metrics.json").write_text(
    json.dumps(result.evaluation.as_dict(), indent=2),
    encoding="utf-8",
)
```

## 6. CLI tương đương

Smoke:

```bash
python scripts/build_features.py --data-root data --split train --output outputs/features_train_5.csv --limit 5 --time-bins 20
python scripts/train.py --features outputs/features_train_5.csv --targets data/train.csv --model ridge --n-components 5 --output-dir outputs/models/ridge_smoke
```

Debug:

```bash
python scripts/build_features.py --data-root data --split train --output outputs/features_train_50.csv --limit 50 --time-bins 64
python scripts/train.py --features outputs/features_train_50.csv --targets data/train.csv --model bayesian_ridge --n-components 20 --output-dir outputs/models/br_50
```

Baseline:

```bash
python scripts/build_features.py --data-root data --split train --output outputs/features_train_200.csv --limit 200 --time-bins 128
python scripts/train.py --features outputs/features_train_200.csv --targets data/train.csv --model bayesian_ridge --n-components 30 --cv 5 --output-dir outputs/models/br_200_cv
python scripts/train.py --features outputs/features_train_200.csv --targets data/train.csv --model bayesian_ridge --n-components 30 --output-dir outputs/models/br_200
```

Search nhỏ:

```bash
python scripts/train.py --features outputs/features_train_200.csv --targets data/train.csv --search --model-candidates bayesian_ridge,ridge,kernel_ridge,extra_trees,boosting --n-components-grid 20,30,40 --output-dir outputs/models/search_200
```

Full:

```bash
python scripts/build_features.py --data-root data --split train --output outputs/features_train_full.csv --time-bins 128
python scripts/train.py --features outputs/features_train_full.csv --targets data/train.csv --model bayesian_ridge --n-components 40 --output-dir outputs/models/br_full
```

## 7. Quy tắc chọn `n_components`

| Số planet train | Gợi ý `n_components` |
|---:|---:|
| 5-20 | 3-10 |
| 50 | 10-20 |
| 200 | 20-40 |
| Full train | 30-80 |

Nếu sample ít mà đặt `n_components=30`, code sẽ tự giảm xuống theo giới hạn PCA. Tuy vậy nên đặt nhỏ đúng với quy mô smoke để metric dễ hiểu hơn.

## 8. Submission flow

Sau khi có model ổn, build feature cho test rồi tạo submission bằng artifact đã save.

```bash
python scripts/build_features.py --data-root data --split test --output outputs/features_test.csv --time-bins 128
```

Trong notebook hoặc script submission, luôn dùng đúng `feature_columns` trong artifact để align test features trước khi predict.
