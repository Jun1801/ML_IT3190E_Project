from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Ariel feature CSV from raw Kaggle parquet data.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--output", type=Path, default=Path("outputs/features_train.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--time-bins", type=int, default=300)
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ariel_ml.config import DatasetConfig, FeatureConfig, PreprocessConfig
    from ariel_ml.dataset_builder import ArielDatasetBuilder
    from ariel_ml.io import ArielDataRepository
    from ariel_ml.pipeline import ArielPreprocessFeaturePipeline

    repository = ArielDataRepository(DatasetConfig(data_root=args.data_root))
    pipeline = ArielPreprocessFeaturePipeline(
        PreprocessConfig(target_time_bins=args.time_bins),
        FeatureConfig(),
    )
    builder = ArielDatasetBuilder(repository=repository, pipeline=pipeline)
    result = builder.build_feature_csv(
        args.split,
        args.output,
        aggregate_observations=not args.no_aggregate,
        limit=args.limit,
        on_error="skip" if args.skip_errors else "raise",
    )
    print(f"Wrote {len(result.features)} rows and {len(result.features.columns)} columns to {args.output}")
    if result.failures:
        print(f"Skipped {len(result.failures)} failed observations")


if __name__ == "__main__":
    main()
