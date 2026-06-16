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
    parser = argparse.ArgumentParser(
        description="Benchmark every model family on shared CV folds and emit a comparison table."
    )
    parser.add_argument("--features", type=Path, default=Path("outputs/features_train.csv"))
    parser.add_argument("--targets", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark.csv"))
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--n-components", type=int, default=30)
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names. Default: every canonical model across all families.",
    )
    parser.add_argument(
        "--sigma-cal-fraction",
        type=float,
        default=0.0,
        help="Fraction of each training fold reserved for sigma calibration (removes circular NLL).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-sigma-calibration", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import warnings

    import pandas as pd
    from sklearn.exceptions import ConvergenceWarning

    from ariel_ml.benchmark import benchmark_models
    from ariel_ml.config import ModelConfig
    from ariel_ml.dataset_builder import align_features_and_targets

    # The sweep deliberately fits many estimators (GPR, MLP, …) whose
    # convergence warnings would otherwise drown out the comparison table.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    features = pd.read_csv(args.features)
    targets = pd.read_csv(args.targets)
    x_frame, y, groups, _ = align_features_and_targets(features, targets)

    model_config = ModelConfig(
        n_components=args.n_components,
        random_state=args.random_state,
        calibrate_sigma=not args.no_sigma_calibration,
    )
    model_names = (
        [name.strip() for name in args.models.split(",") if name.strip()]
        if args.models
        else None
    )

    result = benchmark_models(
        x_frame.to_numpy(dtype=float),
        y,
        model_names=model_names,
        model_config=model_config,
        n_splits=args.cv,
        groups=groups,
        random_state=args.random_state,
        sigma_cal_fraction=args.sigma_cal_fraction,
    )

    frame = result.to_frame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(frame.to_string(index=False))
    best_nll = result.best("gaussian_nll")
    if best_nll is not None:
        print(f"\nBest by gaussian_nll (lower=better): {best_nll.model_name} ({best_nll.family}) = {best_nll.metrics['gaussian_nll']:.6f}")
    best_gll = result.best("ariel_gll_score", maximize=True)
    if best_gll is not None:
        print(f"Best by ariel_gll_score (higher=better): {best_gll.model_name} ({best_gll.family}) = {best_gll.metrics['ariel_gll_score']:.6f}")
    print(f"\nSaved table -> {args.output}")


if __name__ == "__main__":
    main()
