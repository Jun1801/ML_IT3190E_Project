"""
Fair model benchmark for Ariel 2025.

Mỗi model được đánh giá bằng GroupKFold CV với sigma_cal_fraction:
  - Model được fit trên phần training của fold.
  - Sigma calibration (residual_rmse + sigma_calibrator) được fit trên một
    tập sigma_cal tách riêng từ training fold — KHÔNG dùng eval fold.
  - NLL và các metric khác được tính trên eval fold (chưa từng dùng trong training).

Việc tách sigma_cal và eval fold loại bỏ circular evaluation:
  trước đây sigma được fit trực tiếp trên eval fold → NLL trông tốt hơn thực tế
  (đặc biệt với ExtraTrees vì sigma của nó hoàn toàn phụ thuộc residual_rmse_val).

Usage:
    python benchmark/run_benchmark.py \\
        --features outputs/features_train.csv \\
        --targets data/train.csv \\
        --models bayesian_ridge,ridge,extra_trees,boosting,lightgbm,xgboost \\
        --n-components 30 --n-splits 5 --sigma-cal-fraction 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_MODELS = [
    "bayesian_ridge",
    "ridge",
    "kernel_ridge",
    "extra_trees",
    "boosting",
    "br_boosting_residual",
]

OPTIONAL_MODELS = ["lightgbm", "xgboost", "br_lgbm_residual"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fair model comparison via GroupKFold CV with separated sigma calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/features_train.csv"),
        help="Path to feature CSV built by build_features.py.",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("data/train.csv"),
        help="Path to train.csv with planet_id and wavelength targets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/result.csv"),
        help="Where to write the comparison table.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of model names to benchmark.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help=f"Also benchmark {OPTIONAL_MODELS} (require optional dependencies).",
    )
    parser.add_argument("--n-components", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--sigma-cal-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of each training fold used for sigma calibration. "
            "Must be in (0, 1). Set to 0 to use the old (circular) behaviour."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-calibrate-sigma",
        action="store_true",
        help="Disable SigmaCalibrator (for ablation).",
    )
    return parser.parse_args()


def _col(value: float | None, precision: int = 6) -> str:
    return "" if value is None else f"{value:.{precision}f}"


def main() -> None:
    args = parse_args()

    import pandas as pd

    from ariel_ml.config import ModelConfig
    from ariel_ml.dataset_builder import align_features_and_targets
    from ariel_ml.training import cross_validate_model

    if not args.features.exists():
        print(f"[ERROR] Feature file not found: {args.features}", file=sys.stderr)
        sys.exit(1)
    if not args.targets.exists():
        print(f"[ERROR] Targets file not found: {args.targets}", file=sys.stderr)
        sys.exit(1)

    features = pd.read_csv(args.features)
    targets = pd.read_csv(args.targets)
    x_frame, y, groups, _ = align_features_and_targets(features, targets)
    x = x_frame.to_numpy(dtype=float)
    n_planets = len(set(groups))
    print(
        f"Dataset: {x.shape[0]} rows × {x.shape[1]} features, "
        f"{n_planets} unique planets, {y.shape[1]} targets"
    )

    config = ModelConfig(
        n_components=args.n_components,
        calibrate_sigma=not args.no_calibrate_sigma,
        random_state=args.random_state,
    )

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.include_optional:
        model_names += [m for m in OPTIONAL_MODELS if m not in model_names]

    if args.sigma_cal_fraction > 0:
        print(
            f"sigma_cal_fraction={args.sigma_cal_fraction}: sigma calibrated on "
            f"{args.sigma_cal_fraction:.0%} of training fold (not eval fold)."
        )
    else:
        print("sigma_cal_fraction=0: sigma calibrated on eval fold (circular, legacy mode).")

    print(f"CV: {args.n_splits}-fold GroupKFold, n_components={args.n_components}\n")

    rows: list[dict] = []
    metric_keys = [
        "rmse_mean", "mae_mean", "gaussian_nll",
        "sigma_mean", "sigma_min", "sigma_max",
        "coverage_1sigma", "coverage_2sigma",
    ]

    for model_name in model_names:
        print(f"  [{model_name}]", end=" ", flush=True)
        t0 = time.time()
        try:
            cv = cross_validate_model(
                x,
                y,
                model_name=model_name,
                model_config=config,
                n_splits=args.n_splits,
                groups=groups,
                random_state=args.random_state,
                sigma_cal_fraction=args.sigma_cal_fraction,
            )
            m = cv.mean_metrics
            elapsed = time.time() - t0
            print(
                f"NLL={m['gaussian_nll']:.4f}  RMSE={m['rmse_mean']:.5f}"
                f"  σ={m['sigma_mean']:.5f}  cov1σ={m['coverage_1sigma']:.3f}"
                f"  ({elapsed:.1f}s)"
            )
            rows.append({"model": model_name, **{k: m[k] for k in metric_keys}})
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s): {exc}")

    if not rows:
        print("[ERROR] No results to save.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model"] + metric_keys
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {args.output}")

    # Also save JSON with full metadata
    meta = {
        "n_components": args.n_components,
        "n_splits": args.n_splits,
        "sigma_cal_fraction": args.sigma_cal_fraction,
        "calibrate_sigma": not args.no_calibrate_sigma,
        "random_state": args.random_state,
        "n_rows": x.shape[0],
        "n_features": x.shape[1],
        "n_targets": y.shape[1],
        "n_planets": n_planets,
        "results": rows,
    }
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Full metadata saved to {json_path}")

    # Print ranking table
    ranked = sorted(rows, key=lambda r: r["gaussian_nll"])
    print("\nRanking by Gaussian NLL (lower = better):")
    print(f"{'Model':<28} {'NLL':>9} {'RMSE':>9} {'σ mean':>9} {'Cov1σ':>7}")
    print("-" * 66)
    for r in ranked:
        print(
            f"{r['model']:<28} "
            f"{r['gaussian_nll']:>9.4f} "
            f"{r['rmse_mean']:>9.5f} "
            f"{r['sigma_mean']:>9.5f} "
            f"{r['coverage_1sigma']:>7.3f}"
        )


if __name__ == "__main__":
    main()
