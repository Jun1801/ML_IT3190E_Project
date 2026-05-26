from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ariel spectrum model from feature CSV.")
    parser.add_argument("--features", type=Path, default=Path("outputs/features_train.csv"))
    parser.add_argument("--targets", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model"))
    parser.add_argument("--model", default="bayesian_ridge")
    parser.add_argument("--n-components", type=int, default=30)
    parser.add_argument("--search", action="store_true", help="Run a small model/PCA search before final refit.")
    parser.add_argument(
        "--model-candidates",
        default="bayesian_ridge,ridge,kernel_ridge,extra_trees,boosting",
        help="Comma-separated model names for --search.",
    )
    parser.add_argument(
        "--n-components-grid",
        default="20,30,40",
        help="Comma-separated PCA component counts for --search.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=0, help="Run K-fold CV instead of one validation split when >1.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-sigma-calibration", action="store_true")
    parser.add_argument("--no-refit-full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import joblib
    import pandas as pd

    from ariel_ml.config import ModelConfig
    from ariel_ml.dataset_builder import align_features_and_targets
    from ariel_ml.training import cross_validate_model, hyperparameter_search, refit_full_model, train_model

    features = pd.read_csv(args.features)
    targets = pd.read_csv(args.targets)
    x_frame, y, groups, target_columns = align_features_and_targets(features, targets)
    model_config = ModelConfig(
        n_components=args.n_components,
        random_state=args.random_state,
        calibrate_sigma=not args.no_sigma_calibration,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.search:
        search = hyperparameter_search(
            x_frame.to_numpy(dtype=float),
            y,
            model_names=[name.strip() for name in args.model_candidates.split(",") if name.strip()],
            n_components_grid=[int(value) for value in args.n_components_grid.split(",") if value.strip()],
            base_config=model_config,
            n_splits=args.cv if args.cv and args.cv > 1 else 5,
            groups=groups,
            random_state=args.random_state,
        )
        search_rows = [
            {
                "model_name": candidate.model_name,
                "n_components": candidate.model_config.n_components,
                **candidate.mean_metrics,
            }
            for candidate in search.candidates
        ]
        (args.output_dir / "search_results.json").write_text(
            json.dumps(search_rows, indent=2),
            encoding="utf-8",
        )
        args.model = search.best_candidate.model_name
        model_config = search.best_candidate.model_config
        print("Best candidate:")
        print(json.dumps(search_rows[min(range(len(search_rows)), key=lambda idx: search_rows[idx]["gaussian_nll"])], indent=2))

    elif args.cv and args.cv > 1:
        cv_result = cross_validate_model(
            x_frame.to_numpy(dtype=float),
            y,
            model_name=args.model,
            model_config=model_config,
            n_splits=args.cv,
            groups=groups,
            random_state=args.random_state,
        )
        metrics = cv_result.mean_metrics
        (args.output_dir / "cv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return

    result = train_model(
        x_frame.to_numpy(dtype=float),
        y,
        model_name=args.model,
        model_config=model_config,
        validation_fraction=args.validation_fraction,
        groups=groups,
        random_state=args.random_state,
    )
    final_model = result.model
    if not args.no_refit_full:
        final_model = refit_full_model(
            x_frame.to_numpy(dtype=float),
            y,
            model_name=args.model,
            model_config=model_config,
        )
    artifact = {
        "model": final_model,
        "validation_model": result.model,
        "feature_columns": list(x_frame.columns),
        "target_columns": target_columns,
        "metrics": result.evaluation.as_dict(),
        "model_name": args.model,
        "model_config": model_config,
    }
    joblib.dump(artifact, args.output_dir / "model.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result.evaluation.as_dict(), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result.evaluation.as_dict(), indent=2))


if __name__ == "__main__":
    main()
