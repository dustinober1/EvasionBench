"""Run phase-5 classical baselines with a shared artifact contract."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import compute_classification_metrics, write_evaluation_artifacts
from src.models import train_tfidf_logreg, train_tree_or_boosting


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared parquet dataset")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root output directory for phase-5 artifacts",
    )
    parser.add_argument(
        "--families",
        choices=["all", "logreg", "tree", "boosting"],
        default="all",
        help="Which model family/families to run (tree runs both tree and boosting)",
    )
    parser.add_argument("--target-col", default="label")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--boosting-max-iter", type=int, default=50)
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment-name", default="evasionbench-classical-baselines")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="CV folds for optional hyperparameter search mode",
    )
    parser.add_argument(
        "--enable-hparam-search",
        action="store_true",
        help="Run phase-8 optimization workflow instead of fixed phase-5 baselines",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["f1_macro", "accuracy", "precision_macro", "recall_macro"],
        default="f1_macro",
        help="Primary metric for optional hyperparameter selection",
    )
    parser.add_argument(
        "--accuracy-floor",
        type=float,
        default=0.6431560071727436,
        help="Minimum holdout accuracy required for winner eligibility in search mode",
    )
    parser.add_argument(
        "--holdout-size",
        type=float,
        default=0.2,
        help="Holdout size for optional hyperparameter search mode",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run model comparison after families finish",
    )
    return parser.parse_args()


def _family_to_dir_name(family: str) -> str:
    if family == "logreg":
        return "logreg"
    if family in {"tree", "boosting"}:
        return family
    raise ValueError(f"Unsupported family: {family}")


def _metadata(
    *,
    family: str,
    random_state: int,
    test_size: float,
    train_rows: int,
    test_rows: int,
    feature_config: dict,
    model_config: dict,
    split_metadata: dict,
) -> dict:
    return {
        "model_family": family,
        "split_seed": random_state,
        "test_size": test_size,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "feature_config": feature_config,
        "model_config": model_config,
        "split_metadata": split_metadata,
        "git_sha": _git_sha(),
    }


def _stringify_params(params: dict) -> dict[str, str]:
    return {
        str(k): json.dumps(v, sort_keys=True)
        if isinstance(v, (dict, list, tuple))
        else str(v)
        for k, v in params.items()
    }


def _run_logreg(
    frame: pd.DataFrame, args: argparse.Namespace, output_root: Path
) -> dict[str, float]:
    trained = train_tfidf_logreg(
        frame,
        target_col=args.target_col,
        random_state=args.random_state,
        test_size=args.test_size,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_features=args.max_features,
        c=args.c,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        solver=args.solver,
    )
    y_pred = trained["model"].predict(trained["X_test"])
    metrics = compute_classification_metrics(trained["y_test"], y_pred)

    family_dir = output_root / _family_to_dir_name("logreg")
    family_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(
        family="logreg",
        random_state=args.random_state,
        test_size=args.test_size,
        train_rows=len(trained["X_train"]),
        test_rows=len(trained["X_test"]),
        feature_config=trained["vectorizer_params"],
        model_config=trained["classifier_params"],
        split_metadata=trained["split_metadata"],
    )

    # Save model for downstream use (e.g., explainability)
    model_path = family_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(trained["model"], f)

    artifacts = write_evaluation_artifacts(
        family_dir, trained["y_test"], y_pred, metrics, metadata
    )
    artifacts.append(model_path)

    with mlflow.start_run(run_name="phase5-logreg"):
        mlflow.log_params(
            {
                "model_family": "logreg",
                "target_col": args.target_col,
                "random_state": args.random_state,
                "test_size": args.test_size,
                "split_strategy": trained["split_metadata"]["method"],
                "split_stratified": trained["split_metadata"]["stratify"],
                "train_rows": len(trained["X_train"]),
                "test_rows": len(trained["X_test"]),
            }
        )
        mlflow.log_params(
            _stringify_params(
                {f"tfidf_{k}": v for k, v in trained["vectorizer_params"].items()}
            )
        )
        mlflow.log_params(
            _stringify_params(
                {f"clf_{k}": v for k, v in trained["classifier_params"].items()}
            )
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "pipeline_stage": "phase-05-classical-baselines",
                "git_sha": _git_sha(),
                "split_strategy": trained["split_metadata"]["method"],
            }
        )
        for path in artifacts:
            mlflow.log_artifact(str(path), artifact_path=f"phase5/{family_dir.name}")

    return metrics


def _run_tree_or_boosting(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    output_root: Path,
    family: str,
) -> dict[str, float]:
    trained = train_tree_or_boosting(
        frame,
        model_family=family,
        target_col=args.target_col,
        random_state=args.random_state,
        test_size=args.test_size,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_features=args.max_features,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        boosting_max_iter=args.boosting_max_iter,
    )
    metrics = compute_classification_metrics(trained["y_test"], trained["predictions"])

    family_dir = output_root / _family_to_dir_name(family)
    family_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(
        family=family,
        random_state=args.random_state,
        test_size=args.test_size,
        train_rows=len(trained["y_train"]),
        test_rows=len(trained["y_test"]),
        feature_config=trained["vectorizer_params"],
        model_config=trained["estimator_params"],
        split_metadata=trained["split_metadata"],
    )

    # Save model bundle for downstream use (includes vectorizer, model, and SVD if applicable)
    model_bundle_path = family_dir / "model_bundle.pkl"
    with open(model_bundle_path, "wb") as f:
        pickle.dump(
            {
                "model": trained["model"],
                "vectorizer": trained["vectorizer"],
                "svd": trained.get("svd"),  # Only for boosting
            },
            f,
        )

    artifacts = write_evaluation_artifacts(
        family_dir,
        trained["y_test"],
        trained["predictions"],
        metrics,
        metadata,
    )
    artifacts.append(model_bundle_path)

    with mlflow.start_run(run_name=f"phase5-{family}"):
        mlflow.log_params(
            {
                "model_family": family,
                "target_col": args.target_col,
                "random_state": args.random_state,
                "test_size": args.test_size,
                "split_strategy": trained["split_metadata"]["method"],
                "split_stratified": trained["split_metadata"]["stratify"],
                "train_rows": len(trained["y_train"]),
                "test_rows": len(trained["y_test"]),
            }
        )
        mlflow.log_params(
            _stringify_params(
                {f"tfidf_{k}": v for k, v in trained["vectorizer_params"].items()}
            )
        )
        mlflow.log_params(
            _stringify_params(
                {f"estimator_{k}": v for k, v in trained["estimator_params"].items()}
            )
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "pipeline_stage": "phase-05-classical-baselines",
                "git_sha": _git_sha(),
                "split_strategy": trained["split_metadata"]["method"],
            }
        )
        for path in artifacts:
            mlflow.log_artifact(str(path), artifact_path=f"phase5/{family_dir.name}")

    return metrics


def _selected_families(raw: str) -> list[str]:
    if raw == "all":
        return ["logreg", "tree", "boosting"]
    if raw == "tree":
        return ["tree", "boosting"]
    return [raw]


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.enable_hparam_search:
        subprocess.run(
            [
                sys.executable,
                "scripts/run_model_optimization.py",
                "--input",
                args.input,
                "--output-root",
                str(output_root),
                "--target-col",
                args.target_col,
                "--random-state",
                str(args.random_state),
                "--holdout-size",
                str(args.holdout_size),
                "--cv-folds",
                str(args.cv_folds),
                "--selection-metric",
                args.selection_metric,
                "--accuracy-floor",
                str(args.accuracy_floor),
                "--families",
                args.families,
                "--tracking-uri",
                args.tracking_uri,
                "--experiment-name",
                args.experiment_name.replace(
                    "classical-baselines", "model-optimization"
                ),
            ],
            check=True,
        )
        return 0

    frame = pd.read_parquet(args.input)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    metrics_summary: dict[str, dict[str, float]] = {}
    for family in _selected_families(args.families):
        if family == "logreg":
            metrics_summary[family] = _run_logreg(frame, args, output_root)
        else:
            metrics_summary[family] = _run_tree_or_boosting(
                frame, args, output_root, family
            )

    summary_path = output_root / "run_summary.json"
    summary_path.write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.compare:
        subprocess.run(
            [
                sys.executable,
                "scripts/compare_classical_models.py",
                "--input-root",
                str(output_root),
                "--output-root",
                str(output_root / "model_comparison"),
            ],
            check=True,
        )

    print(
        json.dumps(
            {
                "families": sorted(metrics_summary.keys()),
                "output_root": str(output_root),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
