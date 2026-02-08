"""Run phase-5 classical baselines with a shared artifact contract."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--output-root", required=True, help="Root output directory for phase-5 artifacts")
    parser.add_argument(
        "--families",
        choices=["all", "logreg", "tree", "boosting"],
        default="all",
        help="Which model family/families to run",
    )
    parser.add_argument("--target-col", default="label")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment-name", default="evasionbench-classical-baselines")
    parser.add_argument("--compare", action="store_true", help="Run model comparison after families finish")
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
) -> dict:
    return {
        "model_family": family,
        "split_seed": random_state,
        "test_size": test_size,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "feature_config": feature_config,
        "model_config": model_config,
        "git_sha": _git_sha(),
    }


def _stringify_params(params: dict) -> dict[str, str]:
    return {str(k): json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else str(v) for k, v in params.items()}


def _run_logreg(frame: pd.DataFrame, args: argparse.Namespace, output_root: Path) -> dict[str, float]:
    trained = train_tfidf_logreg(
        frame,
        target_col=args.target_col,
        random_state=args.random_state,
        test_size=args.test_size,
    )
    y_pred = trained["model"].predict(trained["X_test"])
    metrics = compute_classification_metrics(trained["y_test"], y_pred)

    family_dir = output_root / _family_to_dir_name("logreg")
    metadata = _metadata(
        family="logreg",
        random_state=args.random_state,
        test_size=args.test_size,
        train_rows=len(trained["X_train"]),
        test_rows=len(trained["X_test"]),
        feature_config=trained["vectorizer_params"],
        model_config=trained["classifier_params"],
    )
    artifacts = write_evaluation_artifacts(family_dir, trained["y_test"], y_pred, metrics, metadata)

    with mlflow.start_run(run_name="phase5-logreg"):
        mlflow.log_params(
            {
                "model_family": "logreg",
                "target_col": args.target_col,
                "random_state": args.random_state,
                "test_size": args.test_size,
                "train_rows": len(trained["X_train"]),
                "test_rows": len(trained["X_test"]),
            }
        )
        mlflow.log_params(
            _stringify_params({f"tfidf_{k}": v for k, v in trained["vectorizer_params"].items()})
        )
        mlflow.log_params(
            _stringify_params({f"clf_{k}": v for k, v in trained["classifier_params"].items()})
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"pipeline_stage": "phase-05-classical-baselines", "git_sha": _git_sha()})
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
    )
    metrics = compute_classification_metrics(trained["y_test"], trained["predictions"])

    family_dir = output_root / _family_to_dir_name(family)
    metadata = _metadata(
        family=family,
        random_state=args.random_state,
        test_size=args.test_size,
        train_rows=len(trained["y_train"]),
        test_rows=len(trained["y_test"]),
        feature_config=trained["vectorizer_params"],
        model_config=trained["estimator_params"],
    )
    artifacts = write_evaluation_artifacts(
        family_dir,
        trained["y_test"],
        trained["predictions"],
        metrics,
        metadata,
    )

    with mlflow.start_run(run_name=f"phase5-{family}"):
        mlflow.log_params(
            {
                "model_family": family,
                "target_col": args.target_col,
                "random_state": args.random_state,
                "test_size": args.test_size,
                "train_rows": len(trained["y_train"]),
                "test_rows": len(trained["y_test"]),
            }
        )
        mlflow.log_params(
            _stringify_params({f"tfidf_{k}": v for k, v in trained["vectorizer_params"].items()})
        )
        mlflow.log_params(
            _stringify_params(
                {f"estimator_{k}": v for k, v in trained["estimator_params"].items()}
            )
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"pipeline_stage": "phase-05-classical-baselines", "git_sha": _git_sha()})
        for path in artifacts:
            mlflow.log_artifact(str(path), artifact_path=f"phase5/{family_dir.name}")

    return metrics


def _selected_families(raw: str) -> list[str]:
    if raw == "all":
        return ["logreg", "tree", "boosting"]
    return [raw]


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.input)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    metrics_summary: dict[str, dict[str, float]] = {}
    for family in _selected_families(args.families):
        if family == "logreg":
            metrics_summary[family] = _run_logreg(frame, args, output_root)
        else:
            metrics_summary[family] = _run_tree_or_boosting(frame, args, output_root, family)

    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(metrics_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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

    print(json.dumps({"families": sorted(metrics_summary.keys()), "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
