"""Run TF-IDF + Logistic Regression baseline and log results to MLflow."""

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
from src.models import train_tfidf_logreg


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


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _stringify_params(params: dict) -> dict[str, str]:
    return {
        str(k): json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else str(v)
        for k, v in params.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/processed/evasionbench_prepared.parquet", help="Prepared data"
    )
    parser.add_argument(
        "--contract",
        default="data/contracts/evasionbench_manifest.json",
        help="Dataset manifest for provenance tags",
    )
    parser.add_argument("--output-root", default="artifacts/models/phase5/logreg")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment-name", default="evasionbench-baselines")
    parser.add_argument("--run-name", default="tfidf-logreg")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    contract_path = Path(args.contract)
    output_root = Path(args.output_root)

    frame = pd.read_parquet(data_path)
    training = train_tfidf_logreg(
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

    y_pred = training["model"].predict(training["X_test"])
    metrics = compute_classification_metrics(training["y_test"], y_pred)

    run_metadata = {
        "model_family": "logreg",
        "split_seed": args.random_state,
        "test_size": args.test_size,
        "split_metadata": training["split_metadata"],
        "feature_config": training["vectorizer_params"],
        "model_config": training["classifier_params"],
        "train_rows": len(training["X_train"]),
        "test_rows": len(training["X_test"]),
        "data_path": str(data_path),
    }

    artifacts = write_evaluation_artifacts(
        output_dir=output_root,
        y_true=training["y_test"],
        y_pred=y_pred,
        metrics=metrics,
        metadata=run_metadata,
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    manifest = _load_manifest(contract_path)
    dataset_meta = manifest.get("dataset", {})

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "model_type": "tfidf_logreg",
                "data_path": str(data_path),
                "target_col": args.target_col,
                "random_state": args.random_state,
                "test_size": args.test_size,
                "split_strategy": training["split_metadata"]["method"],
                "split_stratified": training["split_metadata"]["stratify"],
                "train_rows": len(training["X_train"]),
                "test_rows": len(training["X_test"]),
            }
        )
        mlflow.log_params(
            _stringify_params({f"tfidf_{k}": v for k, v in training["vectorizer_params"].items()})
        )
        mlflow.log_params(
            _stringify_params({f"clf_{k}": v for k, v in training["classifier_params"].items()})
        )
        mlflow.log_metrics(metrics)

        tags = {
            "dataset_checksum": manifest.get("checksum_sha256", ""),
            "dataset_revision": dataset_meta.get("revision", ""),
            "dataset_id": dataset_meta.get("dataset_id", ""),
            "dataset_split": dataset_meta.get("split", ""),
            "git_sha": _git_sha(),
            "pipeline_stage": "phase-05-classical-baselines",
            "split_strategy": training["split_metadata"]["method"],
        }
        mlflow.set_tags(tags)

        for artifact in artifacts:
            mlflow.log_artifact(str(artifact), artifact_path="evaluation")

        print(f"MLflow run complete: run_id={run.info.run_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
