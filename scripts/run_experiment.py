"""Run a script-first baseline experiment and log results to MLflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
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
    return json.loads(path.read_text())


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
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment-name", default="evasionbench-baselines")
    parser.add_argument("--run-name", default="tfidf-logreg")
    parser.add_argument("--target-col", default="label")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    contract_path = Path(args.contract)

    frame = pd.read_parquet(data_path)
    training = train_tfidf_logreg(
        frame,
        target_col=args.target_col,
        random_state=args.random_state,
    )

    y_pred = training["model"].predict(training["X_test"])
    metrics = compute_classification_metrics(training["y_test"], y_pred)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    manifest = _load_manifest(contract_path)
    dataset_meta = manifest.get("dataset", {})

    with tempfile.TemporaryDirectory() as tmp:
        artifacts = write_evaluation_artifacts(
            output_dir=Path(tmp),
            y_true=training["y_test"],
            y_pred=y_pred,
            metrics=metrics,
        )

        with mlflow.start_run(run_name=args.run_name) as run:
            mlflow.log_params(
                {
                    "model_type": "tfidf_logreg",
                    "data_path": str(data_path),
                    "target_col": args.target_col,
                    "random_state": args.random_state,
                    "train_rows": len(training["X_train"]),
                    "test_rows": len(training["X_test"]),
                }
            )
            mlflow.log_metrics(metrics)

            tags = {
                "dataset_checksum": manifest.get("checksum_sha256", ""),
                "dataset_revision": dataset_meta.get("revision", ""),
                "dataset_id": dataset_meta.get("dataset_id", ""),
                "dataset_split": dataset_meta.get("split", ""),
                "git_sha": _git_sha(),
                "pipeline_stage": "phase-02-data-lineage",
            }
            mlflow.set_tags(tags)

            for artifact in artifacts:
                mlflow.log_artifact(str(artifact), artifact_path="evaluation")

            print(f"MLflow run complete: run_id={run.info.run_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
