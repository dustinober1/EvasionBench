"""Run phase-6 transformer baselines with a shared artifact contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.transformers
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import write_evaluation_artifacts
from src.models import train_transformer


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
    parser.add_argument("--output-root", required=True, help="Root output directory for phase-6 artifacts")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Hugging Face model name")
    parser.add_argument("--max-epochs", type=int, default=3, help="Maximum training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--target-col", default="label")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment-name", default="evasionbench-transformer-baselines")
    parser.add_argument("--model-name-registry", default="evasionbench-distilbert", help="MLflow registered model name")
    return parser.parse_args()


def _metadata(
    *,
    model_name: str,
    random_state: int,
    test_size: float,
    train_rows: int,
    test_rows: int,
    model_config: dict,
    split_metadata: dict,
) -> dict:
    return {
        "model_family": "transformer",
        "model_name": model_name,
        "split_seed": random_state,
        "test_size": test_size,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "model_config": model_config,
        "split_metadata": split_metadata,
        "git_sha": _git_sha(),
    }


def _stringify_params(params: dict) -> dict[str, str]:
    return {
        str(k): json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else str(v)
        for k, v in params.items()
    }


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.input)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Enable transformers autologging
    mlflow.transformers.autolog()

    with mlflow.start_run(run_name="phase6-transformer"):
        trained = train_transformer(
            frame,
            target_col=args.target_col,
            random_state=args.random_state,
            test_size=args.test_size,
            model_name=args.model_name,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
        )

        # Compute metrics
        from src.evaluation import compute_classification_metrics

        metrics = compute_classification_metrics(trained["y_true"], trained["y_pred"])

        # Write evaluation artifacts
        metadata = _metadata(
            model_name=args.model_name,
            random_state=args.random_state,
            test_size=args.test_size,
            train_rows=trained["split_metadata"]["train_rows"],
            test_rows=trained["split_metadata"]["test_rows"],
            model_config=trained["model_config"],
            split_metadata=trained["split_metadata"],
        )
        artifacts = write_evaluation_artifacts(
            output_root, trained["y_true"], trained["y_pred"], metrics, metadata
        )

        # Save model checkpoints
        model_dir = output_root / "model"
        model_dir.mkdir(exist_ok=True)
        trained["model"].save_pretrained(model_dir)
        trained["tokenizer"].save_pretrained(model_dir)

        # Log custom metrics
        mlflow.log_metrics(metrics)

        # Log params
        mlflow.log_params(
            {
                "model_family": "transformer",
                "model_name": args.model_name,
                "target_col": args.target_col,
                "random_state": args.random_state,
                "test_size": args.test_size,
                "split_strategy": trained["split_metadata"]["method"],
                "split_stratified": trained["split_metadata"]["stratify"],
                "train_rows": trained["split_metadata"]["train_rows"],
                "test_rows": trained["split_metadata"]["test_rows"],
                "max_epochs": args.max_epochs,
                "learning_rate": args.learning_rate,
            }
        )
        mlflow.log_params(_stringify_params(trained["model_config"]))

        # Log artifacts
        for path in artifacts:
            mlflow.log_artifact(str(path), artifact_path="phase6/transformer")

        # Log model explicitly with MLflow transformers flavor
        model_info = mlflow.transformers.log_model(
            transformers_model={"model": trained["model"], "tokenizer": trained["tokenizer"]},
            artifact_path="transformer_model",
            task="text-classification",
            # input_example="Sample question text [SEP] Sample answer text",
        )

        # Register model
        mlflow.register_model(
            model_uri=model_info.model_uri,
            name=args.model_name_registry,
            tags={"phase": "06", "task": "binary-classification"},
        )

        # Set tags
        mlflow.set_tags(
            {
                "pipeline_stage": "phase-06-transformer-baselines",
                "git_sha": _git_sha(),
                "split_strategy": trained["split_metadata"]["method"],
                "label2id": json.dumps(trained["label2id"]),
                "id2label": json.dumps(trained["id2label"]),
            }
        )

    print(
        json.dumps(
            {
                "model_name": args.model_name,
                "metrics": metrics,
                "output_root": str(output_root),
                "model_uri": model_info.model_uri,
                "registered_model": args.model_name_registry,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
