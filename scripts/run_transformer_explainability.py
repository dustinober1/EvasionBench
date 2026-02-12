"""Run transformer explainability analysis with Captum attributions."""

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

from src.explainability import explain_transformer_batch


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
    parser = argparse.ArgumentParser(
        description="Generate Captum attributions for transformer predictions"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained transformer checkpoint directory",
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/evasionbench_prepared.parquet",
        help="Path to prepared dataset parquet file",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root output directory for XAI artifacts",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of samples to explain (default: 20)",
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="Column name containing text (default: text)",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Column name containing labels (default: label)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run attribution on (default: cpu)",
    )
    parser.add_argument(
        "--tracking-uri",
        default="file:./mlruns",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-name",
        default="evasionbench-transformer-xai",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    output_root = Path(args.output_root)

    # Validate inputs
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}", file=sys.stderr)
        return 1

    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    # Load transformer model and tokenizer
    print(f"Loading transformer model from {model_path}...")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Model loaded successfully: {type(model).__name__}")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1

    # Load data
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
        print(f"Data loaded: {len(df)} rows")
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1

    # Create text column from question + answer if it doesn't exist
    if args.text_col not in df.columns:
        if "question" in df.columns and "answer" in df.columns:
            print(f"Creating 'text' column from question + answer...")
            df["text"] = (
                df["question"].fillna("").astype(str)
                + " [SEP] "
                + df["answer"].fillna("").astype(str)
            )
            text_col = "text"
        else:
            print(
                f"Error: Text column '{args.text_col}' not found in data. "
                f"Available columns: {df.columns.tolist()}",
                file=sys.stderr,
            )
            return 1
    else:
        text_col = args.text_col

    # Validate label column
    if args.label_col not in df.columns:
        print(
            f"Error: Label column '{args.label_col}' not found in data. "
            f"Available columns: {df.columns.tolist()}",
            file=sys.stderr,
        )
        return 1

    # Setup MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Generate explainability artifacts
    print(f"Generating Captum attributions for {args.n_samples} samples...")
    print(f"Using device: {args.device}")

    with mlflow.start_run(run_name="transformer-xai"):
        result = explain_transformer_batch(
            model=model,
            tokenizer=tokenizer,
            data=df,
            output_dir=output_root,
            text_col=text_col,
            label_col=args.label_col,
            n_samples=args.n_samples,
            random_state=args.random_state,
            device=args.device,
        )

        # Log parameters
        mlflow.log_params(
            {
                "model_path": str(model_path),
                "data_path": str(data_path),
                "n_samples": args.n_samples,
                "text_col": text_col,
                "label_col": args.label_col,
                "device": args.device,
                "random_state": args.random_state,
                "explainer_type": "LayerIntegratedGradients",
                "target_layer": "embeddings",
            }
        )

        # Log metrics
        summary = result["summary"]
        mlflow.log_metrics(
            {
                "n_samples_explained": summary["n_samples"],
                "avg_attribution_sum": summary["avg_attribution_sum"],
            }
        )

        # Log artifacts
        mlflow.log_artifact(
            str(output_root / "transformer_xai.json"), artifact_path="xai"
        )
        mlflow.log_artifact(
            str(output_root / "transformer_xai_summary.json"), artifact_path="xai"
        )
        mlflow.log_artifact(
            str(output_root / "transformer_xai.html"), artifact_path="xai"
        )

        # Set tags
        mlflow.set_tags(
            {
                "pipeline_stage": "phase-06-transformer-xai",
                "explainer": "captum",
                "attribution_method": "LayerIntegratedGradients",
                "git_sha": _git_sha(),
            }
        )

    print(f"✓ Explainability artifacts generated in {output_root}")
    print(f"  - transformer_xai.json: {summary['n_samples']} sample explanations")
    print(f"  - transformer_xai_summary.json: aggregate statistics")
    print(f"  - transformer_xai.html: interactive visualization")
    print(f"  - Avg attribution sum: {summary['avg_attribution_sum']:.4f}")

    # Output summary JSON
    output_summary = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "output_root": str(output_root),
        "n_samples": summary["n_samples"],
        "avg_attribution_sum": summary["avg_attribution_sum"],
        "artifacts": {
            "transformer_xai.json": str(output_root / "transformer_xai.json"),
            "transformer_xai_summary.json": str(
                output_root / "transformer_xai_summary.json"
            ),
            "transformer_xai.html": str(output_root / "transformer_xai.html"),
        },
    }

    summary_file = output_root / "run_summary.json"
    summary_file.write_text(
        json.dumps(output_summary, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(output_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
