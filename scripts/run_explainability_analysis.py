"""Generate SHAP explainability artifacts for Phase 5 classical models."""

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

from src.explainability import explain_classical_model


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
    parser.add_argument(
        "--data", required=True, help="Prepared parquet dataset (same as Phase 5 input)"
    )
    parser.add_argument(
        "--models-root",
        default="artifacts/models/phase5",
        help="Root directory containing Phase 5 model artifacts",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/explainability/phase6",
        help="Root output directory for XAI artifacts",
    )
    parser.add_argument(
        "--families",
        choices=["all", "logreg", "tree", "boosting"],
        default="all",
        help="Which model family/families to generate explanations for",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=100, help="Background samples for SHAP")
    parser.add_argument("--top-k", type=int, default=20, help="Top features for global importance")
    parser.add_argument("--samples-per-class", type=int, default=5, help="Samples per class for local explanations")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment-name", default="evasionbench-explainability")
    return parser.parse_args()


def _family_to_dir_name(family: str) -> str:
    """Convert family name to directory name."""
    if family == "logreg":
        return "logreg"
    if family in {"tree", "boosting"}:
        return family
    raise ValueError(f"Unsupported family: {family}")


def _selected_families(raw: str) -> list[str]:
    """Convert CLI families argument to list."""
    if raw == "all":
        return ["logreg", "tree", "boosting"]
    if raw == "tree":
        return ["tree", "boosting"]
    return [raw]


def main() -> int:
    args = parse_args()
    models_root = Path(args.models_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Validate data path
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        return 1

    # Setup MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Track results across families
    xai_summary: dict[str, dict] = {}

    for family in _selected_families(args.families):
        family_dir = models_root / _family_to_dir_name(family)

        # Check if model directory exists
        if not family_dir.exists():
            print(f"Warning: Model directory not found: {family_dir}, skipping {family}", file=sys.stderr)
            continue

        # Check for required metadata
        metadata_path = family_dir / "run_metadata.json"
        if not metadata_path.exists():
            print(f"Warning: Metadata not found: {metadata_path}, skipping {family}", file=sys.stderr)
            continue

        # Load metadata to get info
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        # Create output directory for this family
        family_output_dir = output_root / _family_to_dir_name(family)
        family_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating SHAP explanations for {family}...", file=sys.stderr)

        try:
            with mlflow.start_run(run_name=f"phase6-xai-{family}"):
                # Generate SHAP explanations
                result = explain_classical_model(
                    model_dir=family_dir,
                    data_path=data_path,
                    output_dir=family_output_dir,
                    random_state=args.random_state,
                    n_samples=args.n_samples,
                    top_k=args.top_k,
                    samples_per_class=args.samples_per_class,
                )

                # Log parameters
                mlflow.log_params(
                    {
                        "model_family": family,
                        "data_path": str(data_path),
                        "random_state": args.random_state,
                        "n_samples": args.n_samples,
                        "top_k": args.top_k,
                        "samples_per_class": args.samples_per_class,
                    }
                )

                # Log explainer type
                mlflow.log_param("explainer_type", result["explainer_type"])

                # Log feature count
                mlflow.log_metric("n_features", len(result["feature_names"]))
                mlflow.log_metric("n_samples_explained", len(result["sample_indices"]))

                # Log artifacts
                for artifact_path in family_output_dir.glob("*"):
                    mlflow.log_artifact(str(artifact_path), artifact_path=f"phase6_xai/{family}")

                # Log metadata tags
                mlflow.set_tags(
                    {
                        "pipeline_stage": "phase-06-explainability",
                        "git_sha": _git_sha(),
                        "model_family": family,
                    }
                )

                # Add to summary
                xai_summary[family] = {
                    "explainer_type": result["explainer_type"],
                    "n_features": len(result["feature_names"]),
                    "n_samples": len(result["sample_indices"]),
                    "output_dir": str(family_output_dir),
                    "artifacts": [p.name for p in family_output_dir.glob("*")],
                }

                print(f"  ✓ {family}: {len(result['feature_names'])} features, {len(result['sample_indices'])} samples", file=sys.stderr)

        except Exception as e:
            print(f"Error generating explanations for {family}: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            continue

    # Write combined summary
    summary_path = output_root / "xai_summary.json"
    summary_path.write_text(json.dumps(xai_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"families": sorted(xai_summary.keys()), "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
