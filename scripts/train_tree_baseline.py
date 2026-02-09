"""Train phase-5 tree or boosting baselines and write canonical artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import compute_classification_metrics, write_evaluation_artifacts
from src.models import train_tree_or_boosting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-col", default="label")
    parser.add_argument("--model-family", choices=["tree", "boosting"], default="tree")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_parquet(args.input)

    trained = train_tree_or_boosting(
        frame,
        model_family=args.model_family,
        target_col=args.target_col,
        random_state=args.random_state,
        test_size=args.test_size,
    )
    metrics = compute_classification_metrics(trained["y_test"], trained["predictions"])

    output_root = Path(args.output_root)
    metadata = {
        "model_family": args.model_family,
        "split_seed": args.random_state,
        "test_size": args.test_size,
        "feature_config": trained["vectorizer_params"],
        "model_config": trained["estimator_params"],
        "split_metadata": trained["split_metadata"],
        "train_rows": int(len(trained["y_train"])),
        "test_rows": int(len(trained["y_test"])),
    }
    if args.model_family == "boosting":
        metadata["feature_reduction"] = {
            "method": "TruncatedSVD",
            "components": trained["estimator_params"].get("svd_components"),
        }

    write_evaluation_artifacts(
        output_root,
        trained["y_test"],
        trained["predictions"],
        metrics,
        metadata,
    )

    print(json.dumps({"model_family": args.model_family, "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
