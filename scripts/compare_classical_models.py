"""Aggregate phase-5 classical baseline artifacts into comparison outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization import plot_macro_f1_comparison, plot_per_class_delta_heatmap


def _normalize_confusion_refs(refs: dict[str, str]) -> dict[str, str]:
    return {family: str(Path(path).resolve()) for family, path in refs.items()}

FAMILIES = ("logreg", "tree", "boosting")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_comparison(input_root: str | Path, output_root: str | Path) -> dict:
    input_path = Path(input_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    ranking_rows: list[dict] = []
    per_class: dict[str, dict[str, float]] = {}
    confusion_refs: dict[str, str] = {}

    for family in FAMILIES:
        family_root = input_path / family
        metrics_path = family_root / "metrics.json"
        report_path = family_root / "classification_report.json"
        confusion_path = family_root / "confusion_matrix.json"
        metadata_path = family_root / "run_metadata.json"

        if not metrics_path.exists() or not report_path.exists() or not confusion_path.exists():
            continue

        metrics = _load_json(metrics_path)
        report = _load_json(report_path)
        metadata = _load_json(metadata_path) if metadata_path.exists() else {}

        ranking_rows.append(
            {
                "model_family": family,
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "f1_macro": float(metrics.get("f1_macro", 0.0)),
                "precision_macro": float(metrics.get("precision_macro", 0.0)),
                "recall_macro": float(metrics.get("recall_macro", 0.0)),
                "artifact_root": str(family_root),
            }
        )

        for label, values in report.items():
            if label in {"accuracy", "macro avg", "weighted avg"}:
                continue
            if not isinstance(values, dict):
                continue
            per_class.setdefault(label, {})[family] = float(values.get("f1-score", 0.0))

        confusion_refs[family] = str(confusion_path)
        _ = metadata

    if not ranking_rows:
        raise ValueError("No family artifacts found under input root")

    ranking = pd.DataFrame(ranking_rows).sort_values(
        by=["f1_macro", "accuracy", "model_family"],
        ascending=[False, False, True],
    )
    ranking_path = output_path / "model_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    families_in_result = ranking["model_family"].tolist()
    per_class_table = pd.DataFrame.from_dict(per_class, orient="index")
    per_class_table = per_class_table.reindex(columns=families_in_result).fillna(0.0)
    per_class_table.index.name = "label"
    per_class_wide = per_class_table.reset_index().sort_values(by="label")
    per_class_path = output_path / "per_class_f1_comparison.csv"
    per_class_wide.to_csv(per_class_path, index=False)

    best_family = str(ranking.iloc[0]["model_family"])
    best_f1 = float(ranking.iloc[0]["f1_macro"])
    deltas = {
        row["model_family"]: float(row["f1_macro"] - best_f1)
        for _, row in ranking.iterrows()
    }

    per_class_delta = per_class_table.sub(per_class_table.max(axis=1), axis=0)
    per_class_delta_wide = per_class_delta.reset_index().sort_values(by="label")
    per_class_delta_path = output_path / "per_class_f1_delta.csv"
    per_class_delta_wide.to_csv(per_class_delta_path, index=False)

    macro_chart_path = output_path / "macro_f1_by_model.png"
    heatmap_path = output_path / "per_class_f1_delta_heatmap.png"
    plot_macro_f1_comparison(ranking, macro_chart_path)
    plot_per_class_delta_heatmap(per_class_delta_wide, heatmap_path)

    model_summary = {
        row["model_family"]: {
            "accuracy": float(row["accuracy"]),
            "f1_macro": float(row["f1_macro"]),
            "precision_macro": float(row["precision_macro"]),
            "recall_macro": float(row["recall_macro"]),
            "artifact_root": str(row["artifact_root"]),
            "confusion_matrix": confusion_refs.get(row["model_family"], ""),
        }
        for _, row in ranking.iterrows()
    }

    summary = {
        "best_model_family": best_family,
        "metric_deltas_vs_best": deltas,
        "families": families_in_result,
        "artifacts": {
            "model_ranking": str(ranking_path),
            "per_class_f1_comparison": str(per_class_path),
            "per_class_f1_delta": str(per_class_delta_path),
            "macro_f1_chart": str(macro_chart_path),
            "per_class_heatmap": str(heatmap_path),
            "confusion_matrices": _normalize_confusion_refs(confusion_refs),
        },
        "models": model_summary,
    }
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "model_ranking": str(ranking_path),
        "per_class_f1_comparison": str(per_class_path),
        "per_class_f1_delta": str(per_class_delta_path),
        "summary": str(summary_path),
    }


def main() -> int:
    args = parse_args()
    outputs = run_comparison(args.input_root, args.output_root)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
