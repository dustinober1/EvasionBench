from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def write_evaluation_artifacts(
    output_dir: Path,
    y_true: Iterable[str],
    y_pred: Iterable[str],
    metrics: dict[str, float],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "classification_report.json"
    matrix_path = output_dir / "confusion_matrix.json"
    metrics_path = output_dir / "metrics.json"

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred).tolist()

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    matrix_path.write_text(json.dumps({"confusion_matrix": matrix}, indent=2, sort_keys=True) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    return [report_path, matrix_path, metrics_path]
