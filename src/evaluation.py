from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

REQUIRED_EVAL_FILES = (
    "metrics.json",
    "classification_report.json",
    "confusion_matrix.json",
    "run_metadata.json",
)


def _resolve_labels(y_true: Sequence[str], y_pred: Sequence[str]) -> list[str]:
    labels = sorted({str(v) for v in list(y_true) + list(y_pred)})
    if not labels:
        raise ValueError("Cannot evaluate empty predictions")
    return labels


def compute_classification_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
) -> dict[str, float]:
    y_true_seq = [str(v) for v in y_true]
    y_pred_seq = [str(v) for v in y_pred]
    return {
        "accuracy": float(accuracy_score(y_true_seq, y_pred_seq)),
        "f1_macro": float(
            f1_score(y_true_seq, y_pred_seq, average="macro", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true_seq, y_pred_seq, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_seq, y_pred_seq, average="macro", zero_division=0)
        ),
    }


def write_evaluation_artifacts(
    output_dir: Path,
    y_true: Iterable[str],
    y_pred: Iterable[str],
    metrics: dict[str, float],
    metadata: dict[str, Any] | None = None,
) -> list[Path]:
    """Write the canonical phase-5 classical model artifact contract."""
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true_seq = [str(v) for v in y_true]
    y_pred_seq = [str(v) for v in y_pred]
    labels = _resolve_labels(y_true_seq, y_pred_seq)

    report = classification_report(
        y_true_seq,
        y_pred_seq,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true_seq, y_pred_seq, labels=labels).tolist()

    report_path = output_dir / "classification_report.json"
    matrix_path = output_dir / "confusion_matrix.json"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "run_metadata.json"

    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    matrix_path.write_text(
        json.dumps(
            {"labels": labels, "confusion_matrix": matrix}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    payload = dict(metadata or {})
    payload.setdefault("labels", labels)
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return [metrics_path, report_path, matrix_path, metadata_path]


def validate_evaluation_contract(output_dir: str | Path) -> None:
    root = Path(output_dir)
    missing = [name for name in REQUIRED_EVAL_FILES if not (root / name).exists()]
    if missing:
        raise ValueError(
            f"Missing required evaluation artifact files: {', '.join(missing)}"
        )

    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    for key in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
        if key not in metrics:
            raise ValueError(f"metrics.json missing required key: {key}")

    report = json.loads(
        (root / "classification_report.json").read_text(encoding="utf-8")
    )
    for key in ("accuracy", "macro avg", "weighted avg"):
        if key not in report:
            raise ValueError(f"classification_report.json missing required key: {key}")

    matrix = json.loads((root / "confusion_matrix.json").read_text(encoding="utf-8"))
    if "labels" not in matrix or "confusion_matrix" not in matrix:
        raise ValueError(
            "confusion_matrix.json missing required keys: labels, confusion_matrix"
        )

    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    for key in ("model_family", "split_seed", "feature_config"):
        if key not in metadata:
            raise ValueError(f"run_metadata.json missing required key: {key}")
