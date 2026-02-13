"""Generate phase-9 error-analysis artifacts from selected model outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import load_model
from src.reporting import normalize_path, utc_now_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/evasionbench_prepared.parquet",
        help="Prepared dataset with question/answer/label columns.",
    )
    parser.add_argument(
        "--selected-model-summary",
        default="artifacts/models/phase8/selected_model.json",
        help="Phase-8 selected model summary JSON.",
    )
    parser.add_argument(
        "--phase5-summary",
        default="artifacts/models/phase5/model_comparison/summary.json",
        help="Phase-5 model-comparison summary JSON.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/analysis/phase9/error_analysis",
        help="Output directory for phase-9 error-analysis artifacts.",
    )
    parser.add_argument(
        "--project-root",
        default=str(ROOT),
        help="Project root used for normalized artifact paths.",
    )
    parser.add_argument(
        "--hard-case-limit",
        type=int,
        default=3,
        help="Maximum number of hardest classes to summarize.",
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=2,
        help="Maximum evidence snippets per class when model inference is available.",
    )
    return parser.parse_args()


def _resolve(path_like: str, *, project_root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(report: dict[str, Any], label: str) -> dict[str, float]:
    row = dict(report.get(label, {}))
    return {
        "precision": float(row.get("precision", 0.0)),
        "recall": float(row.get("recall", 0.0)),
        "f1": float(row.get("f1-score", 0.0)),
        "support": float(row.get("support", 0.0)),
    }


def _per_class_labels(report: dict[str, Any]) -> list[str]:
    excluded = {"accuracy", "macro avg", "weighted avg"}
    labels: list[str] = []
    for key, value in report.items():
        if key in excluded:
            continue
        if not isinstance(value, dict):
            continue
        labels.append(str(key))
    return sorted(labels)


def _render_heatmap(
    *, matrix: np.ndarray, labels: list[str], output_path: Path
) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix.astype(float),
        np.where(row_sums == 0, 1.0, row_sums),
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(normalized, cmap="magma")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized rate")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Phase-9 Class Failure Heatmap")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color="#f6f6f6" if normalized[row, col] > 0.45 else "#101010",
                fontsize=8,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _clip(text: str, *, width: int = 200) -> str:
    raw = " ".join(str(text).split())
    if len(raw) <= width:
        return raw
    return raw[: width - 3] + "..."


def _infer_examples(
    *,
    data_path: Path,
    model_family: str,
    hardest_labels: list[str],
    examples_per_class: int,
    project_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    if not data_path.exists():
        return {}

    frame = pd.read_parquet(data_path)
    required = {"question", "answer", "label"}
    if not required.issubset(frame.columns):
        return {}

    predictor = load_model(
        model_name=model_family, artifacts_root=project_root / "artifacts"
    )
    if not hasattr(predictor, "predict"):
        return {}

    payload = predictor.predict(
        frame["question"].astype(str).tolist(),
        frame["answer"].astype(str).tolist(),
        return_proba=True,
    )
    predictions = payload.get("predictions", [])
    confidence = payload.get("confidence", [])
    if not predictions:
        return {}

    rows = frame.copy()
    rows["predicted_label"] = predictions
    rows["confidence"] = confidence if confidence else [None] * len(rows)
    rows["is_error"] = rows["label"].astype(str) != rows["predicted_label"].astype(str)

    examples: dict[str, list[dict[str, Any]]] = {}
    for label in hardest_labels:
        failures = rows[(rows["label"].astype(str) == label) & rows["is_error"]]
        if failures.empty:
            continue
        selected = failures.sort_values("confidence", ascending=False).head(
            examples_per_class
        )
        snippets: list[dict[str, Any]] = []
        for _, row in selected.iterrows():
            snippets.append(
                {
                    "true_label": str(row["label"]),
                    "predicted_label": str(row["predicted_label"]),
                    "confidence": (
                        float(row["confidence"])
                        if row["confidence"] is not None
                        else None
                    ),
                    "question_snippet": _clip(str(row["question"])),
                    "answer_snippet": _clip(str(row["answer"]), width=260),
                }
            )
        examples[label] = snippets
    return examples


def _artifact_index(
    *,
    output_root: Path,
    generated_files: list[Path],
    source_data: Path,
    metadata: dict[str, Any],
    project_root: Path,
) -> Path:
    payload = {
        "phase": "phase9",
        "stage": "error_analysis",
        "source_data": normalize_path(source_data, base=project_root),
        "generated_at": utc_now_iso(),
        "generated_files": [
            normalize_path(path, base=output_root) for path in sorted(generated_files)
        ],
        "metadata": metadata,
    }
    return write_json(output_root / "artifact_index.json", payload)


def run(args: argparse.Namespace) -> list[Path]:
    project_root = Path(args.project_root).resolve()
    output_root = _resolve(args.output_root, project_root=project_root)
    output_root.mkdir(parents=True, exist_ok=True)

    data_path = _resolve(args.input, project_root=project_root)
    selected_summary_path = _resolve(
        args.selected_model_summary, project_root=project_root
    )
    phase5_summary_path = _resolve(args.phase5_summary, project_root=project_root)

    if not selected_summary_path.exists():
        raise FileNotFoundError(
            f"Missing selected model summary: {normalize_path(selected_summary_path, base=project_root)}"
        )
    if not phase5_summary_path.exists():
        raise FileNotFoundError(
            f"Missing phase-5 summary: {normalize_path(phase5_summary_path, base=project_root)}"
        )

    selected_summary = _load_json(selected_summary_path)
    phase5_summary = _load_json(phase5_summary_path)

    selected_family = str(selected_summary.get("best_model_family", ""))
    selected_root = _resolve(
        str(selected_summary.get("artifact_root", "")), project_root=project_root
    )
    selected_report_path = selected_root / "classification_report.json"
    selected_confusion_path = selected_root / "confusion_matrix.json"
    if not selected_report_path.exists() or not selected_confusion_path.exists():
        raise FileNotFoundError(
            "Selected model artifacts must include classification_report.json and "
            f"confusion_matrix.json under {normalize_path(selected_root, base=project_root)}"
        )

    baseline_family = str(phase5_summary.get("best_model_family", ""))
    baseline_root_raw = (
        phase5_summary.get("models", {}).get(baseline_family, {}).get("artifact_root")
    )
    baseline_root = (
        _resolve(str(baseline_root_raw), project_root=project_root)
        if baseline_root_raw
        else project_root / "artifacts" / "models" / "phase5" / baseline_family
    )
    baseline_report_path = baseline_root / "classification_report.json"
    if not baseline_report_path.exists():
        raise FileNotFoundError(
            "Baseline classification report not found: "
            + normalize_path(baseline_report_path, base=project_root)
        )

    selected_report = _load_json(selected_report_path)
    baseline_report = _load_json(baseline_report_path)
    selected_confusion = _load_json(selected_confusion_path)

    labels = list(selected_confusion.get("labels", []))
    matrix = np.array(selected_confusion.get("confusion_matrix", []), dtype=int)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Selected confusion matrix must be square")
    if matrix.shape[0] != len(labels):
        raise ValueError("Selected confusion matrix labels do not match matrix shape")

    metric_labels = sorted(set(_per_class_labels(selected_report)))

    per_class_delta: list[dict[str, Any]] = []
    for label in metric_labels:
        selected_row = _metric_row(selected_report, label)
        baseline_available = label in baseline_report
        baseline_row = (
            _metric_row(baseline_report, label)
            if baseline_available
            else {
                "precision": None,
                "recall": None,
                "f1": None,
                "support": None,
            }
        )
        per_class_delta.append(
            {
                "label": label,
                "selected_precision": selected_row["precision"],
                "selected_recall": selected_row["recall"],
                "selected_f1": selected_row["f1"],
                "baseline_precision": baseline_row["precision"],
                "baseline_recall": baseline_row["recall"],
                "baseline_f1": baseline_row["f1"],
                "precision_delta": (
                    selected_row["precision"] - baseline_row["precision"]
                    if baseline_available
                    else None
                ),
                "recall_delta": (
                    selected_row["recall"] - baseline_row["recall"]
                    if baseline_available
                    else None
                ),
                "f1_delta": (
                    selected_row["f1"] - baseline_row["f1"]
                    if baseline_available
                    else None
                ),
                "baseline_available": baseline_available,
                "support": selected_row["support"],
            }
        )

    route_rows: list[dict[str, Any]] = []
    total_errors = int(matrix.sum() - np.trace(matrix))
    for row_idx, true_label in enumerate(labels):
        true_errors = int(matrix[row_idx].sum() - matrix[row_idx, row_idx])
        if true_errors <= 0:
            continue
        for col_idx, predicted_label in enumerate(labels):
            if row_idx == col_idx:
                continue
            count = int(matrix[row_idx, col_idx])
            if count <= 0:
                continue
            route_rows.append(
                {
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": count,
                    "true_label_total_errors": true_errors,
                    "true_label_route_share": round(count / true_errors, 6),
                    "global_error_share": (
                        round(count / total_errors, 6) if total_errors > 0 else 0.0
                    ),
                }
            )
    routes = pd.DataFrame(
        route_rows,
        columns=[
            "true_label",
            "predicted_label",
            "count",
            "true_label_total_errors",
            "true_label_route_share",
            "global_error_share",
        ],
    )
    if not routes.empty:
        routes = routes.sort_values(
            ["count", "global_error_share"], ascending=False
        ).reset_index(drop=True)

    hardest = sorted(per_class_delta, key=lambda row: row["selected_recall"])[
        : args.hard_case_limit
    ]
    hardest_labels = [str(row["label"]) for row in hardest]
    examples = _infer_examples(
        data_path=data_path,
        model_family=selected_family,
        hardest_labels=hardest_labels,
        examples_per_class=args.examples_per_class,
        project_root=project_root,
    )

    error_summary = {
        "analysis_version": "phase9-error-analysis-v1",
        "generated_at": utc_now_iso(),
        "selected_model": {
            "family": selected_family,
            "artifact_root": normalize_path(selected_root, base=project_root),
            "classification_report": normalize_path(
                selected_report_path, base=project_root
            ),
            "confusion_matrix": normalize_path(
                selected_confusion_path, base=project_root
            ),
        },
        "baseline_model": {
            "family": baseline_family,
            "artifact_root": normalize_path(baseline_root, base=project_root),
            "classification_report": normalize_path(
                baseline_report_path, base=project_root
            ),
        },
        "total_misclassifications": total_errors,
        "top_misclassification_routes": route_rows[:3],
        "hardest_classes": hardest,
        "per_class_deltas": per_class_delta,
    }

    error_summary_path = write_json(output_root / "error_summary.json", error_summary)
    routes_path = output_root / "misclassification_routes.csv"
    routes.to_csv(routes_path, index=False)

    heatmap_path = output_root / "class_failure_heatmap.png"
    _render_heatmap(matrix=matrix, labels=labels, output_path=heatmap_path)

    hard_cases_path = output_root / "hard_cases.md"
    hard_case_lines = [
        "# Hard Cases",
        "",
        "This artifact summarizes the lowest-recall classes and representative model failures.",
        "",
    ]
    for item in hardest:
        label = str(item["label"])
        hard_case_lines.append(
            f"## {label} (selected recall={item['selected_recall']:.3f}, f1={item['selected_f1']:.3f})"
        )
        class_examples = examples.get(label, [])
        if not class_examples:
            hard_case_lines.append(
                "- No row-level evidence snippets were available for this class in current artifacts."
            )
            hard_case_lines.append("")
            continue
        for example in class_examples:
            confidence = example.get("confidence")
            confidence_str = (
                f"{confidence:.3f}" if isinstance(confidence, float) else "n/a"
            )
            hard_case_lines.append(
                f"- true=`{example['true_label']}` predicted=`{example['predicted_label']}` "
                f"confidence=`{confidence_str}`"
            )
            hard_case_lines.append(f"  - question: {example['question_snippet']}")
            hard_case_lines.append(f"  - answer: {example['answer_snippet']}")
        hard_case_lines.append("")
    hard_cases_path.write_text(
        "\n".join(hard_case_lines).strip() + "\n", encoding="utf-8"
    )

    generated = [
        error_summary_path,
        routes_path,
        hard_cases_path,
        heatmap_path,
    ]
    index_path = _artifact_index(
        output_root=output_root,
        generated_files=generated,
        source_data=data_path,
        metadata={
            "selected_family": selected_family,
            "baseline_family": baseline_family,
            "top_route_count": int(routes["count"].iloc[0]) if not routes.empty else 0,
        },
        project_root=project_root,
    )
    generated.append(index_path)

    print(
        f"wrote error summary: {normalize_path(error_summary_path, base=project_root)}"
    )
    print(
        f"wrote misclassification routes: {normalize_path(routes_path, base=project_root)}"
    )
    print(f"wrote hard cases: {normalize_path(hard_cases_path, base=project_root)}")
    print(f"wrote heatmap: {normalize_path(heatmap_path, base=project_root)}")
    print(f"wrote artifact index: {normalize_path(index_path, base=project_root)}")
    return generated


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
