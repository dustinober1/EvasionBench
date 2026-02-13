"""Generate conditional phase-9 exploratory analysis artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.question_behavior import classify_question
from src.inference import load_model
from src.reporting import normalize_path, utc_now_iso, write_json

TEMPORAL_COLUMNS = ("date", "timestamp", "quarter", "fiscal_quarter", "year", "period")
SEGMENT_COLUMNS = ("sector", "company", "company_name", "industry", "category")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/evasionbench_prepared.parquet",
        help="Prepared dataset path.",
    )
    parser.add_argument(
        "--selected-model-summary",
        default="artifacts/models/phase8/selected_model.json",
        help="Phase-8 selected model summary path.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/analysis/phase9/exploration",
        help="Output directory for exploration artifacts.",
    )
    parser.add_argument(
        "--project-root",
        default=str(ROOT),
        help="Project root used for normalized paths.",
    )
    return parser.parse_args()


def _resolve(path_like: str, *, project_root: Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _predict_labels(
    *,
    frame: pd.DataFrame,
    selected_summary: Mapping[str, Any],
    project_root: Path,
) -> tuple[pd.Series | None, str]:
    family = str(selected_summary.get("best_model_family", "")).strip()
    if not family:
        return None, "selected model family missing"

    predictor = load_model(model_name=family, artifacts_root=project_root / "artifacts")
    if not hasattr(predictor, "predict"):
        return None, "predictor unavailable"

    payload = predictor.predict(
        frame["question"].astype(str).tolist(),
        frame["answer"].astype(str).tolist(),
        return_proba=False,
    )
    predictions = payload.get("predictions", [])
    if not predictions:
        return None, "empty prediction output"
    return pd.Series(predictions, index=frame.index), "ok"


def _build_temporal_summary(
    *,
    frame: pd.DataFrame,
    predictions: pd.Series | None,
) -> dict[str, Any]:
    available = [column for column in TEMPORAL_COLUMNS if column in frame.columns]
    if not available:
        return {
            "status": "skipped",
            "reason": "Missing temporal columns required for time-based slicing",
            "required_any_of": list(TEMPORAL_COLUMNS),
            "available_columns": list(frame.columns),
        }

    rows = frame.copy()
    column = available[0]
    if column in {"date", "timestamp"}:
        parsed = pd.to_datetime(rows[column], errors="coerce", utc=True)
        period = parsed.dt.to_period("Q").astype(str)
        period = period.where(~parsed.isna(), "unknown")
    elif column == "year":
        period = rows[column].astype(str).str.strip()
    else:
        period = rows[column].astype(str).str.strip()
    rows["period"] = period

    rows["is_evasive"] = (
        rows["label"].astype(str).isin(["intermediate", "fully_evasive", "evasive"])
    )

    summary = (
        rows.groupby("period", dropna=False)
        .agg(
            n=("label", "count"),
            evasion_rate=("is_evasive", "mean"),
        )
        .reset_index()
        .sort_values("period")
    )

    result: dict[str, Any] = {
        "status": "generated",
        "grouping_column": column,
        "n_periods": int(summary.shape[0]),
        "periods": summary.to_dict(orient="records"),
    }

    if predictions is None:
        result["model_performance_drift"] = {
            "status": "skipped",
            "reason": "Predictions unavailable for drift computation",
        }
        return result

    rows["predicted_label"] = predictions
    drift_rows: list[dict[str, Any]] = []
    for period_value, block in rows.groupby("period", dropna=False):
        y_true = block["label"].astype(str)
        y_pred = block["predicted_label"].astype(str)
        accuracy = float((y_true == y_pred).mean())
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        drift_rows.append(
            {
                "period": str(period_value),
                "n": int(block.shape[0]),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )
    result["model_performance_drift"] = {
        "status": "generated",
        "rows": sorted(drift_rows, key=lambda row: row["period"]),
    }
    return result


def _build_segment_summary(
    *,
    frame: pd.DataFrame,
    predictions: pd.Series | None,
) -> dict[str, Any]:
    available = [column for column in SEGMENT_COLUMNS if column in frame.columns]
    if not available:
        return {
            "status": "skipped",
            "reason": "Missing segment columns required for contextual slicing",
            "required_any_of": list(SEGMENT_COLUMNS),
            "available_columns": list(frame.columns),
        }

    column = available[0]
    rows = frame.copy()
    rows["segment"] = rows[column].fillna("unknown").astype(str)

    distribution = (
        rows.groupby(["segment", "label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["segment", "count"], ascending=[True, False])
    )

    result: dict[str, Any] = {
        "status": "generated",
        "grouping_column": column,
        "segment_distribution": distribution.to_dict(orient="records")[:200],
    }

    if predictions is None:
        result["segment_confusion"] = {
            "status": "skipped",
            "reason": "Predictions unavailable for segment confusion analysis",
        }
        return result

    rows["predicted_label"] = predictions
    confusion = (
        rows.groupby(["segment", "label", "predicted_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["segment", "count"], ascending=[True, False])
    )
    result["segment_confusion"] = {
        "status": "generated",
        "rows": confusion.to_dict(orient="records")[:400],
    }
    return result


def _build_question_intent_error_map(
    *,
    frame: pd.DataFrame,
    predictions: pd.Series | None,
) -> pd.DataFrame:
    columns = [
        "question_type",
        "n_samples",
        "n_errors",
        "error_rate",
        "intermediate_to_fully_evasive",
        "fully_evasive_to_intermediate",
    ]
    rows = frame.copy()
    rows["question_type"] = rows["question"].astype(str).map(classify_question)

    if predictions is None:
        return pd.DataFrame(columns=columns)

    rows["predicted_label"] = predictions.astype(str)
    rows["label"] = rows["label"].astype(str)
    rows["is_error"] = rows["label"] != rows["predicted_label"]
    rows["intermediate_to_fully_evasive"] = (rows["label"] == "intermediate") & (
        rows["predicted_label"] == "fully_evasive"
    )
    rows["fully_evasive_to_intermediate"] = (rows["label"] == "fully_evasive") & (
        rows["predicted_label"] == "intermediate"
    )

    summary = (
        rows.groupby("question_type", dropna=False)
        .agg(
            n_samples=("label", "count"),
            n_errors=("is_error", "sum"),
            intermediate_to_fully_evasive=("intermediate_to_fully_evasive", "sum"),
            fully_evasive_to_intermediate=("fully_evasive_to_intermediate", "sum"),
        )
        .reset_index()
        .sort_values("n_errors", ascending=False)
    )
    summary["error_rate"] = summary["n_errors"] / summary["n_samples"]
    return summary[columns]


def _artifact_index(
    *,
    output_root: Path,
    source_data: Path,
    generated_files: list[Path],
    temporal_status: str,
    segment_status: str,
    question_intent_status: str,
    project_root: Path,
) -> Path:
    payload = {
        "phase": "phase9",
        "stage": "exploration",
        "generated_at": utc_now_iso(),
        "source_data": normalize_path(source_data, base=project_root),
        "generated_files": [
            normalize_path(path, base=output_root) for path in sorted(generated_files)
        ],
        "metadata": {
            "temporal_status": temporal_status,
            "segment_status": segment_status,
            "question_intent_status": question_intent_status,
        },
    }
    return write_json(output_root / "artifact_index.json", payload)


def _write_skip_bundle(
    *,
    output_root: Path,
    input_path: Path,
    project_root: Path,
    reason: str,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    temporal = {
        "status": "skipped",
        "reason": reason,
        "prediction_status": "skipped",
    }
    segment = {
        "status": "skipped",
        "reason": reason,
        "prediction_status": "skipped",
    }
    temporal_path = write_json(output_root / "temporal_summary.json", temporal)
    segment_path = write_json(output_root / "segment_summary.json", segment)

    question_intent_path = output_root / "question_intent_error_map.csv"
    pd.DataFrame(
        columns=[
            "question_type",
            "n_samples",
            "n_errors",
            "error_rate",
            "intermediate_to_fully_evasive",
            "fully_evasive_to_intermediate",
        ]
    ).to_csv(question_intent_path, index=False)

    generated = [temporal_path, segment_path, question_intent_path]
    index_path = _artifact_index(
        output_root=output_root,
        source_data=input_path,
        generated_files=generated,
        temporal_status="skipped",
        segment_status="skipped",
        question_intent_status="skipped",
        project_root=project_root,
    )
    payload = _load_json(index_path)
    payload["metadata"]["reason"] = reason
    write_json(index_path, payload)
    generated.append(index_path)
    return generated


def run(args: argparse.Namespace) -> list[Path]:
    project_root = Path(args.project_root).resolve()
    input_path = _resolve(args.input, project_root=project_root)
    output_root = _resolve(args.output_root, project_root=project_root)
    selected_summary_path = _resolve(
        args.selected_model_summary, project_root=project_root
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        return _write_skip_bundle(
            output_root=output_root,
            input_path=input_path,
            project_root=project_root,
            reason=(
                "Missing prepared dataset: "
                + normalize_path(input_path, base=project_root)
            ),
        )

    frame = pd.read_parquet(input_path)
    required = {"question", "answer", "label"}
    if not required.issubset(frame.columns):
        return _write_skip_bundle(
            output_root=output_root,
            input_path=input_path,
            project_root=project_root,
            reason="Prepared dataset must include columns: question, answer, label",
        )

    selected_summary: dict[str, Any] = {}
    if selected_summary_path.exists():
        selected_summary = _load_json(selected_summary_path)

    predictions, prediction_status = _predict_labels(
        frame=frame,
        selected_summary=selected_summary,
        project_root=project_root,
    )

    temporal_summary = _build_temporal_summary(frame=frame, predictions=predictions)
    temporal_summary["prediction_status"] = prediction_status
    temporal_path = write_json(output_root / "temporal_summary.json", temporal_summary)

    segment_summary = _build_segment_summary(frame=frame, predictions=predictions)
    segment_summary["prediction_status"] = prediction_status
    segment_path = write_json(output_root / "segment_summary.json", segment_summary)

    question_intent = _build_question_intent_error_map(
        frame=frame,
        predictions=predictions,
    )
    question_intent_path = output_root / "question_intent_error_map.csv"
    question_intent.to_csv(question_intent_path, index=False)

    generated = [temporal_path, segment_path, question_intent_path]
    index_path = _artifact_index(
        output_root=output_root,
        source_data=input_path,
        generated_files=generated,
        temporal_status=str(temporal_summary.get("status", "unknown")),
        segment_status=str(segment_summary.get("status", "unknown")),
        question_intent_status=(
            "generated" if not question_intent.empty else "skipped"
        ),
        project_root=project_root,
    )
    generated.append(index_path)

    print(f"wrote temporal summary: {normalize_path(temporal_path, base=project_root)}")
    print(f"wrote segment summary: {normalize_path(segment_path, base=project_root)}")
    print(
        "wrote question-intent map: "
        + normalize_path(question_intent_path, base=project_root)
    )
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
