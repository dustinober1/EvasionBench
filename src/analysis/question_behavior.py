"""Deterministic question taxonomy and behavior metrics for phase 4."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.artifacts import ensure_phase4_layout, write_phase4_artifact_index

REQUIRED_COLUMNS = ("question", "answer", "label")
TAXONOMY = ("yes_no", "comparison", "procedural", "opinion", "factual", "multi_part", "other")


REFUSAL_PATTERNS = (
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bi do not\b",
    r"\bi don't\b",
    r"\bnot able\b",
    r"\bunable\b",
    r"\bcan't help\b",
)


def _validate_input(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _safe_text(value: object) -> str:
    return str(value or "").strip().lower()


def classify_question(question: str) -> str:
    q = _safe_text(question)
    if not q:
        return "other"
    if "?" in q and (" and " in q or ";" in q):
        return "multi_part"
    if re.search(r"^(is|are|was|were|do|does|did|can|could|should|would|will|has|have)\b", q):
        return "yes_no"
    if "compare" in q or "difference" in q or " vs " in q:
        return "comparison"
    if re.search(r"\b(how|steps|process|procedure)\b", q):
        return "procedural"
    if re.search(r"\b(opinion|think|feel|believe|should)\b", q):
        return "opinion"
    if re.search(r"\b(what|who|when|where|why)\b", q):
        return "factual"
    return "other"


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", _safe_text(text)))


def _jaccard(a: str, b: str) -> float:
    left = _token_set(a)
    right = _token_set(b)
    if not left and not right:
        return 0.0
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def _refusal_rate(series: pd.Series) -> pd.Series:
    pattern = "|".join(REFUSAL_PATTERNS)
    return series.fillna("").astype(str).str.lower().str.contains(pattern, regex=True)


def _plot_behavior(comparison: pd.DataFrame, out_path: Path) -> None:
    pivot = comparison.pivot(index="question_type", columns="label", values="mean_answer_length").fillna(0.0)
    pivot.plot(kind="bar", figsize=(11, 5))
    plt.ylabel("mean answer length")
    plt.title("Answer length by question type and label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_question_behavior(
    frame: pd.DataFrame,
    output_root: str | Path,
    *,
    source_data: str | Path,
    emit_assignments: bool = False,
    emit_summary: bool = True,
) -> list[Path]:
    _validate_input(frame)
    layout = ensure_phase4_layout(output_root)
    out_dir = layout["question_behavior"]
    generated: list[Path] = []

    rows = frame.copy()
    rows["question"] = rows["question"].fillna("").astype(str)
    rows["answer"] = rows["answer"].fillna("").astype(str)
    rows["label"] = rows["label"].astype(str)
    rows["question_type"] = rows["question"].map(classify_question)
    rows["answer_length"] = rows["answer"].str.len()
    rows["semantic_alignment_jaccard"] = [
        _jaccard(q, a) for q, a in zip(rows["question"], rows["answer"])
    ]
    rows["refusal_marker"] = _refusal_rate(rows["answer"]).astype(int)

    assignment_path = out_dir / "question_assignments.parquet"
    if emit_assignments:
        rows[
            [
                "question",
                "answer",
                "label",
                "question_type",
                "answer_length",
                "semantic_alignment_jaccard",
                "refusal_marker",
            ]
        ].to_parquet(assignment_path, index=False)
        generated.append(assignment_path)

    metrics = (
        rows.groupby(["question_type", "label"], dropna=False)
        .agg(
            mean_answer_length=("answer_length", "mean"),
            mean_alignment=("semantic_alignment_jaccard", "mean"),
            refusal_rate=("refusal_marker", "mean"),
            n=("question", "count"),
        )
        .reset_index()
        .sort_values(["question_type", "label"])
        .reset_index(drop=True)
    )

    metrics_csv = out_dir / "question_behavior_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)
    generated.append(metrics_csv)

    summary = (
        metrics.groupby("question_type", dropna=False)
        .agg(
            labels_compared=("label", "nunique"),
            mean_answer_length=("mean_answer_length", "mean"),
            mean_alignment=("mean_alignment", "mean"),
            mean_refusal_rate=("refusal_rate", "mean"),
        )
        .reset_index()
        .sort_values("question_type")
        .reset_index(drop=True)
    )

    summary_csv = out_dir / "question_behavior_comparison.csv"
    summary.to_csv(summary_csv, index=False)
    generated.append(summary_csv)

    per_type_spread = (
        metrics.groupby("question_type", dropna=False)["refusal_rate"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "min_refusal_rate", "max": "max_refusal_rate"})
    )
    per_type_spread["refusal_rate_spread"] = (
        per_type_spread["max_refusal_rate"] - per_type_spread["min_refusal_rate"]
    )

    plot_path = out_dir / "question_behavior_by_label.png"
    _plot_behavior(metrics, plot_path)
    generated.append(plot_path)

    taxonomy_path = out_dir / "taxonomy_metadata.json"
    taxonomy_path.write_text(
        json.dumps(
            {
                "taxonomy": list(TAXONOMY),
                "version": "v1",
                "rules": "deterministic regex/keyword based",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generated.append(taxonomy_path)

    summary_json = out_dir / "question_behavior_summary.json"
    summary_md = out_dir / "question_behavior_summary.md"
    if emit_summary:
        payload = {
            "taxonomy_version": "v1",
            "hypothesis_notes": [
                "Higher refusal rates in specific question types may indicate evasive behavior strategies.",
                "Lower alignment and shorter answers can suggest deflection for targeted question categories.",
            ],
            "tracked_metrics": [
                "mean_answer_length",
                "mean_alignment",
                "refusal_rate",
            ],
            "refusal_spread_by_question_type": per_type_spread.to_dict(orient="records"),
        }
        summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary_md.write_text(
            "# Question Behavior Summary\n\n"
            "- Deterministic taxonomy assignment across seven categories.\n"
            "- Metrics compare answer length, alignment, and refusal markers by label.\n"
            "- Outputs are hypothesis-linked for report inclusion.\n",
            encoding="utf-8",
        )
        generated.extend([summary_json, summary_md])

    write_phase4_artifact_index(
        output_root,
        stage="question_behavior",
        generated_files=generated,
        source_data=source_data,
        metadata={
            "analysis_version": "v1",
            "model": "rule_taxonomy_v1",
            "hypothesis_summary": "question_behavior/question_behavior_summary.json",
        },
    )
    generated.append(Path(output_root) / "artifact_index.json")
    return generated
