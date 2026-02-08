"""Deterministic question-answer semantic similarity analysis for phase 4."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.analysis.artifacts import ensure_phase4_layout, write_phase4_artifact_index

REQUIRED_COLUMNS = ("question", "answer", "label")


def _validate_input(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _safe_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _build_similarity_rows(frame: pd.DataFrame) -> pd.DataFrame:
    questions = _safe_text(frame["question"])
    answers = _safe_text(frame["answer"])

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    combined = pd.concat([questions, answers], axis=0).tolist()
    matrix = vectorizer.fit_transform(combined)

    question_matrix = matrix[: len(frame)]
    answer_matrix = matrix[len(frame) :]
    similarities = [
        float(cosine_similarity(question_matrix[i], answer_matrix[i])[0, 0])
        for i in range(len(frame))
    ]

    rows = frame.copy()
    rows["question"] = questions
    rows["answer"] = answers
    rows["label"] = rows["label"].astype(str)
    rows["semantic_similarity"] = similarities
    return rows


def _plot_similarity_by_label(summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    summary.set_index("label")["mean_similarity"].plot(kind="bar")
    plt.ylabel("mean cosine similarity")
    plt.title("Q-A semantic similarity by label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _build_hypothesis_summary(summary: pd.DataFrame) -> dict:
    top = summary.sort_values("mean_similarity", ascending=False).reset_index(drop=True)
    highest = top.iloc[0]
    lowest = top.iloc[-1]
    delta = float(highest["mean_similarity"] - lowest["mean_similarity"])
    confidence_note = (
        "moderate_signal" if abs(delta) >= 0.05 else "weak_signal_needs_follow_up"
    )
    return {
        "hypotheses": [
            {
                "id": "H1",
                "name": "relevance_deflection",
                "direction": "lower_similarity_implies_more_evasion",
                "confidence_note": confidence_note,
                "finding": (
                    f"Highest alignment label: {highest['label']} ({highest['mean_similarity']:.4f}); "
                    f"lowest alignment label: {lowest['label']} ({lowest['mean_similarity']:.4f}); "
                    f"delta={delta:.4f}."
                ),
            },
            {
                "id": "H2",
                "name": "vague_response_behavior",
                "direction": "broader_variance_implies_less_question_targeting",
                "finding": "Similarity standard deviations are reported per label for interpretation.",
            },
        ],
        "notes": "Deterministic TF-IDF + cosine pipeline used as script-first semantic baseline.",
    }


def run_qa_semantic(
    frame: pd.DataFrame,
    output_root: str | Path,
    *,
    source_data: str | Path,
    emit_hypothesis_summary: bool = True,
) -> list[Path]:
    _validate_input(frame)
    layout = ensure_phase4_layout(output_root)
    out_dir = layout["semantic_similarity"]

    rows = _build_similarity_rows(frame)
    generated: list[Path] = []

    rows_path = out_dir / "qa_similarity_rows.parquet"
    rows.to_parquet(rows_path, index=False)
    generated.append(rows_path)

    summary = (
        rows.groupby("label", dropna=False)["semantic_similarity"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_similarity",
                "median": "median_similarity",
                "std": "std_similarity",
                "count": "n",
            }
        )
        .sort_values("label")
        .reset_index(drop=True)
    )

    summary_csv = out_dir / "semantic_similarity_by_label.csv"
    summary_json = out_dir / "semantic_similarity_by_label.json"
    summary.to_csv(summary_csv, index=False)
    summary.to_json(summary_json, orient="records", indent=2)
    generated.extend([summary_csv, summary_json])

    plot_path = out_dir / "semantic_similarity_by_label.png"
    _plot_similarity_by_label(summary, plot_path)
    generated.append(plot_path)

    hypothesis_path = out_dir / "hypothesis_summary.json"
    hypothesis_md = out_dir / "hypothesis_summary.md"
    hypothesis_summary = _build_hypothesis_summary(summary)
    if emit_hypothesis_summary:
        hypothesis_path.write_text(json.dumps(hypothesis_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_lines = ["# Semantic Similarity Hypothesis Summary", ""]
        for item in hypothesis_summary["hypotheses"]:
            md_lines.extend([f"## {item['id']}: {item['name']}", item["finding"], ""])
        md_lines.append(hypothesis_summary["notes"])
        hypothesis_md.write_text("\n".join(md_lines), encoding="utf-8")
        generated.extend([hypothesis_path, hypothesis_md])

    write_phase4_artifact_index(
        output_root,
        stage="semantic_similarity",
        generated_files=generated,
        source_data=source_data,
        metadata={
            "analysis_version": "v1",
            "model": "tfidf_cosine_ngram_1_2",
            "hypothesis_summary": "semantic_similarity/hypothesis_summary.json",
        },
    )
    generated.append(Path(output_root) / "artifact_index.json")
    return generated
