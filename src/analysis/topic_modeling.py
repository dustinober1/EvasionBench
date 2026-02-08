"""Deterministic topic modeling analysis for phase 4."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src.analysis.artifacts import ensure_phase4_layout, write_phase4_artifact_index

REQUIRED_COLUMNS = ("answer", "label")


def _validate_input(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _safe_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _top_terms(model: NMF, feature_names: list[str], n_terms: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for topic_idx, topic_vec in enumerate(model.components_):
        top_ids = np.argsort(topic_vec)[::-1][:n_terms]
        for rank, fid in enumerate(top_ids, start=1):
            rows.append(
                {
                    "topic": int(topic_idx),
                    "rank": int(rank),
                    "term": feature_names[int(fid)],
                    "weight": float(topic_vec[int(fid)]),
                }
            )
    return pd.DataFrame(rows).sort_values(["topic", "rank"]).reset_index(drop=True)


def _plot_topic_prevalence(prevalence: pd.DataFrame, out_path: Path) -> None:
    pivot = prevalence.pivot(index="topic", columns="label", values="mean_weight").fillna(0.0)
    pivot = pivot.sort_index()
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("mean topic weight")
    plt.title("Topic prevalence by label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_topic_modeling(
    frame: pd.DataFrame,
    output_root: str | Path,
    *,
    source_data: str | Path,
    topics: int = 8,
    seed: int = 42,
    emit_summary: bool = True,
) -> list[Path]:
    _validate_input(frame)

    layout = ensure_phase4_layout(output_root)
    out_dir = layout["topic_modeling"]
    generated: list[Path] = []

    answers = _safe_text(frame["answer"])
    labels = frame["label"].astype(str)

    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", min_df=2, max_df=0.95)
    dtm = vectorizer.fit_transform(answers)

    n_topics = max(2, min(int(topics), max(2, min(dtm.shape[0] - 1, dtm.shape[1] - 1))))
    model = NMF(n_components=n_topics, random_state=int(seed), init="nndsvda", max_iter=400)
    doc_topics = model.fit_transform(dtm)

    feature_names = vectorizer.get_feature_names_out().tolist()
    top_terms = _top_terms(model, feature_names)
    top_terms_path = out_dir / "topic_top_terms.csv"
    top_terms.to_csv(top_terms_path, index=False)
    generated.append(top_terms_path)

    doc_df = pd.DataFrame(doc_topics, columns=[f"topic_{i}" for i in range(doc_topics.shape[1])])
    doc_df.insert(0, "label", labels.to_list())
    doc_df.insert(0, "row_id", list(range(len(doc_df))))
    doc_topics_path = out_dir / "document_topics.parquet"
    doc_df.to_parquet(doc_topics_path, index=False)
    generated.append(doc_topics_path)

    prevalence_rows: list[dict[str, object]] = []
    for topic in range(doc_topics.shape[1]):
        col = f"topic_{topic}"
        grouped = doc_df.groupby("label", dropna=False)[col].mean().sort_index()
        for label, value in grouped.items():
            prevalence_rows.append(
                {"topic": int(topic), "label": str(label), "mean_weight": float(value)}
            )
    prevalence = pd.DataFrame(prevalence_rows).sort_values(["topic", "label"]).reset_index(drop=True)

    prevalence_csv = out_dir / "topic_prevalence_by_label.csv"
    prevalence.to_csv(prevalence_csv, index=False)
    generated.append(prevalence_csv)

    prevalence_plot = out_dir / "topic_prevalence_by_label.png"
    _plot_topic_prevalence(prevalence, prevalence_plot)
    generated.append(prevalence_plot)

    summary_payload = {
        "n_topics": int(n_topics),
        "seed": int(seed),
        "vectorizer": {
            "min_df": 2,
            "max_df": 0.95,
            "stop_words": "english",
            "ngram_range": [1, 1],
        },
        "hypothesis_notes": [
            "Topics with higher prevalence in evasive labels may capture deflection or uncertainty patterns.",
            "Compare prevalence deltas across labels to evaluate behavior-specific topical focus.",
        ],
    }
    summary_json = out_dir / "topic_summary.json"
    summary_md = out_dir / "topic_summary.md"
    if emit_summary:
        summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary_md.write_text(
            "# Topic Modeling Summary\n\n"
            f"- Topics: {n_topics}\n"
            f"- Seed: {seed}\n"
            "- Output includes top terms and label-stratified prevalence for hypothesis interpretation.\n",
            encoding="utf-8",
        )
        generated.extend([summary_json, summary_md])

    config_path = out_dir / "model_config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": int(seed),
                "requested_topics": int(topics),
                "resolved_topics": int(n_topics),
                "model": "nmf",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generated.append(config_path)

    write_phase4_artifact_index(
        output_root,
        stage="topic_modeling",
        generated_files=generated,
        source_data=source_data,
        metadata={
            "analysis_version": "v1",
            "model": "nmf_tfidf",
            "hypothesis_summary": "topic_modeling/topic_summary.json",
        },
    )
    generated.append(Path(output_root) / "artifact_index.json")
    return generated
