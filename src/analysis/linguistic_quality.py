"""Readability, POS, and discourse analyses for phase 3."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import spacy
import textstat

from src.analysis.artifacts import ensure_phase3_layout, write_artifact_index

REQUIRED_COLUMNS = ("answer", "label")
HEDGING_MARKERS = {"maybe", "perhaps", "possibly", "likely", "might", "could"}
EVASIVE_MARKERS = {"cannot", "unable", "not sure", "no comment", "decline"}


def _validate(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return spacy.blank("en")


def _plot_metric(df: pd.DataFrame, metric: str, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    ordered = df.sort_values("label")
    plt.bar(ordered["label"], ordered[metric])
    plt.title(metric.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _readability(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_docs = 6000
    if frame.shape[0] > max_docs:
        frame = (
            frame.groupby("label", group_keys=False)
            .apply(
                lambda part: part.sample(
                    max(1, int(max_docs * len(part) / len(frame))), random_state=42
                )
            )
            .reset_index(drop=True)
        )

    rows = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        text = str(row["answer"])
        rows.append(
            {
                "row_id": idx,
                "label": str(row["label"]),
                "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
                "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
                "smog_index": float(textstat.smog_index(text)),
            }
        )
    raw = pd.DataFrame(rows)
    agg = (
        raw.groupby("label", dropna=False)[
            ["flesch_reading_ease", "flesch_kincaid_grade", "smog_index"]
        ]
        .mean()
        .reset_index()
        .sort_values("label")
        .reset_index(drop=True)
    )
    return raw, agg


def _pos(frame: pd.DataFrame, nlp) -> pd.DataFrame:
    max_docs = 3000
    if frame.shape[0] > max_docs:
        sampled = (
            frame.groupby("label", group_keys=False)
            .apply(
                lambda part: part.sample(
                    max(1, int(max_docs * len(part) / len(frame))), random_state=42
                )
            )
            .reset_index(drop=True)
        )
    else:
        sampled = frame

    counts_by_label: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}

    labels = sampled["label"].astype(str).tolist()
    texts = sampled["answer"].astype(str).tolist()
    pipeline_disabled = [
        name for name in ("ner", "parser", "lemmatizer") if name in nlp.pipe_names
    ]
    docs = nlp.pipe(texts, batch_size=128, disable=pipeline_disabled)
    for label, doc in zip(labels, docs):
        counts = counts_by_label.setdefault(label, {})
        total = totals.setdefault(label, 0)
        for token in doc:
            tag = token.pos_ or "X"
            counts[tag] = counts.get(tag, 0) + 1
            total += 1
        totals[label] = total

    records = []
    for label in sorted(counts_by_label):
        total = max(totals[label], 1)
        for tag, count in sorted(counts_by_label[label].items()):
            records.append(
                {
                    "label": label,
                    "pos": tag,
                    "count": int(count),
                    "proportion": float(count / total),
                }
            )

    return pd.DataFrame(records)


def _discourse(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in frame.assign(label=frame["label"].astype(str)).groupby(
        "label", dropna=False
    ):
        hedging_hits = 0
        evasive_hits = 0
        total_rows = int(group.shape[0])
        for text in group["answer"].astype(str).str.lower().tolist():
            hedging_hits += sum(1 for marker in HEDGING_MARKERS if marker in text)
            evasive_hits += sum(1 for marker in EVASIVE_MARKERS if marker in text)

        rows.append(
            {
                "label": label,
                "samples": total_rows,
                "hedging_rate": float(hedging_hits / max(total_rows, 1)),
                "evasive_marker_rate": float(evasive_hits / max(total_rows, 1)),
            }
        )

    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def run_linguistic_quality(
    frame: pd.DataFrame,
    output_root: str | Path,
    *,
    source_data: str | Path,
    sections: Iterable[str] | None = None,
) -> list[Path]:
    _validate(frame)
    sections_set = set(sections or ["readability", "pos", "discourse"])
    nlp = load_nlp()

    layout = ensure_phase3_layout(output_root)
    out_dir = layout["linguistic_quality"]
    generated: list[Path] = []

    interpretation_lines = ["# Linguistic Quality Interpretation", ""]

    if "readability" in sections_set:
        raw, agg = _readability(frame)
        raw_csv = out_dir / "readability_raw.csv"
        agg_csv = out_dir / "readability_summary.csv"
        raw.to_csv(raw_csv, index=False)
        agg.to_csv(agg_csv, index=False)
        generated.extend([raw_csv, agg_csv])

        for metric in ("flesch_reading_ease", "flesch_kincaid_grade", "smog_index"):
            plot_path = out_dir / f"{metric}.png"
            _plot_metric(agg, metric, plot_path)
            generated.append(plot_path)

        interpretation_lines.append("## Readability")
        interpretation_lines.append(
            "Readability metrics are aggregated by label for cross-group comparison."
        )
        interpretation_lines.append("")

    if "pos" in sections_set:
        pos_df = _pos(frame, nlp)
        pos_csv = out_dir / "pos_proportions.csv"
        pos_df.to_csv(pos_csv, index=False)
        generated.append(pos_csv)

        interpretation_lines.append("## POS")
        interpretation_lines.append(
            "POS token proportions are normalized per label to support fair comparison."
        )
        interpretation_lines.append("")

    if "discourse" in sections_set:
        discourse_df = _discourse(frame)
        discourse_csv = out_dir / "discourse_markers.csv"
        discourse_df.to_csv(discourse_csv, index=False)
        generated.append(discourse_csv)

        for metric in ("hedging_rate", "evasive_marker_rate"):
            plot_path = out_dir / f"{metric}.png"
            _plot_metric(discourse_df, metric, plot_path)
            generated.append(plot_path)

        interpretation_lines.append("## Discourse")
        interpretation_lines.append(
            "Hedging and evasive cue frequencies indicate potential response-style differences."
        )
        interpretation_lines.append("")

    interpretation = out_dir / "linguistic_interpretation.md"
    interpretation.write_text("\n".join(interpretation_lines), encoding="utf-8")
    generated.append(interpretation)

    write_artifact_index(
        output_root,
        stage="linguistic_quality",
        generated_files=generated,
        source_data=source_data,
        metadata={
            "sections": sorted(sections_set),
            "spacy_model": getattr(nlp, "meta", {}).get("name", "blank-en"),
        },
    )
    generated.append(Path(output_root) / "artifact_index.json")
    return generated
