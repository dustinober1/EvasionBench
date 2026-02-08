"""Core EDA and length-statistics analyses for phase 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from src.analysis.artifacts import ensure_phase3_layout, write_artifact_index

REQUIRED_COLUMNS = ("question", "answer", "label")


def _validate_input(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _write_csv_json(frame: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    frame.to_csv(csv_path, index=False)
    frame.to_json(json_path, orient="records", indent=2)


def _save_plot(series: pd.Series, out_path: Path, title: str, xlabel: str) -> None:
    plt.figure(figsize=(8, 4))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _length_tests(frame: pd.DataFrame) -> dict:
    frame = frame.copy()
    frame["question_length"] = frame["question"].astype(str).str.len()
    frame["answer_length"] = frame["answer"].astype(str).str.len()

    result: dict[str, dict] = {}
    labels = sorted(frame["label"].astype(str).unique().tolist())

    for measure in ("question_length", "answer_length"):
        samples = [
            frame.loc[frame["label"].astype(str) == label, measure].astype(float).tolist()
            for label in labels
        ]
        usable = [sample for sample in samples if len(sample) > 0]
        if len(usable) < 2:
            result[measure] = {
                "labels": labels,
                "kruskal": {"p_value": None, "statistic": None},
                "pairwise_mannwhitney": [],
                "interpretation": "Insufficient groups for significance testing.",
            }
            continue

        try:
            k_stat, k_p = kruskal(*usable)
        except ValueError:
            # SciPy raises when all observations are identical across groups.
            k_stat, k_p = 0.0, 1.0
        pairwise = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                left = frame.loc[frame["label"].astype(str) == labels[i], measure].astype(float)
                right = frame.loc[frame["label"].astype(str) == labels[j], measure].astype(float)
                if left.empty or right.empty:
                    continue
                try:
                    mw_stat, mw_p = mannwhitneyu(left, right, alternative="two-sided")
                except ValueError:
                    mw_stat, mw_p = 0.0, 1.0
                pairwise.append(
                    {
                        "left_label": labels[i],
                        "right_label": labels[j],
                        "left_n": int(left.shape[0]),
                        "right_n": int(right.shape[0]),
                        "statistic": float(mw_stat),
                        "p_value": float(mw_p),
                    }
                )

        interpretation = (
            "Statistically significant differences detected across labels."
            if k_p < 0.05
            else "No statistically significant differences detected across labels."
        )
        result[measure] = {
            "labels": labels,
            "kruskal": {"p_value": float(k_p), "statistic": float(k_stat)},
            "pairwise_mannwhitney": pairwise,
            "interpretation": interpretation,
        }

    return result


def run_core_stats(
    frame: pd.DataFrame,
    output_root: str | Path,
    *,
    source_data: str | Path,
    sections: Iterable[str] | None = None,
) -> list[Path]:
    _validate_input(frame)
    sections_set = set(sections or ["quality", "lengths"])

    layout = ensure_phase3_layout(output_root)
    out_dir = layout["core_stats"]
    generated: list[Path] = []

    if "quality" in sections_set:
        quality = pd.DataFrame(
            {
                "metric": [
                    "row_count",
                    "missing_question_rate",
                    "missing_answer_rate",
                    "duplicate_answer_rate",
                ],
                "value": [
                    int(frame.shape[0]),
                    float((frame["question"].astype(str).str.strip() == "").mean()),
                    float((frame["answer"].astype(str).str.strip() == "").mean()),
                    float(frame["answer"].astype(str).duplicated().mean()),
                ],
            }
        )
        quality_csv = out_dir / "quality_metrics.csv"
        quality_json = out_dir / "quality_metrics.json"
        _write_csv_json(quality, quality_csv, quality_json)
        generated.extend([quality_csv, quality_json])

        class_dist = (
            frame.assign(label=frame["label"].astype(str))
            .groupby("label", dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "label"], ascending=[False, True])
            .reset_index(drop=True)
        )
        class_csv = out_dir / "class_distribution.csv"
        class_json = out_dir / "class_distribution.json"
        _write_csv_json(class_dist, class_csv, class_json)
        generated.extend([class_csv, class_json])

        class_plot = out_dir / "class_distribution.png"
        _save_plot(class_dist.set_index("label")["count"], class_plot, "Label Distribution", "label")
        generated.append(class_plot)

    if "lengths" in sections_set:
        lengths = frame.assign(
            label=frame["label"].astype(str),
            question_length=frame["question"].astype(str).str.len(),
            answer_length=frame["answer"].astype(str).str.len(),
        )
        summary = (
            lengths.groupby("label", dropna=False)[["question_length", "answer_length"]]
            .agg(["mean", "median", "std", "min", "max"])
            .reset_index()
        )
        summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.to_flat_index()]
        summary = summary.sort_values("label", ascending=True).reset_index(drop=True)
        summary_csv = out_dir / "length_summary.csv"
        summary_json = out_dir / "length_summary.json"
        _write_csv_json(summary, summary_csv, summary_json)
        generated.extend([summary_csv, summary_json])

        test_result = _length_tests(frame)
        tests_path = out_dir / "length_tests.json"
        tests_path.write_text(json.dumps(test_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        generated.append(tests_path)

        interpretation_path = out_dir / "length_interpretation.md"
        lines = ["# Length Test Interpretation", ""]
        for measure, payload in test_result.items():
            lines.append(f"## {measure}")
            lines.append(payload["interpretation"])
            lines.append("")
        interpretation_path.write_text("\n".join(lines), encoding="utf-8")
        generated.append(interpretation_path)

        for measure in ("question_length", "answer_length"):
            plot = out_dir / f"{measure}_by_label.png"
            grouped = lengths.groupby("label", dropna=False)[measure].mean().sort_values(ascending=False)
            _save_plot(grouped, plot, f"Mean {measure} by label", "label")
            generated.append(plot)

    write_artifact_index(
        output_root,
        stage="core_stats",
        generated_files=generated,
        source_data=source_data,
        metadata={"sections": sorted(sections_set)},
    )
    generated.append(Path(output_root) / "artifact_index.json")
    return generated
