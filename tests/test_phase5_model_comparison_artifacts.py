from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _prepared_data(path: Path) -> None:
    rows = []
    for idx in range(80):
        label = "evasive" if idx % 2 == 0 else "non_evasive"
        rows.append(
            {
                "question": f"Question {idx}",
                "answer": f"Answer {idx} with phrase {idx % 5}",
                "label": label,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_phase5_comparison_outputs_exist_and_are_consistent(tmp_path: Path) -> None:
    data_path = tmp_path / "prepared.parquet"
    output_root = tmp_path / "phase5"
    _prepared_data(data_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/run_classical_baselines.py",
            "--input",
            str(data_path),
            "--output-root",
            str(output_root),
            "--families",
            "all",
            "--compare",
            "--tracking-uri",
            f"file:{tmp_path / 'mlruns'}",
            "--experiment-name",
            "phase5-test",
        ],
        check=True,
    )

    comparison_root = output_root / "model_comparison"
    ranking_path = comparison_root / "model_ranking.csv"
    per_class_path = comparison_root / "per_class_f1_comparison.csv"
    summary_path = comparison_root / "summary.json"

    assert ranking_path.exists()
    assert per_class_path.exists()
    assert summary_path.exists()
    assert (comparison_root / "macro_f1_by_model.png").exists()
    assert (comparison_root / "per_class_f1_delta_heatmap.png").exists()

    ranking = pd.read_csv(ranking_path)
    assert {"logreg", "tree", "boosting"}.issubset(set(ranking["model_family"]))

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "metric_deltas_vs_best" in summary
    assert set(summary["families"]) == set(ranking["model_family"])
