from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _prepared_data(path: Path) -> None:
    rows = []
    for idx in range(180):
        if idx % 10 == 0:
            label = "fully_evasive"
            answer = "We cannot discuss that guidance right now"
        elif idx % 2 == 0:
            label = "direct"
            answer = "Revenue increased and margins improved significantly"
        else:
            label = "intermediate"
            answer = "We are seeing progress, but there are several moving parts"

        rows.append(
            {
                "question": f"Question {idx}",
                "answer": answer,
                "label": label,
            }
        )

    pd.DataFrame(rows).to_parquet(path, index=False)


def test_phase8_optimizer_writes_contract_and_selection(tmp_path: Path) -> None:
    data_path = tmp_path / "prepared.parquet"
    output_root = tmp_path / "phase8"
    _prepared_data(data_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/run_model_optimization.py",
            "--input",
            str(data_path),
            "--output-root",
            str(output_root),
            "--families",
            "logreg",
            "--cv-folds",
            "3",
            "--max-trials",
            "4",
            "--calibration-method",
            "none",
            "--accuracy-floor",
            "1.1",
            "--tracking-uri",
            f"file:{tmp_path / 'mlruns'}",
            "--experiment-name",
            "phase8-test",
        ],
        check=True,
    )

    assert (output_root / "optimization_trials.csv").exists()
    assert (output_root / "cv_summary.json").exists()
    assert (output_root / "selected_model.json").exists()
    assert (output_root / "holdout_metrics.json").exists()

    selected = json.loads((output_root / "selected_model.json").read_text("utf-8"))
    assert selected["best_model_family"] == "logreg"
    assert selected["winner_rule"] == "accuracy_floor_relaxed"

    winner_root = output_root / selected["best_model_family"]
    assert (winner_root / "model.pkl").exists()
    assert (winner_root / "metrics.json").exists()
    assert (winner_root / "classification_report.json").exists()
    assert (winner_root / "confusion_matrix.json").exists()
    assert (winner_root / "run_metadata.json").exists()
