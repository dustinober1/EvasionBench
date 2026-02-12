from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation import compute_classification_metrics, validate_evaluation_contract
from src.models import train_tfidf_logreg


def _prepared_data(path: Path) -> None:
    rows = []
    for idx in range(40):
        label = "evasive" if idx % 2 == 0 else "non_evasive"
        rows.append(
            {
                "question": f"Question {idx}",
                "answer": f"Answer text {idx} with context",
                "label": label,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_logreg_baseline_writes_contract_artifacts(tmp_path: Path) -> None:
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
            "logreg",
            "--tracking-uri",
            f"file:{tmp_path / 'mlruns'}",
            "--experiment-name",
            "phase5-test",
        ],
        check=True,
    )

    family_root = output_root / "logreg"
    validate_evaluation_contract(family_root)

    metrics = json.loads((family_root / "metrics.json").read_text(encoding="utf-8"))
    assert 0.0 <= metrics["accuracy"] <= 1.0

    report = json.loads(
        (family_root / "classification_report.json").read_text(encoding="utf-8")
    )
    assert "macro avg" in report
    assert "weighted avg" in report

    metadata = json.loads(
        (family_root / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["model_family"] == "logreg"
    assert metadata["split_seed"] == 42
    assert "feature_config" in metadata
    assert "model_config" in metadata
    assert metadata["split_metadata"]["method"] == "train_test_split"


def test_run_experiment_errors_when_target_missing(tmp_path: Path) -> None:
    data_path = tmp_path / "prepared.parquet"
    pd.DataFrame(
        [
            {"question": "Q1", "answer": "A1", "wrong": "evasive"},
            {"question": "Q2", "answer": "A2", "wrong": "non_evasive"},
        ]
    ).to_parquet(data_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "--data",
            str(data_path),
            "--target-col",
            "label",
            "--tracking-uri",
            f"file:{tmp_path / 'mlruns'}",
            "--experiment-name",
            "phase5-test",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Missing target column" in (result.stderr + result.stdout)


def test_logreg_baseline_deterministic_metrics() -> None:
    rows = []
    for idx in range(60):
        label = "evasive" if idx % 2 == 0 else "non_evasive"
        rows.append(
            {
                "question": f"Question {idx}",
                "answer": f"Answer text {idx} with context",
                "label": label,
            }
        )
    frame = pd.DataFrame(rows)

    training_first = train_tfidf_logreg(
        frame,
        target_col="label",
        random_state=123,
        test_size=0.25,
        ngram_min=1,
        ngram_max=2,
        min_df=1,
        max_features=500,
        c=0.75,
        class_weight="balanced",
        solver="liblinear",
    )
    preds_first = training_first["model"].predict(training_first["X_test"])
    metrics_first = compute_classification_metrics(
        training_first["y_test"], preds_first
    )

    training_second = train_tfidf_logreg(
        frame,
        target_col="label",
        random_state=123,
        test_size=0.25,
        ngram_min=1,
        ngram_max=2,
        min_df=1,
        max_features=500,
        c=0.75,
        class_weight="balanced",
        solver="liblinear",
    )
    preds_second = training_second["model"].predict(training_second["X_test"])
    metrics_second = compute_classification_metrics(
        training_second["y_test"], preds_second
    )

    assert metrics_first == metrics_second
