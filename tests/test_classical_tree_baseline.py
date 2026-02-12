from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.evaluation import compute_classification_metrics, validate_evaluation_contract
from src.models import train_tree_or_boosting


def _prepared_data(path: Path) -> None:
    rows = []
    for idx in range(60):
        label = "evasive" if idx % 3 else "non_evasive"
        rows.append(
            {
                "question": f"Question {idx}",
                "answer": f"Answer text {idx}",
                "label": label,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_tree_baseline_writes_contract_artifacts(tmp_path: Path) -> None:
    data_path = tmp_path / "prepared.parquet"
    output_root = tmp_path / "tree"
    _prepared_data(data_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/train_tree_baseline.py",
            "--input",
            str(data_path),
            "--output-root",
            str(output_root),
            "--model-family",
            "tree",
            "--random-state",
            "42",
        ],
        check=True,
    )

    validate_evaluation_contract(output_root)
    metrics = json.loads((output_root / "metrics.json").read_text(encoding="utf-8"))
    assert "f1_macro" in metrics

    metadata = json.loads(
        (output_root / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["model_family"] == "tree"
    assert metadata["split_seed"] == 42
    assert metadata["split_metadata"]["method"] == "train_test_split"
    assert metadata["split_metadata"]["stratify"] in {True, False}
    assert metadata["feature_config"]["ngram_range"] == [1, 2]
    assert "model_config" in metadata


def test_tree_baseline_missing_target_column(tmp_path: Path) -> None:
    data_path = tmp_path / "bad.parquet"
    pd.DataFrame([{"question": "Q", "answer": "A"}]).to_parquet(data_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_tree_baseline.py",
            "--input",
            str(data_path),
            "--output-root",
            str(tmp_path / "out"),
            "--target-col",
            "label",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Missing target column" in (result.stderr + result.stdout)


def test_tree_baseline_missing_feature_columns(tmp_path: Path) -> None:
    data_path = tmp_path / "bad_features.parquet"
    pd.DataFrame([{"question": "Q", "label": "evasive"}]).to_parquet(
        data_path, index=False
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_tree_baseline.py",
            "--input",
            str(data_path),
            "--output-root",
            str(tmp_path / "out"),
            "--target-col",
            "label",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Missing feature columns" in (result.stderr + result.stdout)


def test_tree_baseline_deterministic_metrics() -> None:
    rows = []
    for idx in range(80):
        label = "evasive" if idx % 2 == 0 else "non_evasive"
        rows.append(
            {
                "question": f"Question {idx}",
                "answer": f"Answer text {idx}",
                "label": label,
            }
        )
    frame = pd.DataFrame(rows)

    training_first = train_tree_or_boosting(
        frame,
        model_family="tree",
        target_col="label",
        random_state=123,
        test_size=0.25,
        n_estimators=30,
        max_depth=4,
    )
    metrics_first = compute_classification_metrics(
        training_first["y_test"], training_first["predictions"]
    )

    training_second = train_tree_or_boosting(
        frame,
        model_family="tree",
        target_col="label",
        random_state=123,
        test_size=0.25,
        n_estimators=30,
        max_depth=4,
    )
    metrics_second = compute_classification_metrics(
        training_second["y_test"], training_second["predictions"]
    )

    assert metrics_first == metrics_second
