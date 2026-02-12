from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation import validate_evaluation_contract


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_validate_evaluation_contract_accepts_required_files(tmp_path: Path) -> None:
    _write(
        tmp_path / "metrics.json",
        {"accuracy": 0.8, "f1_macro": 0.8, "precision_macro": 0.8, "recall_macro": 0.8},
    )
    _write(
        tmp_path / "classification_report.json",
        {
            "evasive": {"f1-score": 0.7},
            "non_evasive": {"f1-score": 0.9},
            "accuracy": 0.8,
            "macro avg": {"f1-score": 0.8},
            "weighted avg": {"f1-score": 0.8},
        },
    )
    _write(
        tmp_path / "confusion_matrix.json",
        {"labels": ["evasive", "non_evasive"], "confusion_matrix": [[7, 3], [1, 9]]},
    )
    _write(
        tmp_path / "run_metadata.json",
        {
            "model_family": "logreg",
            "split_seed": 42,
            "feature_config": {"ngram_range": [1, 2]},
        },
    )

    validate_evaluation_contract(tmp_path)


def test_validate_evaluation_contract_fails_on_missing_key(tmp_path: Path) -> None:
    _write(tmp_path / "metrics.json", {"accuracy": 0.9})
    _write(
        tmp_path / "classification_report.json",
        {"accuracy": 0.9, "macro avg": {}, "weighted avg": {}},
    )
    _write(
        tmp_path / "confusion_matrix.json", {"labels": ["a"], "confusion_matrix": [[1]]}
    )
    _write(
        tmp_path / "run_metadata.json",
        {"model_family": "tree", "split_seed": 42, "feature_config": {}},
    )

    with pytest.raises(ValueError, match="metrics.json missing required key"):
        validate_evaluation_contract(tmp_path)
