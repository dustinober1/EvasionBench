from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis.question_behavior import TAXONOMY, run_question_behavior


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": [
                "Is this policy valid?",
                "How do I configure this?",
                "Compare option A vs option B",
                "What is encryption?",
            ],
            "answer": [
                "I cannot confirm policy validity.",
                "Open settings and apply the configuration steps.",
                "Option A is faster while option B is cheaper.",
                "Encryption protects data at rest and in transit.",
            ],
            "label": ["evasive", "direct", "direct", "direct"],
        }
    )


def test_question_behavior_outputs_required_files(tmp_path):
    generated = run_question_behavior(
        sample_frame(),
        tmp_path / "phase4",
        source_data="prepared.parquet",
        emit_assignments=True,
    )

    out_dir = tmp_path / "phase4" / "question_behavior"
    assert generated
    assert (out_dir / "question_assignments.parquet").exists()
    assert (out_dir / "question_behavior_metrics.csv").exists()
    assert (out_dir / "question_behavior_summary.json").exists()

    taxonomy = json.loads(
        (out_dir / "taxonomy_metadata.json").read_text(encoding="utf-8")
    )
    assert taxonomy["taxonomy"] == list(TAXONOMY)


def test_question_behavior_is_deterministic(tmp_path):
    out_root = tmp_path / "phase4"
    run_question_behavior(
        sample_frame(), out_root, source_data="prepared.parquet", emit_assignments=True
    )
    first = (
        out_root / "question_behavior" / "question_behavior_metrics.csv"
    ).read_text(encoding="utf-8")

    run_question_behavior(
        sample_frame(), out_root, source_data="prepared.parquet", emit_assignments=True
    )
    second = (
        out_root / "question_behavior" / "question_behavior_metrics.csv"
    ).read_text(encoding="utf-8")
    assert first == second


def test_question_behavior_raises_on_missing_columns(tmp_path):
    with pytest.raises(ValueError, match="Missing required columns"):
        run_question_behavior(
            pd.DataFrame({"question": ["q"]}), tmp_path / "phase4", source_data="x"
        )
