from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.topic_modeling import run_topic_modeling


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": ["q1", "q2", "q3", "q4", "q5", "q6"],
            "answer": [
                "network security policy and governance",
                "network intrusion detection and response",
                "recipe ingredients and cooking steps",
                "kitchen meal preparation methods",
                "security controls and compliance checks",
                "cooking temperatures and food safety",
            ],
            "label": ["direct", "direct", "evasive", "evasive", "direct", "evasive"],
        }
    )


def test_topic_modeling_outputs_required_files(tmp_path):
    generated = run_topic_modeling(sample_frame(), tmp_path / "phase4", source_data="prepared.parquet")

    out_dir = tmp_path / "phase4" / "topic_modeling"
    assert generated
    assert (out_dir / "topic_top_terms.csv").exists()
    assert (out_dir / "topic_prevalence_by_label.csv").exists()
    assert (out_dir / "topic_summary.json").exists()


def test_topic_modeling_is_deterministic(tmp_path):
    out_root = tmp_path / "phase4"
    run_topic_modeling(sample_frame(), out_root, source_data="prepared.parquet", topics=3, seed=42)
    first = (out_root / "topic_modeling" / "topic_top_terms.csv").read_text(encoding="utf-8")

    run_topic_modeling(sample_frame(), out_root, source_data="prepared.parquet", topics=3, seed=42)
    second = (out_root / "topic_modeling" / "topic_top_terms.csv").read_text(encoding="utf-8")
    assert first == second


def test_topic_modeling_raises_on_missing_columns(tmp_path):
    with pytest.raises(ValueError, match="Missing required columns"):
        run_topic_modeling(pd.DataFrame({"answer": ["x"]}), tmp_path / "phase4", source_data="x")
