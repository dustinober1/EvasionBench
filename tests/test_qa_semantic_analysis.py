from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis.qa_semantic import run_qa_semantic


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": [
                "How do I boil pasta?",
                "What is the weather?",
                "Is this safe?",
                "Compare apples and oranges",
            ],
            "answer": [
                "Boil water then add pasta and cook for eight minutes.",
                "The weather is sunny and warm.",
                "I cannot provide safety guarantees.",
                "Apples are sweeter while oranges are citrus.",
            ],
            "label": ["direct", "direct", "evasive", "direct"],
        }
    )


def test_semantic_analysis_outputs_required_files(tmp_path):
    generated = run_qa_semantic(sample_frame(), tmp_path / "phase4", source_data="prepared.parquet")

    out_dir = tmp_path / "phase4" / "semantic_similarity"
    assert generated
    assert (out_dir / "qa_similarity_rows.parquet").exists()
    assert (out_dir / "semantic_similarity_by_label.csv").exists()
    assert (out_dir / "hypothesis_summary.json").exists()

    payload = json.loads((out_dir / "hypothesis_summary.json").read_text(encoding="utf-8"))
    assert payload["hypotheses"]


def test_semantic_analysis_is_deterministic(tmp_path):
    out_root = tmp_path / "phase4"
    run_qa_semantic(sample_frame(), out_root, source_data="prepared.parquet")
    first = (out_root / "semantic_similarity" / "semantic_similarity_by_label.csv").read_text(encoding="utf-8")

    run_qa_semantic(sample_frame(), out_root, source_data="prepared.parquet")
    second = (out_root / "semantic_similarity" / "semantic_similarity_by_label.csv").read_text(encoding="utf-8")
    assert first == second


def test_semantic_analysis_raises_on_missing_columns(tmp_path):
    with pytest.raises(ValueError, match="Missing required columns"):
        run_qa_semantic(pd.DataFrame({"question": ["q"]}), tmp_path / "phase4", source_data="x")
