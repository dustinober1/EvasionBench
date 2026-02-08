from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis.lexical import run_lexical


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "answer": [
                "I do not know",
                "I am unable to help",
                "That is a direct answer",
                "Direct response now",
            ],
            "label": ["evasive", "evasive", "direct", "direct"],
        }
    )


def test_lexical_outputs_schema(tmp_path):
    out = tmp_path / "phase3"
    run_lexical(sample_frame(), out, source_data="data/processed/evasionbench_prepared.parquet")

    summary = pd.read_csv(out / "lexical" / "lexical_summary.csv")
    expected = {
        "label",
        "num_texts",
        "token_count",
        "unique_token_count",
        "type_token_ratio",
        "avg_word_length",
    }
    assert expected.issubset(summary.columns)

    payload = json.loads((out / "lexical" / "top_ngrams.json").read_text(encoding="utf-8"))
    assert "direct" in payload
    assert "unigrams" in payload["direct"]


def test_lexical_is_deterministic(tmp_path):
    out = tmp_path / "phase3"
    run_lexical(sample_frame(), out, source_data="x")
    first = (out / "lexical" / "top_ngrams.json").read_text(encoding="utf-8")

    run_lexical(sample_frame(), out, source_data="x")
    second = (out / "lexical" / "top_ngrams.json").read_text(encoding="utf-8")
    assert first == second


def test_lexical_rejects_invalid_schema(tmp_path):
    with pytest.raises(ValueError, match="Missing required columns"):
        run_lexical(pd.DataFrame({"label": ["x"]}), tmp_path / "phase3", source_data="x")
