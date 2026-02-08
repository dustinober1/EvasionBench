from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.linguistic_quality import run_linguistic_quality


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "answer": [
                "Maybe I cannot say for sure.",
                "I can provide a direct answer now.",
                "Perhaps I am unable to comment.",
                "This is a clear response.",
            ],
            "label": ["evasive", "direct", "evasive", "direct"],
        }
    )


def test_linguistic_outputs_and_bounds(tmp_path):
    out = tmp_path / "phase3"
    run_linguistic_quality(sample_frame(), out, source_data="data/processed/evasionbench_prepared.parquet")

    read_df = pd.read_csv(out / "linguistic_quality" / "readability_summary.csv")
    assert {"label", "flesch_reading_ease", "flesch_kincaid_grade", "smog_index"}.issubset(read_df.columns)

    pos_df = pd.read_csv(out / "linguistic_quality" / "pos_proportions.csv")
    assert {"label", "pos", "count", "proportion"}.issubset(pos_df.columns)
    assert ((pos_df["proportion"] >= 0.0) & (pos_df["proportion"] <= 1.0)).all()

    disc_df = pd.read_csv(out / "linguistic_quality" / "discourse_markers.csv")
    assert {"label", "hedging_rate", "evasive_marker_rate"}.issubset(disc_df.columns)


def test_linguistic_is_deterministic(tmp_path):
    out = tmp_path / "phase3"
    run_linguistic_quality(sample_frame(), out, source_data="x")
    first = (out / "linguistic_quality" / "discourse_markers.csv").read_text(encoding="utf-8")

    run_linguistic_quality(sample_frame(), out, source_data="x")
    second = (out / "linguistic_quality" / "discourse_markers.csv").read_text(encoding="utf-8")
    assert first == second


def test_linguistic_rejects_invalid_schema(tmp_path):
    with pytest.raises(ValueError, match="Missing required columns"):
        run_linguistic_quality(pd.DataFrame({"label": ["x"]}), tmp_path / "phase3", source_data="x")
