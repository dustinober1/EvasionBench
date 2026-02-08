from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis.core_stats import run_core_stats


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": ["q1", "q2", "q3", "q4"],
            "answer": ["short", "much longer answer", "short", "medium"],
            "label": ["evasive", "direct", "evasive", "direct"],
        }
    )


def test_core_stats_outputs_required_files(tmp_path):
    generated = run_core_stats(
        sample_frame(),
        tmp_path / "phase3",
        source_data="data/processed/evasionbench_prepared.parquet",
    )

    assert generated
    core_dir = tmp_path / "phase3" / "core_stats"
    assert (core_dir / "class_distribution.csv").exists()
    assert (core_dir / "length_tests.json").exists()

    payload = json.loads((core_dir / "length_tests.json").read_text(encoding="utf-8"))
    assert "question_length" in payload
    assert "answer_length" in payload
    assert "kruskal" in payload["question_length"]
    assert "pairwise_mannwhitney" in payload["question_length"]


def test_core_stats_is_deterministic(tmp_path):
    frame = sample_frame()
    out = tmp_path / "phase3"
    run_core_stats(frame, out, source_data="x")
    first = (out / "core_stats" / "class_distribution.csv").read_text(encoding="utf-8")

    run_core_stats(frame, out, source_data="x")
    second = (out / "core_stats" / "class_distribution.csv").read_text(encoding="utf-8")
    assert first == second


def test_core_stats_raises_on_missing_columns(tmp_path):
    with pytest.raises(ValueError, match="Missing required columns"):
        run_core_stats(pd.DataFrame({"question": ["only"]}), tmp_path / "phase3", source_data="x")
