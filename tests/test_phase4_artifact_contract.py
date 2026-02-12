from __future__ import annotations

import json

import pytest

from src.analysis.artifacts import (
    PHASE4_FAMILIES,
    ensure_phase4_layout,
    write_phase4_artifact_index,
)


def test_phase4_layout_is_deterministic(tmp_path):
    first = ensure_phase4_layout(tmp_path / "phase4")
    second = ensure_phase4_layout(tmp_path / "phase4")

    assert set(first.keys()) == {"root", *PHASE4_FAMILIES}
    assert first == second
    for key, path in first.items():
        assert path.exists(), key


def test_phase4_artifact_index_contains_required_keys(tmp_path):
    output_root = tmp_path / "phase4"
    marker = output_root / "semantic_similarity" / "semantic_similarity_by_label.csv"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("label,mean_similarity\n", encoding="utf-8")

    index_path = write_phase4_artifact_index(
        output_root,
        stage="semantic_similarity",
        generated_files=[marker],
        source_data="data/processed/evasionbench_prepared.parquet",
        metadata={
            "hypothesis_summary": "semantic_similarity/hypothesis_summary.md",
            "analysis_version": "v1",
        },
    )

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "phase4"
    assert payload["families"] == list(PHASE4_FAMILIES)
    assert len(payload["entries"]) == 1

    entry = payload["entries"][0]
    assert entry["stage"] == "semantic_similarity"
    assert entry["source_data"] == "data/processed/evasionbench_prepared.parquet"
    assert entry["generated_files"] == [
        "semantic_similarity/semantic_similarity_by_label.csv"
    ]
    assert (
        entry["metadata"]["hypothesis_summary"]
        == "semantic_similarity/hypothesis_summary.md"
    )


def test_phase4_artifact_index_validates_hypothesis_pointer(tmp_path):
    with pytest.raises(ValueError, match="missing required keys"):
        write_phase4_artifact_index(
            tmp_path / "phase4",
            stage="topic_modeling",
            generated_files=[],
            source_data="x",
            metadata={"analysis_version": "v1"},
        )
