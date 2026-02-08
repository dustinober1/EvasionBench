from __future__ import annotations

import json

from src.analysis.artifacts import PHASE3_FAMILIES, ensure_phase3_layout, write_artifact_index


def test_phase3_layout_is_deterministic(tmp_path):
    first = ensure_phase3_layout(tmp_path / "phase3")
    second = ensure_phase3_layout(tmp_path / "phase3")

    assert set(first.keys()) == {"root", *PHASE3_FAMILIES}
    assert first == second
    for key, path in first.items():
        assert path.exists(), key


def test_artifact_index_contains_required_keys(tmp_path):
    output_root = tmp_path / "phase3"
    marker = output_root / "core_stats" / "class_distribution.csv"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("label,count\n", encoding="utf-8")

    index_path = write_artifact_index(
        output_root,
        stage="core_stats",
        generated_files=[marker],
        source_data="data/processed/evasionbench_prepared.parquet",
        metadata={"sections": ["quality", "lengths"]},
    )

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "phase3"
    assert len(payload["entries"]) == 1

    entry = payload["entries"][0]
    assert entry["stage"] == "core_stats"
    assert entry["source_data"] == "data/processed/evasionbench_prepared.parquet"
    assert entry["generated_files"] == ["core_stats/class_distribution.csv"]
    assert entry["metadata"]["sections"] == ["quality", "lengths"]
