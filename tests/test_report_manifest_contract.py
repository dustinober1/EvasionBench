from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.build_report_manifest import ManifestBuildError, generate_manifest
from src.reporting import (
    REQUIRED_MANIFEST_SECTIONS,
    REQUIRED_PROVENANCE_KEYS,
    validate_report_manifest,
)


def _write(path: Path, content: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _args(tmp_path: Path) -> argparse.Namespace:
    dataset_path = tmp_path / "data" / "processed" / "evasionbench_prepared.parquet"
    phase3_root = tmp_path / "artifacts" / "analysis" / "phase3"
    phase4_root = tmp_path / "artifacts" / "analysis" / "phase4"
    phase5_root = tmp_path / "artifacts" / "models" / "phase5"
    phase6_root = tmp_path / "artifacts" / "models" / "phase6" / "transformer"
    xai_root = tmp_path / "artifacts" / "explainability" / "phase6"
    diagnostics_root = tmp_path / "artifacts" / "diagnostics" / "phase6"

    # Dataset
    _write(dataset_path, "placeholder parquet bytes")

    # Analyses
    _write(phase3_root / "artifact_index.json")
    _write(phase3_root / "lexical" / "top_terms.csv", "term,count\nfoo,1\n")
    _write(phase4_root / "artifact_index.json")
    _write(phase4_root / "semantic_similarity" / "summary.json")

    # Models
    _write(phase5_root / "run_summary.json")
    _write(phase5_root / "logreg" / "metrics.json")
    _write(phase5_root / "tree" / "metrics.json")
    _write(phase5_root / "boosting" / "metrics.json")
    _write(phase6_root / "metrics.json")
    _write(phase6_root / "run_metadata.json")
    _write(phase6_root / "model" / "config.json")

    # Explainability
    _write(xai_root / "xai_summary.json")
    _write(xai_root / "transformer" / "transformer_xai_summary.json")
    _write(xai_root / "transformer" / "transformer_xai.html", "<html></html>\n")

    # Diagnostics
    _write(diagnostics_root / "label_diagnostics_summary.json")
    _write(diagnostics_root / "label_diagnostics_report.md", "# report\n")

    return argparse.Namespace(
        dataset_path=str(dataset_path),
        phase3_root=str(phase3_root),
        phase4_root=str(phase4_root),
        phase5_root=str(phase5_root),
        phase6_root=str(phase6_root),
        xai_root=str(xai_root),
        diagnostics_root=str(diagnostics_root),
        output=str(
            tmp_path / "artifacts" / "reports" / "phase7" / "provenance_manifest.json"
        ),
    )


def test_manifest_schema_requires_sections() -> None:
    with pytest.raises(ValueError, match="Manifest sections missing required keys"):
        validate_report_manifest(
            {
                "manifest_version": "v1",
                "generated_at": "2026-02-11T00:00:00Z",
                "generated_by": "scripts/build_report_manifest.py",
                "sections": {"dataset": []},
            }
        )


def test_manifest_builder_writes_required_provenance_keys(tmp_path: Path) -> None:
    args = _args(tmp_path)
    manifest = generate_manifest(args)

    assert tuple(manifest["sections"].keys()) == REQUIRED_MANIFEST_SECTIONS

    for section in REQUIRED_MANIFEST_SECTIONS:
        entries = manifest["sections"][section]
        assert entries, f"section should not be empty: {section}"
        for entry in entries:
            for key in REQUIRED_PROVENANCE_KEYS:
                assert key in entry
                assert entry[key]


def test_manifest_builder_is_deterministic_sorted_order(tmp_path: Path) -> None:
    args = _args(tmp_path)
    manifest = generate_manifest(args)

    model_entries = manifest["sections"]["models"]
    observed = [entry["artifact_path"] for entry in model_entries]
    assert observed == sorted(observed)


def test_manifest_builder_fails_with_actionable_missing_paths(tmp_path: Path) -> None:
    args = _args(tmp_path)
    (Path(args.phase3_root) / "artifact_index.json").unlink()

    with pytest.raises(ManifestBuildError, match="run scripts/run_phase3_analyses.py"):
        generate_manifest(args)
