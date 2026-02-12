from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.reporting import build_report_manifest, make_provenance_entry, write_json

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, content: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _manifest_fixture(project_root: Path) -> Path:
    # Backing artifact files referenced by manifest
    _write(project_root / "data/processed/evasionbench_prepared.parquet", "data\n")
    _write(
        project_root / "artifacts/analysis/phase3/core_stats/class_distribution.png",
        "img\n",
    )
    _write(
        project_root
        / "artifacts/analysis/phase4/semantic_similarity/semantic_similarity_by_label.csv",
        "x,y\n",
    )
    _write(
        project_root / "artifacts/models/phase5/run_summary.json",
        '{"logreg": {"f1_macro": 0.7}, "tree": {"f1_macro": 0.8}}\n',
    )
    _write(
        project_root / "artifacts/models/phase6/transformer/metrics.json",
        '{"accuracy": 0.6, "f1_macro": 0.55}\n',
    )
    _write(
        project_root / "artifacts/explainability/phase6/xai_summary.json",
        '{"logreg": {"n_samples": 5}}\n',
    )
    _write(
        project_root / "artifacts/diagnostics/phase6/label_diagnostics_summary.json",
        '{"label_issues": 1, "quality_score": 98.5}\n',
    )

    manifest = build_report_manifest(
        section_entries={
            "dataset": [
                make_provenance_entry(
                    stage="prepare_data",
                    script="scripts/prepare_data.py",
                    artifact_path="data/processed/evasionbench_prepared.parquet",
                    generated_at="2026-02-11T00:00:00Z",
                )
            ],
            "analyses": [
                make_provenance_entry(
                    stage="analyze_phase3",
                    script="scripts/run_phase3_analyses.py",
                    artifact_path="artifacts/analysis/phase3/core_stats/class_distribution.png",
                    generated_at="2026-02-11T00:00:00Z",
                ),
                make_provenance_entry(
                    stage="analyze_phase4",
                    script="scripts/run_phase4_analyses.py",
                    artifact_path="artifacts/analysis/phase4/semantic_similarity/semantic_similarity_by_label.csv",
                    generated_at="2026-02-11T00:00:00Z",
                ),
            ],
            "models": [
                make_provenance_entry(
                    stage="phase5_classical_baselines",
                    script="scripts/run_classical_baselines.py",
                    artifact_path="artifacts/models/phase5/run_summary.json",
                    generated_at="2026-02-11T00:00:00Z",
                ),
                make_provenance_entry(
                    stage="phase6_transformer_baselines",
                    script="scripts/run_transformer_baselines.py",
                    artifact_path="artifacts/models/phase6/transformer/metrics.json",
                    generated_at="2026-02-11T00:00:00Z",
                ),
            ],
            "explainability": [
                make_provenance_entry(
                    stage="phase6_xai_classical",
                    script="scripts/run_explainability_analysis.py",
                    artifact_path="artifacts/explainability/phase6/xai_summary.json",
                    generated_at="2026-02-11T00:00:00Z",
                )
            ],
            "diagnostics": [
                make_provenance_entry(
                    stage="phase6_label_diagnostics",
                    script="scripts/run_label_diagnostics.py",
                    artifact_path="artifacts/diagnostics/phase6/label_diagnostics_summary.json",
                    generated_at="2026-02-11T00:00:00Z",
                )
            ],
        },
        generated_at="2026-02-11T00:00:00Z",
    )

    manifest_path = project_root / "artifacts/reports/phase7/provenance_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def test_report_generation_outputs_required_sections(tmp_path: Path) -> None:
    manifest_path = _manifest_fixture(tmp_path)
    output_path = tmp_path / "artifacts/reports/phase7/report.md"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/generate_research_report.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--template",
            str(ROOT / "templates/research_report.md.j2"),
            "--project-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    content = output_path.read_text(encoding="utf-8")

    expected_sections = [
        "## Methodology",
        "## Core Analyses",
        "## Modeling",
        "## Explainability",
        "## Label Quality",
        "## Reproducibility",
    ]
    positions = [content.index(section) for section in expected_sections]
    assert positions == sorted(positions)

    assert "Generated at:" in content
    assert "Git SHA:" in content
    assert "Pipeline run id:" in content

    assert "artifacts/analysis/phase3/core_stats/class_distribution.png" in content
    assert (
        "artifacts/analysis/phase4/semantic_similarity/semantic_similarity_by_label.csv"
        in content
    )


def test_report_generation_fails_on_missing_required_manifest_segments(
    tmp_path: Path,
) -> None:
    manifest_path = _manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sections"]["diagnostics"] = []
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    output_path = tmp_path / "artifacts/reports/phase7/report.md"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/generate_research_report.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--template",
            str(ROOT / "templates/research_report.md.j2"),
            "--project-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "Failed to build report context" in proc.stderr
