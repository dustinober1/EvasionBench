from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.reporting import build_report_manifest, build_traceability_map, make_provenance_entry, write_json

ROOT = Path(__file__).resolve().parents[1]


def _build_fixture(project_root: Path) -> tuple[Path, Path, list[str]]:
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
                )
            ],
            "models": [
                make_provenance_entry(
                    stage="phase5_classical_baselines",
                    script="scripts/run_classical_baselines.py",
                    artifact_path="artifacts/models/phase5/run_summary.json",
                    generated_at="2026-02-11T00:00:00Z",
                )
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

    trace_ids = sorted(build_traceability_map(manifest).keys())
    lines = [
        "# Render Contract Fixture",
        "Generated at: `2026-02-11T00:00:00Z`",
        "Git SHA: `deadbeef`",
        "Pipeline run id: `phase7-test`",
        "",
        "## Reproducibility",
    ]
    lines.extend([f"- `{trace_id}`" for trace_id in trace_ids])

    markdown_path = project_root / "artifacts/reports/phase7/report.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return markdown_path, manifest_path, trace_ids


def test_render_outputs_and_traceability_contract(tmp_path: Path) -> None:
    markdown_path, manifest_path, expected_trace_ids = _build_fixture(tmp_path)
    output_root = tmp_path / "artifacts/reports/phase7"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/render_research_report.py",
            "--input",
            str(markdown_path),
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--project-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

    html_path = output_root / "report.html"
    pdf_path = output_root / "report.pdf"
    traceability_path = output_root / "report_traceability.json"

    assert html_path.exists() and html_path.stat().st_size > 0
    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    assert traceability_path.exists() and traceability_path.stat().st_size > 0

    markdown_text = markdown_path.read_text(encoding="utf-8")
    html_text = html_path.read_text(encoding="utf-8")
    pdf_bytes = pdf_path.read_bytes()

    assert "Pipeline run id:" in markdown_text
    assert "Pipeline run id:" in html_text
    assert b"Pipeline run id:" in pdf_bytes

    payload = json.loads(traceability_path.read_text(encoding="utf-8"))
    rendered_ids = sorted(payload["items"].keys())
    assert rendered_ids == expected_trace_ids
    assert all(item["referenced_in_report"] for item in payload["items"].values())
