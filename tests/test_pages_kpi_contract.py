from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from test_pages_dataset_contract import ROOT, _fixture_project

REQUIRED_KPI_FIELDS = [
    "model_family",
    "accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
]


def _run_build(*, manifest_path: Path, publish_root: Path, project_root: Path):
    return subprocess.run(
        [
            sys.executable,
            "scripts/build_pages_dataset.py",
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(publish_root),
            "--project-root",
            str(project_root),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_pages_build_writes_canonical_kpi_and_quality_report(tmp_path: Path) -> None:
    manifest_path = _fixture_project(tmp_path)
    publish_root = tmp_path / "artifacts/publish"

    proc = _run_build(
        manifest_path=manifest_path,
        publish_root=publish_root,
        project_root=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr

    kpi = json.loads(
        (publish_root / "data/kpi_summary.json").read_text(encoding="utf-8")
    )
    quality = json.loads(
        (publish_root / "data/site_quality_report.json").read_text(encoding="utf-8")
    )

    for key in REQUIRED_KPI_FIELDS:
        assert key in kpi
        assert str(kpi[key]).strip().lower() not in {"unknown", "n/a", "none", "null"}

    assert quality["missing_fields"] == []
    assert quality["placeholder_cards"] == []
    assert quality["fallback_usage"]["kpi_source"] == "phase8_selected_model"


def test_pages_build_fails_when_canonical_kpi_contract_is_unresolvable(
    tmp_path: Path,
) -> None:
    manifest_path = _fixture_project(tmp_path)
    publish_root = tmp_path / "artifacts/publish"

    (tmp_path / "artifacts/models/phase8/selected_model.json").unlink()
    (tmp_path / "artifacts/models/phase5/model_comparison/summary.json").unlink()
    (tmp_path / "artifacts/models/phase5/run_summary.json").write_text(
        "{}\n", encoding="utf-8"
    )

    proc = _run_build(
        manifest_path=manifest_path,
        publish_root=publish_root,
        project_root=tmp_path,
    )
    assert proc.returncode != 0
    assert "Canonical KPI contract violation" in proc.stderr
