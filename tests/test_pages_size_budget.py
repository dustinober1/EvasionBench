from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from test_pages_dataset_contract import ROOT, _fixture_project


def test_pages_dataset_enforces_single_file_budget(tmp_path: Path) -> None:
    manifest_path = _fixture_project(tmp_path)
    publish_root = tmp_path / "artifacts/publish"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/build_pages_dataset.py",
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(publish_root),
            "--project-root",
            str(tmp_path),
            "--max-single-file-mb",
            "0.000001",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "exceeds per-file budget" in proc.stderr
