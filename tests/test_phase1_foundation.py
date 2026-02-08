from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_repo_structure import check_repository_structure


def _build_minimal_repo(root: Path) -> None:
    required_dirs = ["scripts", "src", "docs", "tests", ".github/workflows"]
    required_files = ["Makefile", ".github/workflows/ci.yml", "scripts/ci_check.sh"]

    for rel_path in required_dirs:
        (root / rel_path).mkdir(parents=True, exist_ok=True)

    for rel_path in required_files:
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok\n", encoding="utf-8")


def test_verify_repo_structure_passes_for_expected_layout(tmp_path: Path) -> None:
    _build_minimal_repo(tmp_path)

    errors = check_repository_structure(tmp_path)

    assert errors == []


def test_verify_repo_structure_fails_when_required_file_missing(tmp_path: Path) -> None:
    _build_minimal_repo(tmp_path)
    (tmp_path / "scripts/ci_check.sh").unlink()

    errors = check_repository_structure(tmp_path)

    assert "Missing required file: scripts/ci_check.sh" in errors


def test_verify_repo_structure_fails_when_forbidden_artifact_exists(
    tmp_path: Path,
) -> None:
    _build_minimal_repo(tmp_path)
    (tmp_path / "src/__pycache__").mkdir(parents=True)

    errors = check_repository_structure(tmp_path)

    assert any(
        "Forbidden committed artifact: src/__pycache__" in item for item in errors
    )
