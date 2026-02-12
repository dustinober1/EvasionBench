"""Validate repository structure for script-first development boundaries."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REQUIRED_DIRECTORIES = ["scripts", "src", "docs", "tests", ".github/workflows"]
REQUIRED_FILES = ["Makefile", ".github/workflows/ci.yml", "scripts/ci_check.sh"]
FORBIDDEN_PATHS = ["__pycache__", ".ipynb_checkpoints", ".DS_Store"]


def check_repository_structure(root: Path) -> list[str]:
    errors: list[str] = []

    for rel_path in REQUIRED_DIRECTORIES:
        path = root / rel_path
        if not path.is_dir():
            errors.append(f"Missing required directory: {rel_path}")

    for rel_path in REQUIRED_FILES:
        path = root / rel_path
        if not path.is_file():
            errors.append(f"Missing required file: {rel_path}")

    tracked_files = _tracked_paths(root)
    candidate_paths = (
        tracked_files if tracked_files is not None else _filesystem_paths(root)
    )
    for tracked_path in candidate_paths:
        for forbidden_name in FORBIDDEN_PATHS:
            if forbidden_name in tracked_path.parts:
                errors.append(f"Forbidden committed artifact: {tracked_path}")
            elif tracked_path.name == forbidden_name:
                errors.append(f"Forbidden committed artifact: {tracked_path}")

    return errors


def _tracked_paths(root: Path) -> list[Path] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    entries = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return [Path(entry) for entry in entries]


def _filesystem_paths(root: Path) -> list[Path]:
    ignored_roots = {".git", ".venv", ".pytest_cache"}
    paths: list[Path] = []
    for path in root.rglob("*"):
        rel_path = path.relative_to(root)
        if rel_path.parts and rel_path.parts[0] in ignored_roots:
            continue
        paths.append(rel_path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root path to validate",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    errors = check_repository_structure(root)

    if errors:
        print("Repository structure check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Repository structure check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
