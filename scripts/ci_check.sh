#!/usr/bin/env bash
set -euo pipefail

echo "[ci-check] Verifying repository structure"
python scripts/verify_repo_structure.py

echo "[ci-check] Running formatter check"
black --check .

echo "[ci-check] Running linter"
ruff check .

echo "[ci-check] Running tests with coverage"
pytest -q --cov=src --cov-report=term-missing

echo "[ci-check] All checks passed"
