# CI Baseline

This repository uses one canonical check entrypoint for both local development and CI:

- `bash scripts/ci_check.sh`

## Triggers

GitHub Actions runs on:

- `push`
- `pull_request`

Workflow file: `.github/workflows/ci.yml`

## Check sequence

`ci_check.sh` runs checks in this order:

1. `python scripts/verify_repo_structure.py`
2. `black --check .`
3. `pytest -q`

The first failing step exits non-zero and fails CI.

## Local parity

Run the same baseline checks locally:

- `make ci-check`
- or `bash scripts/ci_check.sh`

## Failure remediation

- Structure check failed:
  - Ensure required directories/files exist.
  - Remove forbidden committed artifacts (`__pycache__`, `.ipynb_checkpoints`, `.DS_Store`).
  - Re-run: `python scripts/verify_repo_structure.py`
- Black formatting failed:
  - Fix with `black .`
  - Re-run: `black --check .`
- Tests failed:
  - Re-run target test directly with `pytest -q tests/<module>.py`
  - Re-run full suite with `pytest -q`

## Contributor expectation

Before opening a PR, run `make ci-check` and ensure it passes locally.
