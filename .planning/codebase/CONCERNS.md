CONCERNS
========

Summary
-------
This section lists areas of technical debt, potential security issues, and fragile components observed in the repository.

Technical debt & fragility
--------------------------
- Notebooks contain exploratory code that may duplicate logic; risk of drift between notebooks and `src/` modules. Action: extract and centralize stable logic into `src/`.
- Minimal automated tests (`tests/test_smoke.py` only). Low test coverage increases refactor risk. Action: add unit and integration tests.
- Dependency versions are not pinned in `requirements.txt`, which can cause reproducibility drift. Action: pin versions or adopt Poetry/lockfile.

Security & secrets
------------------
- No secrets checked into repository found in top-level files. CI and DVC remote configurations likely store credentials outside repo. Action: verify CI secrets and DVC remote are secured.

Performance hotspots
--------------------
- Model training (transformers + torch) can be resource intensive; notebooks may run large experiments without limits. Action: provide small dataset fixtures for CI and unit tests, and recommend GPU/CPU config docs.

CI / Reproducibility
--------------------
- CI runs basic tests and formatting only. No deployment or reproducibility checks beyond DVC presence. Action: add reproducibility targets, e.g., `make reproduce` that runs `dvc repro` and a small training pipeline.

Other notes
-----------
- DVC remote not configured in repo files — confirm remote storage and access policies.
- Dockerfile exists but no explicit build/test pipeline for images — consider adding a `build` job in CI.

Recommended immediate follow-ups
--------------------------------
1. Add test coverage (`pytest-cov`) and expand unit tests for `src/` modules.
2. Pin dependencies or adopt a lockfile for reproducibility.
3. Extract reusable code from notebooks into `src/` and add tests.
4. Document DVC remote and MLflow setup in `docs/` and ensure secrets are not in repo.
