CONVENTIONS
===========

Coding style
------------
- Python style: `black` is configured via `pyproject.toml` (line-length 88) and enforced in CI (`black --check .`).
- Pre-commit configured (`.pre-commit-config.yaml`) — follow its hooks for consistent diffs.
- Recommend adding `ruff` for lints and `isort` for import ordering if desired.

Project patterns
----------------
- Keep reusable code in `src/` modules and use notebooks for exploratory analysis. Move any production-ready logic from notebooks into `src/`.
- Use `scripts/` for standalone utilities (data download, one-off transformations).

Naming
------
- Files in `notebooks/` follow numeric prefixes (e.g., `01_data_quality_and_statistics.ipynb`).
- Python modules use lowercase snake_case filenames (`src/features.py`).

Error handling
--------------
- Modules should raise Python exceptions for unrecoverable errors; notebooks may print stack traces during exploration.
- `api/main.py` should validate inputs with Pydantic models (pydantic is in `requirements.txt`) — verify request models are present.

Documentation
-------------
- Project-level docs live in `docs/` and `README.md`. `docs/mlflow_guide.md` and `docs/dvc_guide.md` provide specific infra guidance.

Formatting & commits
--------------------
- CI runs `black --check .` and repository contains `.pre-commit-config.yaml`. Developers should run `black .` locally before committing.

Suggested improvements
----------------------
- Add `pyproject.toml` sections for `ruff` and `isort` and include them in `.pre-commit-config.yaml`.
- Move shared logic out of notebooks and into `src/` for testability.
