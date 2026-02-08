STRUCTURE
=========

Repository layout
-----------------
Top-level directories and their purposes:

- `notebooks/` — exploratory analysis and experiment notebooks (many numbered notebooks for workflow)
- `src/` — reusable Python modules: `models.py`, `data.py`, `features.py`, `evaluation.py`, `utils.py`, `explainability.py`, `visualization.py`.
- `api/` — serving layer: `main.py` (FastAPI application)
- `dashboard/` — demo UI: `app.py` (Streamlit)
- `scripts/` — utility scripts (e.g., `download_data.py`)
- `data/` — DVC-tracked datasets and artifacts
- `docs/` and `papers/` — documentation and related writeups
- `tests/` — test suite (currently minimal: `tests/test_smoke.py`)
- `docker/` — Dockerfile for containerization

Key files
---------
- `README.md` — project overview and quick start
- `requirements.txt` — Python dependencies
- `.github/workflows/ci.yml` — test and lint CI
- `dvc.yaml` — data pipeline definition
- `pyproject.toml` — Black/formatter config

Module conventions & naming
--------------------------
- Python modules live under `src/` and follow simple module-per-concern naming (e.g., `src/models.py`).
- Notebooks are numbered and follow a canonical workflow (`01_...`, `02_...`).

Where to look for common tasks
------------------------------
- Run tests: `pytest -q` (CI runs same step)
- Data pipeline: `dvc repro` and `scripts/download_data.py`
- Start API: `uvicorn api.main:app --reload`
- Launch dashboard: `streamlit run dashboard/app.py`

Size & complexity
-----------------
- Codebase is moderately sized with most logic in a few `src/*.py` modules and many notebooks. Test coverage and CI checks are basic; consider increasing automated tests before major refactors.
