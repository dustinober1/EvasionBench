STACK
=====

Overview
--------
This project is a Python-based research/data-science portfolio focused on NLP for evasion detection. Primary runtimes and tooling:

- Language: Python 3.x (CI uses 3.11) — see `.github/workflows/ci.yml` and `pyproject.toml`.
- Package management: pip / requirements.txt. A `pyproject.toml` exists for formatter config.
- Data/versioning: DVC (`dvc.yaml`) for dataset pipelines and `data/` under repository.
- ML infra: `transformers`, `torch`, `sentence-transformers`, `huggingface_hub` (see `requirements.txt`).
- Experiment tracking: `mlflow` is listed in `requirements.txt` and docs (`docs/mlflow_guide.md`).

Containers & Deployment
-----------------------
- Docker: `docker/Dockerfile` provides a reproducible container for parts of the project (review for build commands).
- API: `api/main.py` implements a FastAPI app intended to run under `uvicorn` (see `requirements.txt`).
- Dashboard: `dashboard/app.py` and `streamlit` listed in `requirements.txt` for quick demos.

CI & Quality
------------
- CI runs on GitHub Actions: `.github/workflows/ci.yml` (Python 3.11, installs `requirements.txt`, runs `pytest` and `black --check`).
- Pre-commit hooks: `.pre-commit-config.yaml` exists; `black` configured via `pyproject.toml`.

Key dependencies (explicit examples)
-----------------------------------
- `transformers`, `torch`, `sentence-transformers` — model training and embedding.
- `scikit-learn`, `xgboost`, `pandas`, `numpy` — feature engineering and classical baselines.
- `mlflow` — experiment tracking (`docs/mlflow_guide.md`).
- `dvc` — data pipeline control (`dvc.yaml`).
- `fastapi`, `uvicorn`, `pydantic` — serving API (`api/main.py`).

Build / Run / Dev commands
--------------------------
- Install deps: `pip install -r requirements.txt` (CI uses this flow).
- Run tests: `pytest -q` (see `tests/test_smoke.py`).
- Start API locally: `uvicorn api.main:app --reload` (port and env variables may be required).
- Run dashboard: `streamlit run dashboard/app.py`.
- Reproduce data pipeline: `dvc repro` (inspect `dvc.yaml`).

Configuration files & locations
-------------------------------
- `requirements.txt` — primary dependency list.
- `pyproject.toml` — Black configuration and potential future tool config.
- `environment.yml` — conda environment (alternative to `requirements.txt`).
- `dvc.yaml` — data pipeline configuration.
- `docker/Dockerfile` — image build instructions.

Notes & Recommendations
-----------------------
- Consider consolidating dependency management (poetry / pip-tools) or adding pinned versions in `requirements.txt`.
- CI currently runs minimal checks (lint + tests). Add `black .` (auto-format) and `ruff`/`flake8` if desired.
- Confirm Dockerfile targets the correct working directory and copies only needed artifacts for smaller images.
