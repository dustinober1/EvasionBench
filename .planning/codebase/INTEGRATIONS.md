INTEGRATIONS
============

Overview
--------
This document lists external systems, APIs, and integrations used or referenced in the repository, and where integration code is located.

Datastores & Data
-----------------
- DVC-backed dataset: `dvc.yaml` and `data/raw/evasionbench.parquet` — data versioning with DVC; remote storage not specified in repo files (check `.dvc/config` or CI secrets).
- Local filesystem under `data/` for raw and processed artifacts.

Model/Artifact Hosting
----------------------
- `huggingface_hub` listed in `requirements.txt` — code likely uploads or downloads models from Hugging Face. Search for usage in `src/` if publishing models.

APIs & Services
---------------
- FastAPI app: `api/main.py` — serves model or evaluation endpoints; runs on `uvicorn`.
- MLflow: `mlflow` in `requirements.txt` and `docs/mlflow_guide.md` — requires an MLflow tracking server and artifact store when used in production.

CI/CD
-----
- GitHub Actions: `.github/workflows/ci.yml` — runs tests and linting. No deployment workflows found (no `deploy` job).

Third-party Libraries (integrations)
-----------------------------------
- HuggingFace Transformers & Hub (`transformers`, `huggingface_hub`) — model weights and tokenizers.
- Datasets (`datasets`) — dataset loading utilities; used in notebooks and scripts.
- Streamlit (`streamlit`) — dashboard integration (`dashboard/app.py`).

Auth & Secrets
--------------
- No explicit auth providers (OAuth, Auth0) found in repo files. Secrets may be referenced in CI secrets or local env files — none checked into repo.
- Review environment-related files (e.g., `.env`, CI secrets in repo settings) for API keys before publishing docs.

Webhooks
--------
- No webhook configurations found in `.github/` or other dirs.

Where integration code lives (examples)
--------------------------------------
- API: `api/main.py`
- Dashboard: `dashboard/app.py`
- Data pipeline: `dvc.yaml`, `scripts/download_data.py`
- Notebooks: `notebooks/` reference dataset and models (ad-hoc integrations)

Potential Action Items
----------------------
- Add documentation for remote DVC storage location if used (S3/GCS) — check environment or `dvc` config.
- If publishing models to Hugging Face, add a script and credentials guidance (`scripts/publish_model.py`).
