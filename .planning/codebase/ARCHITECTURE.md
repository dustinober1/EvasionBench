ARCHITECTURE
============

High-level overview
-------------------
EvasionBench is organized as a research-first Python project that separates concerns into notebooks, source modules, an API, and visualization/dashboard components. Primary responsibilities:

- Data ingestion & versioning: `scripts/download_data.py`, `dvc.yaml`, `data/`.
- Training & models: `src/models.py`, `notebooks/*` for experiments.
- Feature engineering: `src/features.py`, `src/data.py`.
- Evaluation & explainability: `src/evaluation.py`, `src/explainability.py`.
- Serving: `api/main.py` (FastAPI) and `dashboard/app.py` (Streamlit) for demos.

Architectural patterns
----------------------
- Layered approach: data -> features -> models -> evaluation -> serving.
- Notebooks as exploratory layer: heavy use of `notebooks/` for experiments and human-in-the-loop analysis.
- Scripts and lightweight modules in `src/` provide reusable code for pipelines and evaluation.

Entry points
------------
- `notebooks/` — exploratory work and design; run interactively.
- `scripts/download_data.py` — initial data fetch.
- `api/main.py` — production/dev serving entry (via `uvicorn api.main:app`).
- `dashboard/app.py` — streamlit app entry (via `streamlit run dashboard/app.py`).

Data flow (prose + ASCII)
------------------------
1. Raw data acquisition: `scripts/download_data.py` -> `data/raw/` (DVC tracks these artifacts)
2. Data processing/features: `src/data.py` & `src/features.py` -> processed features
3. Model training/evaluation: `src/models.py`, `src/evaluation.py`, `notebooks/` -> trained models, metrics (optionally tracked by MLflow)
4. Serving: `api/main.py` or `dashboard/app.py` reads models (local or HF) and exposes endpoints/UI

Key abstractions & responsibilities (file pointers)
-----------------------------------------------
- Data & features: `src/data.py`, `src/features.py`
- Models: `src/models.py` (training pipelines and model wrappers)
- Evaluation: `src/evaluation.py`, `src/visualization.py` (plots, metrics)
- Explainability: `src/explainability.py` (SHAP / LIME hooks referenced in `requirements.txt`)
- Utilities: `src/utils.py` (helpers reused across modules)

Scalability & deployment notes
------------------------------
- Current setup is research-oriented; not hardened for high-scale production. For production use:
  - Separate training infra from serving (use model registry / artifact store).
  - Add a configuration mechanism (12-factor style env vars / config files).
  - Harden the FastAPI app with input validation, auth, and logging.

Diagrams & callouts
-------------------
Basic ASCII diagram:

  [raw data] --(dvc)--> [data/] ---> [src/data.py] ---> [src/features.py] ---> [src/models.py] ---> [trained model files]
                                                                                         |
                                                                                         v
                                                                                     [api/main.py]

Notes
-----
- The architecture favors reproducibility (DVC + CI) but requires explicit remote config for DVC and MLflow to be production-ready.
