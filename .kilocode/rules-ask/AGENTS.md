# Ask Mode Rules

## Project Documentation Context

- **Script-first policy**: Notebooks in `notebooks/` are legacy/reference only - canonical implementations are in `scripts/` with logic in `src/` (see [`docs/script_first_workflow.md`](docs/script_first_workflow.md))
- **Phase organization**: Analysis work is organized into phases 3-5, each with specific artifact requirements documented in [`src/analysis/artifacts.py`](src/analysis/artifacts.py)
- **Data pipeline**: Raw data → `data/raw/` → processed data → `data/processed/` (see `scripts/prepare_data.py`)

## Key Reference Files

- **Model training**: [`src/models.py`](src/models.py) - TF-IDF + LogReg, Tree, Boosting baselines
- **Evaluation utilities**: [`src/evaluation.py`](src/evaluation.py) - metrics computation and artifact writing
- **Analysis modules**: [`src/analysis/`](src/analysis/) - core_stats, lexical, linguistic_quality, qa_semantic, etc.
