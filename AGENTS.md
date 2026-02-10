# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Script-First Workflow (Critical)

This project enforces script-first development. All executable logic must have a script entrypoint:
- **`scripts/`**: Executable workflow entrypoints (run these, not notebooks)
- **`src/`**: Implementation modules imported by scripts
- **`tests/`**: Contract and behavior tests

Notebooks in `notebooks/` are legacy/reference only. Do not add production logic to notebooks.

## Commands

```bash
make ci-check          # Full CI: structure verify + black + pytest
make test              # pytest -q
pytest tests/test_smoke.py -v   # Run single test file
pytest -k "test_name"  # Run tests matching pattern
make lint              # black --check .
make format            # black .
```

## Phase-Based Artifact Contracts

Analysis is organized into phases with strict artifact requirements:

- **Phase 3** (`artifacts/analysis/phase3/`): `core_stats`, `lexical`, `linguistic_quality`
- **Phase 4** (`artifacts/analysis/phase4/`): `semantic_similarity`, `topic_modeling`, `question_behavior`
- **Phase 5** (`artifacts/models/phase5/`): Model outputs

Model evaluation requires 4 files (see [`src/evaluation.py`](src/evaluation.py:17)):
- `metrics.json`, `classification_report.json`, `confusion_matrix.json`, `run_metadata.json`

## Key Patterns

- **Path setup in scripts**: All scripts use `ROOT = Path(__file__).resolve().parents[1]` and add to `sys.path`
- **Feature combination**: Question+answer combined with ` [SEP] ` token for TF-IDF (see [`src/models.py`](src/models.py:31))
- **Artifact index**: Each phase writes `artifact_index.json` via [`src/analysis/artifacts.py`](src/analysis/artifacts.py)
