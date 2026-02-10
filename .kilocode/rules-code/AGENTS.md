# Code Mode Rules

## Project-Specific Coding Patterns

- **Path setup pattern**: All scripts must use `ROOT = Path(__file__).resolve().parents[1]` and add to `sys.path` before importing from `src/`
- **Feature combination**: Question+answer text uses ` [SEP] ` separator for TF-IDF vectorization (see [`src/models.py:31`](src/models.py:31))
- **Artifact index**: Phase analyses must call `write_artifact_index()` or `write_phase4_artifact_index()` from [`src/analysis/artifacts.py`](src/analysis/artifacts.py)
- **Evaluation contract**: Model outputs require 4 files: `metrics.json`, `classification_report.json`, `confusion_matrix.json`, `run_metadata.json` (validated by [`src/evaluation.py`](src/evaluation.py))

## Phase Family Constants

- Phase 3: `("core_stats", "lexical", "linguistic_quality")`
- Phase 4: `("semantic_similarity", "topic_modeling", "question_behavior")`

These are defined in [`src/analysis/artifacts.py`](src/analysis/artifacts.py) - do not hardcode elsewhere.
