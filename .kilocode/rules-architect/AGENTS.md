# Architect Mode Rules

## Architectural Constraints

- **Script-first enforcement**: All new features must have `scripts/` entrypoints - no production logic in notebooks
- **Phase isolation**: Each analysis phase (3, 4, 5) has independent artifact contracts - cross-phase dependencies must go through `artifact_index.json`
- **Model output contract**: Phase 5 model outputs require exactly 4 files validated by [`src/evaluation.py`](src/evaluation.py)

## Directory Structure

```
scripts/          # Executable entrypoints (run these)
src/              # Implementation modules
tests/            # Contract and behavior tests
artifacts/
  analysis/
    phase3/       # core_stats, lexical, linguistic_quality
    phase4/       # semantic_similarity, topic_modeling, question_behavior
  models/
    phase5/       # Model outputs with evaluation artifacts
```

## Extension Points

- **New analysis families**: Add to `PHASE3_FAMILIES` or `PHASE4_FAMILIES` in [`src/analysis/artifacts.py`](src/analysis/artifacts.py)
- **New model families**: Follow pattern in [`src/models.py`](src/models.py) - must return dict compatible with `write_evaluation_artifacts()`
