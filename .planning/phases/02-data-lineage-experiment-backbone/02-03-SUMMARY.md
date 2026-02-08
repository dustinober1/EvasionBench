---
phase: 02-data-lineage-experiment-backbone
plan: "03"
subsystem: modeling
tags: [mlflow, experiment-tracking, baseline, metrics]
requires:
  - phase: 02-01
    provides: dataset manifest provenance fields
  - phase: 02-02
    provides: prepared data artifact
provides:
  - script-first baseline experiment runner
  - structured model/evaluation helpers with MLflow logging
  - test-backed verification of params/metrics/tags capture
affects: [modeling, reporting]
tech-stack:
  added: []
  patterns: [metadata-rich MLflow runs, deterministic train/eval interfaces]
key-files:
  created: [scripts/run_experiment.py, tests/test_mlflow_tracking.py]
  modified: [src/models.py, src/evaluation.py, docs/mlflow_guide.md, requirements.txt]
key-decisions:
  - "Log dataset checksum/revision and git SHA as required run tags."
  - "Keep baseline model deterministic with fixed random state and explicit feature build path."
patterns-established:
  - "single script command logs params, metrics, tags, and evaluation artifacts"
  - "evaluation artifacts serialized as JSON for pipeline/report portability"
duration: 60min
completed: 2026-02-08
---

# Phase 2 Plan 03 Summary

**MLflow-tracked baseline experiments now run from scripts with reproducibility tags and structured evaluation artifacts**

## Performance

- **Duration:** 60 min
- **Started:** 2026-02-08T21:05:00Z
- **Completed:** 2026-02-08T22:05:00Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Added `scripts/run_experiment.py` to run train/eval and log to MLflow.
- Implemented deterministic baseline training and structured metric/artifact emitters.
- Added MLflow tracking tests and practical runbook docs.

## Task Commits

1. **Task 1: Build script-first experiment runner with MLflow logging** - `fe74f57` (feat)
2. **Task 2: Wire model/evaluation modules for structured metric emission** - `fe74f57` (feat)
3. **Task 3: Add MLflow tracking tests and operational runbook updates** - `fe74f57` (feat)

**Plan metadata:** included in `docs(02-03): complete mlflow experiment tracking plan`

## Files Created/Modified

- `scripts/run_experiment.py` - experiment CLI with MLflow params/metrics/tags/artifacts logging.
- `src/models.py` - deterministic TF-IDF + Logistic Regression training helper.
- `src/evaluation.py` - metrics and artifact serialization helpers.
- `tests/test_mlflow_tracking.py` - validates required MLflow logging contract.
- `docs/mlflow_guide.md` - canonical command, UI usage, troubleshooting.

## Decisions Made

- Keep baseline implementation lightweight but fully script-driven for future phase reuse.
- Log provenance tags from manifest to avoid notebook-only experiment metadata paths.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] MLflow runtime missing `pkg_resources` in environment**
- **Found during:** test collection and experiment run verification
- **Issue:** MLflow import failed without compatible setuptools packaging shim.
- **Fix:** Pinned `setuptools<81` and verified import/test paths.
- **Files modified:** `requirements.txt`
- **Verification:** `pytest -q tests/test_mlflow_tracking.py`, `python scripts/run_experiment.py ...`
- **Committed in:** `fe74f57`

**2. [Rule 1 - Auto-fix Bug] Prepared labels were not mapped from source column**
- **Found during:** experiment run verification
- **Issue:** all labels collapsed to `unknown`, causing single-class training failure.
- **Fix:** mapped `eva4b_label` to `label` in `scripts/prepare_data.py`.
- **Files modified:** `scripts/prepare_data.py`
- **Verification:** `make data-prepare`, `python scripts/run_experiment.py ...`
- **Committed in:** `fe74f57`

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** fixes were required for runnable MLflow tracking with real labels.

## Issues Encountered

- None after environment/dependency and label-mapping fixes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Experiment tracking backbone is operational with provenance-rich metadata for downstream analyses.

---
*Phase: 02-data-lineage-experiment-backbone*
*Completed: 2026-02-08*
