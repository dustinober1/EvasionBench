---
phase: 05-classical-baseline-modeling
plan: "03"
subsystem: modeling
tags: [scikit-learn, mlflow, baselines, tree, boosting, evaluation]

# Dependency graph
requires:
  - phase: 05-classical-baseline-modeling
    provides: Shared phase-5 evaluation artifact contract and runner
provides:
  - Deterministic tree/boosting baseline trainer with metadata-rich artifacts
  - Tree/boosting family support in shared runner
  - Contract-focused tests for tree baselines
affects: [05-04, reporting]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Deterministic tree/boosting baselines with shared artifact contract metadata

key-files:
  created:
    - scripts/train_tree_baseline.py
    - tests/test_classical_tree_baseline.py
  modified:
    - src/models.py
    - scripts/run_classical_baselines.py

key-decisions:
  - "None - followed plan as specified"

patterns-established:
  - "Tree/boosting family runs share the phase-5 evaluation contract and metadata schema"

# Metrics
duration: 36m
completed: 2026-02-09
---

# Phase 5 Plan 03: Classical Baseline Modeling Summary

**Deterministic tree and boosting baselines now train via scripts and emit contract-compliant metrics, reports, and metadata.**

## Performance

- **Duration:** 36 min
- **Started:** 2026-02-09T00:44:24Z
- **Completed:** 2026-02-09T01:20:01Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- Added deterministic tree/boosting training utilities with explicit feature validation and metadata.
- Extended the shared classical runner to execute tree and boosting families together.
- Added contract and determinism tests for tree baseline outputs and failure modes.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add deterministic tree/boosting training path in model utilities** - `70045c6` (feat)
2. **Task 2: Emit full per-class evaluation artifacts for tree/boosting runs** - `72905c1` (feat)
3. **Task 3: Add tree/boosting baseline tests** - `8cf778c` (test)

**Plan metadata:** pending

## Files Created/Modified
- `/Users/dustinober/Projects/EvasionBench/src/models.py` - Validates feature columns and returns deterministic tree/boosting outputs.
- `/Users/dustinober/Projects/EvasionBench/scripts/train_tree_baseline.py` - Tree/boosting training script emitting contract artifacts.
- `/Users/dustinober/Projects/EvasionBench/scripts/run_classical_baselines.py` - Runner now maps tree family to tree + boosting.
- `/Users/dustinober/Projects/EvasionBench/tests/test_classical_tree_baseline.py` - Contract, determinism, and failure-mode tests.

## Decisions Made
None - followed plan as specified.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Validate required feature columns for tree baselines**
- **Found during:** Task 1 (Add deterministic tree/boosting training path in model utilities)
- **Issue:** Tree/boosting training previously allowed missing `question`/`answer` columns, which would silently degrade correctness.
- **Fix:** Added explicit feature column validation in shared feature builder and propagated errors.
- **Files modified:** `/Users/dustinober/Projects/EvasionBench/src/models.py`
- **Verification:** `pytest -q /Users/dustinober/Projects/EvasionBench/tests/test_classical_tree_baseline.py`
- **Committed in:** 70045c6 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Auto-fix ensured correctness without expanding scope.

## Issues Encountered
- MLflow emitted a filesystem tracking backend deprecation warning during verification; no functional impact on outputs.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Tree/boosting baselines are ready for cross-model comparison and reporting integration.
- Shared runner consistently emits contract artifacts across classical families.

## Self-Check: PASSED

---
*Phase: 05-classical-baseline-modeling*
*Completed: 2026-02-09*
