---
phase: 02-data-lineage-experiment-backbone
plan: "01"
subsystem: data
tags: [dataset, lineage, manifest, deterministic]
requires: []
provides:
  - deterministic dataset download command
  - machine-readable dataset manifest contract
  - contract regression tests for deterministic output
affects: [data-pipeline, dvc, experiments]
tech-stack:
  added: []
  patterns: [script-first data acquisition, checksum-backed data contract]
key-files:
  created: [scripts/write_data_manifest.py, tests/test_data_lineage_contract.py, data/contracts/evasionbench_manifest.json]
  modified: [src/data.py, scripts/download_data.py, Makefile, .gitignore]
key-decisions:
  - "Pin dataset source controls (id/split/revision/cache) in CLI and log effective values."
  - "Persist row count/schema/checksum in JSON for strict downstream validation."
patterns-established:
  - "data-fetch canonical command runs download + manifest generation"
  - "manifest serialization uses stable ordering for deterministic diffs"
duration: 65min
completed: 2026-02-08
---

# Phase 2 Plan 01 Summary

**Deterministic EvasionBench acquisition with a persisted schema/row/checksum contract and regression coverage**

## Performance

- **Duration:** 65 min
- **Started:** 2026-02-08T20:30:00Z
- **Completed:** 2026-02-08T21:35:00Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- Reworked data download into an explicit, deterministic CLI surface.
- Added manifest writer that records dataset provenance, checksum, row count, and schema.
- Added canonical `make data-fetch` and deterministic contract tests.

## Task Commits

1. **Task 1: Refactor dataset download for deterministic provenance** - `bc71bec` (feat)
2. **Task 2: Add data manifest writer for schema/row/checksum contract** - `bc71bec` (feat)
3. **Task 3: Expose canonical command and regression tests for data contract** - `bc71bec` (feat)

**Plan metadata:** included in `docs(02-01): complete deterministic data contract plan`

## Files Created/Modified

- `src/data.py` - deterministic loader API plus checksum helper.
- `scripts/download_data.py` - configurable script entrypoint with provenance logging.
- `scripts/write_data_manifest.py` - manifest generation CLI.
- `data/contracts/evasionbench_manifest.json` - canonical dataset contract artifact.
- `Makefile` - adds `data-fetch` target.
- `tests/test_data_lineage_contract.py` - manifest contract tests.

## Decisions Made

- Keep manifest field ordering and serialization stable for reproducible diffs.
- Use parquet-file SHA256 as the contract checksum of record.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Script entrypoints could not import `src` when run directly**
- **Found during:** verification commands
- **Issue:** `python scripts/*.py` failed with `ModuleNotFoundError: src`.
- **Fix:** Added repository-root bootstrap to script entrypoints.
- **Files modified:** `scripts/download_data.py`, `scripts/write_data_manifest.py`
- **Verification:** `python scripts/write_data_manifest.py --help`
- **Committed in:** `bc71bec`

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Required for script-first execution correctness; no scope expansion.

## Issues Encountered

- Parent-level `/data/` gitignore required force-adding manifest contract path.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Manifest contract is in place and consumable by strict validation and DVC stages.

---
*Phase: 02-data-lineage-experiment-backbone*
*Completed: 2026-02-08*
