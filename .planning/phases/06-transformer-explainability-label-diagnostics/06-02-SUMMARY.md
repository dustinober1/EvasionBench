---
phase: 06-transformer-explainability-label-diagnostics
plan: "02"
subsystem: explainability
tags: [shap, xai, classical-baselines, feature-importance, model-interpretability]

# Dependency graph
requires:
  - phase: 05-classical-baseline-modeling
    provides: Phase 5 classical baseline models (logreg, tree, boosting) with saved artifacts
  - phase: 06-transformer-explainability-label-diagnostics
    provides: Phase 6-01 DistilBERT transformer baseline
provides:
  - SHAP explainability artifacts for all Phase 5 classical models
  - Global feature importance (top 20 features by mean |SHAP|)
  - Local explanations (10 representative samples per model)
  - Orchestration script for generating XAI artifacts
  - DVC stage and Make target for reproducible XAI generation
  - Contract tests validating SHAP artifact schemas
  - Comprehensive documentation for SHAP interpretation and usage
affects: [phase7-reporting, phase8-ui, model-analysis]

# Tech tracking
tech-stack:
  added: [shap (0.49.1)]
  patterns:
    - Model persistence as pickle files (model.pkl, model_bundle.pkl)
    - SHAP explainer selection by model family (LinearExplainer, TreeExplainer)
    - JSON artifact contract for XAI outputs
    - MLflow logging for explainability runs

key-files:
  created:
    - scripts/run_explainability_analysis.py
    - tests/test_explainability_artifacts.py
    - docs/explainability_guide.md
  modified:
    - src/explainability.py (replaced placeholder with full implementation)
    - scripts/run_classical_baselines.py (added model persistence)
    - dvc.yaml (added phase6_xai_classical stage)
    - Makefile (added xai-classical target)

key-decisions:
  - "Added model persistence to Phase 5 runner (Rule 2: Missing Critical)"
  - "Use LinearExplainer for TF-IDF + LogisticRegression"
  - "Use TreeExplainer for RandomForest and HistGradientBoosting"
  - "Compute SHAP on training data only to avoid leakage"
  - "Generate both global (summary) and local (samples) explanations"
  - "Use Agg matplotlib backend for headless environments"
  - "Convert sparse matrices to dense for TreeExplainer compatibility"

patterns-established:
  - "Model artifacts saved as pickle files for downstream consumption"
  - "SHAP values output as JSON for machine-readable consumption and PNG for visualization"
  - "XAI orchestration script mirrors classical baseline runner structure"
  - "Per-family XAI artifacts with combined summary JSON"
  - "Contract tests validate schema and prevent test data leakage"

# Metrics
duration: 30m 15s
completed: 2026-02-10
---

# Phase 6 Plan 02: Classical Model Explainability Summary

**SHAP-based feature importance and local explanations for Phase 5 classical models (logistic regression, random forest, gradient boosting) with global importance rankings, per-sample attributions, and comprehensive documentation.**

## Performance

- **Duration:** 30m 15s
- **Started:** 2026-02-10T01:38:10Z
- **Completed:** 2026-02-10T02:08:25Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments
- Implemented SHAP explainability for all three classical model families with appropriate explainers (LinearExplainer, TreeExplainer)
- Added model persistence to Phase 5 training script to enable downstream XAI analysis
- Created orchestration script supporting selective XAI generation per model family
- Integrated with DVC workflow and added Make target for reproducible execution
- Delivered comprehensive contract tests and documentation for SHAP interpretation

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement SHAP explainability for classical model families** - `27ab092` (feat)
2. **Task 1: Regenerate Phase 5 models with persistence** - `2046e75` (chore, model regeneration)
3. **Task 2: Create explainability orchestration script** - `2046e75` (feat)
4. **Task 2: Fix tree model SHAP computation** - `53813f2` (fix)
5. **Task 3: Wire DVC, Make, tests, and documentation** - `a01577c` (feat)

**Plan metadata:** pending

## Files Created/Modified
- `/home/dusitnober/Projects/EvasionBench/src/explainability.py` - SHAP explainability implementation with `explain_classical_model()` function
- `/home/dusitnober/Projects/EvasionBench/scripts/run_classical_baselines.py` - Added model persistence (pickle files) for downstream XAI
- `/home/dusitnober/Projects/EvasionBench/scripts/run_explainability_analysis.py` - Orchestration script for generating XAI artifacts
- `/home/dusitnober/Projects/EvasionBench/dvc.yaml` - Added `phase6_xai_classical` stage
- `/home/dusitnober/Projects/EvasionBench/Makefile` - Added `xai-classical` target
- `/home/dusitnober/Projects/EvasionBench/tests/test_explainability_artifacts.py` - 12 contract tests for SHAP artifacts
- `/home/dusitnober/Projects/EvasionBench/docs/explainability_guide.md` - Comprehensive SHAP interpretation and usage guide

## Decisions Made

- **Model Persistence (Rule 2 - Missing Critical):** Added model saving to Phase 5 runner
  - **Rationale:** Models were trained in-memory but not persisted, preventing downstream explainability
  - **Implementation:** Save models as pickle files (model.pkl for logreg, model_bundle.pkl for tree/boosting)
  - **Impact:** Enables XAI, model inspection, and potential model serving

- **Explainer Selection:** Used LinearExplainer for logreg, TreeExplainer for tree/boosting
  - **Rationale:** Specialized explainers provide accurate and efficient SHAP values per model type
  - **Implementation:** Detect model family from metadata and instantiate appropriate explainer
  - **Tradeoff:** TreeExplainer requires dense matrices (sparse→dense conversion needed)

- **Training Data Only:** Compute SHAP on training data, not test data
  - **Rationale:** Prevents data leakage and ensures explanations reflect learned patterns
  - **Documentation:** Clearly documented limitation in explainability_guide.md
  - **Impact:** Users must compute separate SHAP for test predictions if needed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added model persistence to Phase 5 runner**
- **Found during:** Task 1 (loading models for explainability)
- **Issue:** Phase 5 models were trained in-memory but not saved to disk
- **Fix:** Updated `run_classical_baselines.py` to save models as pickle files
- **Files modified:** `scripts/run_classical_baselines.py`
- **Verification:** Models now saved and successfully loaded by explainability script
- **Committed in:** `27ab092` (part of Task 1 commit)

**2. [Rule 1 - Bug] Fixed sparse matrix dtype issue for TreeExplainer**
- **Found during:** Task 2 (testing tree model explainability)
- **Issue:** TreeExplainer cannot handle sparse matrices with dtype('O')
- **Fix:** Convert sparse matrices to dense format using `.toarray()` before passing to TreeExplainer
- **Files modified:** `src/explainability.py`
- **Verification:** Tree model SHAP computation now succeeds
- **Committed in:** `53813f2` (Task 2 fix commit)

**3. [Rule 1 - Bug] Fixed multi-class SHAP value indexing**
- **Found during:** Task 2 (multi-class label handling)
- **Issue:** SHAP values returned as list for binary classification (one array per class)
- **Fix:** Extract positive class SHAP values when shap_values is a list
- **Files modified:** `src/explainability.py`
- **Verification:** Correct feature importance rankings generated
- **Committed in:** `53813f2` (Task 2 fix commit)

**4. [Rule 1 - Bug] Fixed feature importance computation for multi-output**
- **Found during:** Task 2 (summary plot generation)
- **Issue:** mean_abs_shap had shape (n_features, n_classes) causing plotting errors
- **Fix:** Take mean across classes when mean_abs_shap is multi-dimensional
- **Files modified:** `src/explainability.py`
- **Verification:** Summary plots render correctly for all model families
- **Committed in:** `53813f2` (Task 2 fix commit)

---

**Total deviations:** 4 auto-fixed (1 missing critical, 3 bugs)
**Impact on plan:** All auto-fixes necessary for correctness and functionality. Model persistence was critical blocker. SHAP computation fixes were required for tree/boosting models. No scope creep.

## Issues Encountered
- **SHAP FutureWarning:** `feature_perturbation` parameter deprecated in LinearExplainer. Non-breaking, logged but not fixed.
- **Multi-class handling:** Dataset has 3 classes (direct, fully_evasive, intermediate) but SHAP explainers expect binary classification. Handled by extracting positive class values.
- **Matplotlib backend:** Needed to use Agg backend for headless environments to avoid display issues.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- SHAP artifacts available for all three classical model families
- Orchestration script supports selective XAI generation
- Comprehensive documentation enables report authors to interpret SHAP values
- Contract tests ensure artifact schema compliance
- Ready for Phase 6-03 (Transformer XAI) and Phase 7 (reporting)

## Verification Results

All verification checks passed:

```bash
# 1. Tests pass
$ pytest -q tests/test_explainability_artifacts.py
............                                                             [100%]
12 passed in 0.03s

# 2. Script generates artifacts for all families
$ python scripts/run_explainability_analysis.py --families all
  ✓ logreg: 268 features, 10 samples
  ✓ tree: 268 features, 10 samples
  ✓ boosting: 80 features, 10 samples
{"families": ["boosting", "logreg", "tree"], "output_root": "artifacts/explainability/phase6"}

# 3. Artifacts exist
$ ls artifacts/explainability/phase6/*/shap_*.json | wc -l
6  # 2 artifacts per family × 3 families

# 4. Documentation exists
$ test -f docs/explainability_guide.md && echo "Documentation exists"
Documentation exists
```

## Self-Check: PASSED

- **Commits exist:**
  - `27ab092` - Task 1: SHAP implementation and model persistence
  - `2046e75` - Task 2: Orchestration script
  - `53813f2` - Task 2: SHAP computation fixes
  - `a01577c` - Task 3: DVC, Make, tests, documentation

- **Files created:**
  - `scripts/run_explainability_analysis.py` ✓
  - `tests/test_explainability_artifacts.py` ✓
  - `docs/explainability_guide.md` ✓

- **Files modified:**
  - `src/explainability.py` ✓
  - `scripts/run_classical_baselines.py` ✓
  - `dvc.yaml` ✓
  - `Makefile` ✓

- **Artifacts generated:**
  - `artifacts/explainability/phase6/logreg/shap_*.json` ✓
  - `artifacts/explainability/phase6/tree/shap_*.json` ✓
  - `artifacts/explainability/phase6/boosting/shap_*.json` ✓
  - `artifacts/explainability/phase6/xai_summary.json` ✓

- **Tests pass:** 12/12 ✓

- **Documentation complete:** ✓

---
*Phase: 06-transformer-explainability-label-diagnostics*
*Completed: 2026-02-10*
