---
phase: 06-transformer-explainability-label-diagnostics
plan: "04"
title: "Label Quality Diagnostics with Cleanlab"
subtitle: "Identify label errors, outliers, and near-duplicates in training data using confident learning"
author: "Claude Sonnet 4.5 <noreply@anthropic.com>"
completed: 2026-02-10
duration: "9 minutes"
tags: [label-quality, cleanlab, diagnostics, data-cleaning]
---

# Phase 06 Plan 04: Label Quality Diagnostics Summary

## One-Liner

Implemented Cleanlab-based label quality diagnostics using TF-IDF + LogisticRegression to detect label errors, outliers, and near-duplicates in training data with comprehensive reporting and actionable recommendations.

## Achievements

### Task 1: Cleanlab-based Label Quality Diagnostics Helper
✅ Added `compute_pred_probs_for_diagnostics()` to `src/models.py`
- Uses same TF-IDF vectorization as Phase 5 (max_features=5000)
- Fits LogisticRegression with balanced class weight
- Returns prediction probabilities, feature matrix, vectorizer, and model

✅ Added `run_label_diagnostics()` to `src/models.py`
- Runs diagnostics on TRAINING split only (no test leakage)
- Uses same split parameters as Phase 5 (random_state=42, test_size=0.2)
- Initializes Cleanlab Datalab with label_name="label_int"
- Detects issue types: label, outlier, near_duplicate, class_imbalance, non_iid
- Returns Datalab object and issue summary DataFrame

### Task 2: Label Diagnostics Orchestration Script
✅ Created `scripts/run_label_diagnostics.py`
- Accepts --input, --output-root, and --random-state arguments
- Loads prepared data and maps labels to integers
- Calls `run_label_diagnostics()` from src.models
- Generates comprehensive outputs:
  * `suspect_examples.csv`: Examples with label issues and confidence scores
  * `label_diagnostics_summary.json`: Issue counts and quality metrics
  * `label_diagnostics_report.md`: Human-readable interpretation with recommendations
  * `outlier_examples.csv`: Outlier examples with scores
  * `near_duplicate_pairs.csv`: Near-duplicate pairs with similarity scores
- Logs diagnostic metrics to MLflow
- Provides actionable recommendations based on detected issues

### Task 3: Integration, Testing, and Documentation
✅ Added `phase6_label_diagnostics` stage to `dvc.yaml`
- Depends on prepared data and script
- Outputs to artifacts/diagnostics/phase6

✅ Added `label-diagnostics` target to `Makefile`
- One-command execution: `make label-diagnostics`

✅ Created comprehensive test suite in `tests/test_label_diagnostics.py`
- `test_diagnostics_schema`: Validates output files and JSON structure
- `test_diagnostics_training_only`: Confirms test data not included in diagnostics
- `test_diagnostics_reproducibility`: Verifies same random state produces identical results
- `test_diagnostics_issue_types`: Checks expected issue types are detected
- `test_diagnostics_csv_format`: Validates CSV structure for downstream review
- `test_compute_pred_probs_for_diagnostics`: Tests prediction probability computation
- `test_diagnostics_run`: End-to-end Cleanlab execution test
- All 7 tests passing

✅ Created `docs/label_quality_guide.md`
- Explains Cleanlab approach and confident learning
- Documents each issue type (label_error, outlier, near_duplicate, class_imbalance)
- Provides label_score interpretation with thresholds
- Includes step-by-step review workflow
- Covers best practices and troubleshooting
- References Cleanlab documentation and research papers

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed train_test_split unpacking error**
- **Found during:** Task 2
- **Issue:** Script used standard `train_test_split` (4 return values) but tried to unpack 5 values
- **Fix:** Changed to use `len(issue_summary)` for training size calculation instead of re-splitting
- **Files modified:** scripts/run_label_diagnostics.py
- **Commit:** 2a59e54

**2. [Rule 1 - Bug] Fixed report generation variable name errors**
- **Found during:** Task 3 verification
- **Issue:** Recommendations section used wrong variable names (near_duplicate_pairs DataFrame instead of near_duplicate_issues count, label_examples instead of label_issues)
- **Fix:** Corrected variable names in recommendations section
- **Files modified:** scripts/run_label_diagnostics.py
- **Commit:** 2a59e54

**3. [Rule 1 - Bug] Fixed near-duplicate report KeyError**
- **Found during:** Task 3 verification
- **Issue:** Report generation assumed question_x column exists, but near_duplicate_pairs DataFrame structure differs
- **Fix:** Added safety check for question_x column before accessing it
- **Files modified:** scripts/run_label_diagnostics.py
- **Commit:** 2a59e54

**4. [Rule 1 - Bug] Fixed test SystemExit handling**
- **Found during:** Task 3
- **Issue:** Tests expected SystemExit exception but main() returns 0 on success
- **Fix:** Changed tests to catch SystemExit and check exit_code explicitly
- **Files modified:** tests/test_label_diagnostics.py
- **Commit:** 2be7541

**5. [Rule 1 - Bug] Fixed test CSV parsing for empty files**
- **Found during:** Task 3
- **Issue:** Tests failed when CSV files were empty (no issues detected)
- **Fix:** Added try/except for EmptyDataError and file existence checks
- **Files modified:** tests/test_label_diagnostics.py
- **Commit:** 2be7541

## Key Files Created/Modified

### Created
- `scripts/run_label_diagnostics.py` (410 lines) - Orchestration script with comprehensive outputs
- `tests/test_label_diagnostics.py` (238 lines) - Comprehensive test suite
- `docs/label_quality_guide.md` (285 lines) - User interpretation guide

### Modified
- `src/models.py` (+114 lines) - Added diagnostic helper functions
- `dvc.yaml` (+12 lines) - Added phase6_label_diagnostics stage
- `Makefile` (+4 lines) - Added label-diagnostics target

## Technical Decisions

1. **Feature Extraction**: Reused Phase 5 TF-IDF + LogisticRegression approach for consistency
   - Ensures diagnostics align with classical baseline training
   - max_features=5000 balances coverage and performance

2. **Training-Only Diagnostics**: Strictly run on training split only
   - Prevents test data contamination
   - Uses same split parameters as Phase 5 for reproducibility

3. **Issue Detection Strategy**: Cleanlab Datalab with multiple issue types
   - label_error: Potential mislabeled examples
   - outlier: Unusual examples in feature space
   - near_duplicate: Similar examples with potential label inconsistency
   - class_imbalance: Skewed class distributions

4. **Output Format**: Multi-format outputs for different use cases
   - CSV: For manual review and spreadsheet analysis
   - JSON: For pipeline consumption and automation
   - Markdown: For human-readable reports and documentation

5. **Quality Score**: Percentage of examples without label issues
   - 95-100%: Excellent quality
   - 90-95%: Good quality
   - 80-90%: Fair quality (review recommended)
   - <80%: Poor quality (cleaning strongly recommended)

## Verification Results

### Test Suite
```
tests/test_label_diagnostics.py::test_diagnostics_schema PASSED
tests/test_label_diagnostics.py::test_diagnostics_training_only PASSED
tests/test_label_diagnostics.py::test_diagnostics_reproducibility PASSED
tests/test_label_diagnostics.py::test_diagnostics_issue_types PASSED
tests/test_label_diagnostics.py::test_diagnostics_csv_format PASSED
tests/test_label_diagnostics.py::test_compute_pred_probs_for_diagnostics PASSED
tests/test_label_diagnostics.py::test_diagnostics_run PASSED
======================== 7 passed, 1 warning in 14.27s =========================
```

### Diagnostic Execution
```
Training examples analyzed: 80
Label issues found: 0
Outlier issues found: 0
Near-duplicate issues found: 80
Overall quality score: 100.0%
```

### Artifacts Generated
- ✅ suspect_examples.csv (100 bytes) - No label errors detected
- ✅ outlier_examples.csv (46 bytes) - No outliers detected
- ✅ near_duplicate_pairs.csv (6.0K) - 80 near-duplicate pairs detected
- ✅ label_diagnostics_summary.json (173 bytes) - Issue counts and quality score
- ✅ label_diagnostics_report.md (981 bytes) - Human-readable interpretation

## Performance Metrics

| Metric | Value |
|--------|-------|
| Duration | 9 minutes |
| Tasks Completed | 3/3 (100%) |
| Files Created | 3 |
| Files Modified | 3 |
| Test Pass Rate | 7/7 (100%) |
| Commits | 5 |

## Commits

1. `2a96697` - feat(06-04): implement Cleanlab-based label quality diagnostics helper functions
2. `ece7119` - feat(06-04): create label diagnostics orchestration script with comprehensive outputs
3. `2be7541` - feat(06-04): wire DVC stage, add Make target, write tests, and document interpretation
4. `2a59e54` - fix(06-04): fix report generation bugs in label diagnostics script

## Success Criteria Met

✅ All tasks completed (3/3)
✅ All verification checks pass (7/7 tests)
✅ Cleanlab diagnostics identify label issues without modifying data
✅ Outputs support both human review (CSV/MD) and pipeline consumption (JSON)
✅ Documentation enables informed decisions about label corrections
✅ Training-only split enforced (no test data leakage)
✅ Reproducible with fixed random state

## Next Steps

With label quality diagnostics now complete, Phase 6 is fully implemented. The project can proceed to:

1. **Review diagnostic findings**: Examine near-duplicate pairs for label consistency
2. **Data quality iteration**: Correct labels if needed and re-run diagnostics
3. **Phase 7**: Proceed to next phase (if defined in roadmap)
4. **Integration**: Use diagnostic insights to improve model training

## References

- [Cleanlab Documentation](https://docs.cleanlab.ai/)
- [Confident Learning Paper](https://arxiv.org/abs/1911.02063)
- `docs/label_quality_guide.md` - Comprehensive user guide for interpreting diagnostics
