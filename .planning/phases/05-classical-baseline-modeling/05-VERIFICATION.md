---
phase: 05-classical-baseline-modeling
verified: 2026-02-09T03:23:20Z
status: passed
score: 4/4 must-haves verified
---

# Phase 5: Classical Baseline Modeling Verification Report

**Phase Goal:** Deliver strong traditional ML baselines and comparison metrics.
**Verified:** 2026-02-09T03:23:20Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | TF-IDF + Logistic Regression baseline trains and evaluates from scripts. | ✓ VERIFIED | `/Users/dustinober/Projects/EvasionBench/scripts/run_classical_baselines.py` runs `train_tfidf_logreg` and writes artifacts via `write_evaluation_artifacts`; DVC stage `phase5_logreg` calls the script. |
| 2 | At least one tree/boosting baseline trains and evaluates from scripts. | ✓ VERIFIED | `/Users/dustinober/Projects/EvasionBench/scripts/train_tree_baseline.py` and `/Users/dustinober/Projects/EvasionBench/scripts/run_classical_baselines.py` invoke `train_tree_or_boosting` for `tree`/`boosting`; DVC stages `phase5_tree` and `phase5_boosting` exist. |
| 3 | Per-class metrics and confusion matrices are generated for every classical model run. | ✓ VERIFIED | `/Users/dustinober/Projects/EvasionBench/src/evaluation.py` writes `classification_report.json` and `confusion_matrix.json` and is called for logreg/tree/boosting in `scripts/run_classical_baselines.py` and `scripts/train_tree_baseline.py`. Artifact files present under `/Users/dustinober/Projects/EvasionBench/artifacts/models/phase5/*/`. |
| 4 | Outputs are stored with run metadata and consumable by reporting/dashboard layers. | ✓ VERIFIED | `run_metadata.json` written per family; comparison outputs (`model_ranking.csv`, `per_class_f1_comparison.csv`, `summary.json`, charts) exist under `/Users/dustinober/Projects/EvasionBench/artifacts/models/phase5/model_comparison` and documented in `/Users/dustinober/Projects/EvasionBench/docs/analysis_workflow.md`. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `/Users/dustinober/Projects/EvasionBench/scripts/run_classical_baselines.py` | Unified phase-5 runner with family selection | ✓ VERIFIED | Implements `--families` with logreg/tree/boosting, writes summary and optional comparison. |
| `/Users/dustinober/Projects/EvasionBench/scripts/run_experiment.py` | Logreg baseline script with artifact contract | ✓ VERIFIED | Trains TF-IDF logreg and writes evaluation artifacts. |
| `/Users/dustinober/Projects/EvasionBench/scripts/train_tree_baseline.py` | Tree/boosting baseline script | ✓ VERIFIED | Trains tree/boosting and writes evaluation artifacts. |
| `/Users/dustinober/Projects/EvasionBench/scripts/compare_classical_models.py` | Comparison aggregation outputs | ✓ VERIFIED | Writes ranking, per-class comparison, deltas, summary JSON, and charts. |
| `/Users/dustinober/Projects/EvasionBench/src/evaluation.py` | Canonical contract writer | ✓ VERIFIED | Writes metrics/report/confusion/metadata and validates required keys. |
| `/Users/dustinober/Projects/EvasionBench/src/models.py` | Deterministic model trainers | ✓ VERIFIED | Provides `train_tfidf_logreg` and `train_tree_or_boosting` with explicit params and split metadata. |
| `/Users/dustinober/Projects/EvasionBench/src/visualization.py` | Comparison charts | ✓ VERIFIED | Generates macro-F1 bar chart and per-class delta heatmap. |
| `/Users/dustinober/Projects/EvasionBench/dvc.yaml` | Phase-5 stages | ✓ VERIFIED | Includes `phase5_logreg`, `phase5_tree`, `phase5_boosting`, `phase5_compare`, `phase5`. |
| `/Users/dustinober/Projects/EvasionBench/tests/test_classical_baseline_contract.py` | Contract validation tests | ✓ VERIFIED | Ensures required files/keys. |
| `/Users/dustinober/Projects/EvasionBench/tests/test_classical_logreg_baseline.py` | Logreg baseline tests | ✓ VERIFIED | Verifies artifacts, metadata, determinism, failure cases. |
| `/Users/dustinober/Projects/EvasionBench/tests/test_classical_tree_baseline.py` | Tree baseline tests | ✓ VERIFIED | Verifies artifacts, determinism, failure cases. |
| `/Users/dustinober/Projects/EvasionBench/tests/test_phase5_model_comparison_artifacts.py` | Comparison artifact tests | ✓ VERIFIED | Validates comparison outputs and summary schema. |
| `/Users/dustinober/Projects/EvasionBench/docs/analysis_workflow.md` | Downstream documentation | ✓ VERIFIED | Documents phase-5 commands and artifact locations. |
| `/Users/dustinober/Projects/EvasionBench/docs/mlflow_guide.md` | MLflow guidance | ✓ VERIFIED | Documents phase-5 MLflow runs. |
| `/Users/dustinober/Projects/EvasionBench/artifacts/models/phase5/` | Generated outputs | ✓ VERIFIED | Family metrics/report/confusion/metadata and comparison outputs present. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `/Users/dustinober/Projects/EvasionBench/scripts/run_classical_baselines.py` | Family trainers | `train_tfidf_logreg` / `train_tree_or_boosting` | WIRED | Imports and calls both trainers with shared CLI args. |
| `/Users/dustinober/Projects/EvasionBench/scripts/run_classical_baselines.py` | Evaluation contract | `write_evaluation_artifacts` | WIRED | Writes metrics/report/confusion/metadata for each family. |
| `/Users/dustinober/Projects/EvasionBench/scripts/run_experiment.py` | Evaluation contract | `write_evaluation_artifacts` | WIRED | Logreg single-run writes contract outputs. |
| `/Users/dustinober/Projects/EvasionBench/scripts/train_tree_baseline.py` | Evaluation contract | `write_evaluation_artifacts` | WIRED | Tree/boosting writes contract outputs. |
| `/Users/dustinober/Projects/EvasionBench/scripts/compare_classical_models.py` | Comparison outputs | CSV/JSON + charts | WIRED | Reads family outputs and writes ranking, deltas, summary, charts. |
| `/Users/dustinober/Projects/EvasionBench/dvc.yaml` | Phase-5 stages | `phase5_*` stages | WIRED | Stage commands and dependencies defined. |

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|---|---|---|
| MODL-01 | ✓ SATISFIED | None |
| MODL-02 | ✓ SATISFIED | None |
| MODL-04 | ✓ SATISFIED | None |

### Anti-Patterns Found

None detected in core phase-5 scripts.

### Human Verification Required

1. **Model comparison chart quality**

**Test:** Open `/Users/dustinober/Projects/EvasionBench/artifacts/models/phase5/model_comparison/macro_f1_by_model.png` and `/Users/dustinober/Projects/EvasionBench/artifacts/models/phase5/model_comparison/per_class_f1_delta_heatmap.png`.
**Expected:** Charts are readable, labeled, and aligned with CSV values.
**Why human:** Visual quality cannot be verified programmatically.

2. **End-to-end pipeline command**

**Test:** Run `make model-phase5` or `python scripts/run_classical_baselines.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/models/phase5 --families all --compare`.
**Expected:** Commands complete without errors and refresh artifacts.
**Why human:** Execution environment dependencies and runtime behavior cannot be verified here.

---

_Verified: 2026-02-09T03:23:20Z_
_Verifier: Claude (gsd-verifier)_
