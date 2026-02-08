# Phase 5: Classical Baseline Modeling - Research

**Researched:** 2026-02-08
**Domain:** Script-first classical ML baselines with reproducible evaluation artifacts
**Confidence:** HIGH

## User Constraints

No `*-CONTEXT.md` decisions were present for this phase at planning time.

## Summary

Phase 5 maps to `MODL-01`, `MODL-02`, and `MODL-04` and must deliver reproducible classical baselines with per-class metrics, confusion matrices, and reporting-ready metadata. The codebase already has a working baseline entrypoint (`scripts/run_experiment.py`) and shared evaluation helpers (`src/evaluation.py`) but currently only covers TF-IDF + Logistic Regression. Phase 5 should standardize baseline output contracts, add at least one tree/boosting model family, and add a comparison/orchestration layer that logs consistent run metadata.

**Primary recommendation:** plan this phase as 4 executable plans in 3 waves:
1. shared classical-baseline artifact/runner contract,
2. parallel implementation of logistic and tree/boosting baselines,
3. cross-model comparison and reporting integration.

## Existing Foundation To Reuse

| Existing Asset | Reuse Value |
|---|---|
| `scripts/run_experiment.py` | MLflow logging pattern and provenance tags are already established |
| `src/models.py` | Existing TF-IDF + Logistic Regression training helper |
| `src/evaluation.py` | Computes macro metrics and writes classification report + confusion matrix |
| `dvc.yaml` | Reproducibility wiring pattern for phase scripts |
| `tests/` suite conventions | Deterministic script-first tests already in place |

## Standard Stack

### Core
| Library/Tool | Purpose | Why Standard |
|---|---|---|
| `scikit-learn` pipelines | TF-IDF, linear models, tree/boosting families | already used and reproducible |
| `xgboost` or sklearn boosting | stronger non-linear baseline | satisfies `MODL-02` |
| `mlflow` | run metadata and metric tracking | existing project standard |
| `pandas`/`numpy` | feature and metric tables | established analysis tooling |
| DVC | deterministic stage reruns | already used across phases |

### Supporting
| Tool | Purpose | When to Use |
|---|---|---|
| `pytest` | artifact contract and metric-schema tests | each model family plan |
| JSON/CSV/Markdown outputs | report + dashboard consumption | every baseline run |
| Makefile target | canonical command for contributors | end-to-end baseline execution |

## Architecture Patterns

### Pattern 1: Shared Baseline Contract + Family-Specific Trainers
Define one output contract (metrics, per-class table, confusion matrix, run metadata) and keep model-family training code modular.

### Pattern 2: Deterministic Split/Seed Control
All model runs must pin split strategy, random seed, and explicit preprocessing so comparisons are fair and reproducible.

### Pattern 3: Comparison Summary As First-Class Artifact
Create a consolidated comparison table and chart across models (accuracy, macro-F1, per-class F1 deltas), not only individual run outputs.

### Anti-Patterns To Avoid
- Logging only aggregate metrics without per-class results.
- Ad-hoc model scripts with inconsistent output schemas.
- Comparing models trained on different data splits.
- Notebook-only baseline experiments.

## Open Questions For Execution

1. Which boosting baseline should be default for portability: `HistGradientBoostingClassifier` (no extra dependency) or `xgboost` (usually stronger)?
2. Should class imbalance handling be enabled by default (`class_weight="balanced"`) for all baselines?
3. Should prediction probabilities be exported for downstream threshold analysis in Phase 6/7?

## Planning Impact

- Use 4 plans in 3 waves to maximize parallel execution after shared contract setup.
- Require every baseline plan to produce `classification_report.json`, `confusion_matrix.json`, and machine-readable run metadata.
- Ensure DVC + MLflow integration is consistent and report/dashboard consumers can ingest outputs without custom adapters.
