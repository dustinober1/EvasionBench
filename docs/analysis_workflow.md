# Analysis Workflow

This project uses script-first analysis execution. No notebook is required for reproducible outputs.

## Phase 3 Command Surface

- Run all phase-3 analyses:
  - `make analysis-phase3`
  - or `python scripts/run_phase3_analyses.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/analysis/phase3 --families all`
- Run family-specific analyses:
  - `python scripts/analyze_core_stats.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/analysis/phase3`
  - `python scripts/analyze_lexical.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/analysis/phase3`
  - `python scripts/analyze_linguistic_quality.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/analysis/phase3`

## Artifact Contract

All phase-3 scripts write under `artifacts/analysis/phase3`:

- `core_stats/`
- `lexical/`
- `linguistic_quality/`
- `artifact_index.json`

`artifact_index.json` contains stage metadata and generated files to support downstream report assembly.

## DVC Reproducibility

Run all pipeline stages:

- `dvc repro`

Key phase-3 stages:

- `analyze_core_stats`
- `analyze_lexical`
- `analyze_linguistic_quality`
- `analyze_phase3`

## Phase 4 Command Surface

- Run all phase-4 analyses:
  - `make analysis-phase4`
  - or `python scripts/run_phase4_analyses.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/analysis/phase4 --families all`
- Run family-specific analyses:
  - `python scripts/analyze_qa_semantic.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/analysis/phase4`
  - `python scripts/analyze_topics.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/analysis/phase4 --topics 8 --seed 42`
  - `python scripts/analyze_question_behavior.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/analysis/phase4`

## Phase 4 Artifact Contract

All phase-4 scripts write under `artifacts/analysis/phase4`:

- `semantic_similarity/`
- `topic_modeling/`
- `question_behavior/`
- `artifact_index.json`

`artifact_index.json` enforces required metadata per stage:

- `metadata.hypothesis_summary` (pointer to hypothesis summary artifact)
- `metadata.analysis_version` (contract version marker)

This keeps Phase-4 outputs stable for downstream report generation and traceability.

Key phase-4 stages:

- `analyze_qa_semantic`
- `analyze_topics`
- `analyze_question_behavior`
- `analyze_phase4`

## spaCy Setup for Linguistic Quality

Install model once:

- `python -m spacy download en_core_web_sm`

If the model is missing, run the command above and rerun `scripts/analyze_linguistic_quality.py`.

## Phase 5 Command Surface

- Run all phase-5 classical baselines plus comparison:
  - `make model-phase5`
  - or `python scripts/run_classical_baselines.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/models/phase5 --families all --compare`
- Run family-specific baselines:
  - `python scripts/run_classical_baselines.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/models/phase5 --families logreg`
  - `python scripts/run_classical_baselines.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/models/phase5 --families tree`
  - `python scripts/run_classical_baselines.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/models/phase5 --families boosting`
- Run standalone comparison:
  - `python scripts/compare_classical_models.py --input-root artifacts/models/phase5 --output-root artifacts/models/phase5/model_comparison`

## Phase 5 Artifact Contract

Each family writes:

- `metrics.json`
- `classification_report.json`
- `confusion_matrix.json`
- `run_metadata.json`

under:

- `artifacts/models/phase5/logreg/`
- `artifacts/models/phase5/tree/`
- `artifacts/models/phase5/boosting/`

Comparison outputs live under `artifacts/models/phase5/model_comparison`:

- `model_ranking.csv`
- `per_class_f1_comparison.csv`
- `per_class_f1_delta.csv`
- `summary.json`
- `macro_f1_by_model.png`
- `per_class_f1_delta_heatmap.png`

`summary.json` contains:

- `best_model_family`
- `metric_deltas_vs_best`
- `artifacts.confusion_matrices` pointers for report/dashboard linking
- `models` map with per-family metrics and artifact roots

Key phase-5 stages:

- `phase5_logreg`
- `phase5_tree`
- `phase5_boosting`
- `phase5_compare`
- `phase5`
