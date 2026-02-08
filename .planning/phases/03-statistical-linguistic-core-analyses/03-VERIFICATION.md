status: passed
phase: 03
verified_on: 2026-02-08
score: 5/5

# Phase 03 Verification

## Must-Have Checks

1. Scripts generate labeled class distribution and data-quality tables/figures suitable for publication.  
   Result: Passed (`scripts/analyze_core_stats.py`, outputs under `artifacts/analysis/phase3/core_stats`).

2. Length distribution analyses include statistical test outputs and interpretation-ready summaries.  
   Result: Passed (`length_tests.json`, `length_interpretation.md`).

3. Lexical and n-gram analyses are reproducible and stratified by evasiveness label.  
   Result: Passed (`scripts/analyze_lexical.py`, per-label n-gram outputs).

4. Readability/POS/discourse marker analyses are generated with consistent chart/table formatting.  
   Result: Passed (`scripts/analyze_linguistic_quality.py`, outputs under `artifacts/analysis/phase3/linguistic_quality`).

5. All generated artifacts are versioned and consumed by report build steps.  
   Result: Passed (`artifact_index.json` and DVC stages `analyze_core_stats`, `analyze_lexical`, `analyze_linguistic_quality`, `analyze_phase3`).

## Executed Verification Commands
- `pytest -q tests/test_phase3_artifact_contract.py tests/test_core_stats_analysis.py tests/test_lexical_analysis.py tests/test_linguistic_quality_analysis.py`
- `python scripts/run_phase3_analyses.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/analysis/phase3 --families all`
- `dvc repro`
