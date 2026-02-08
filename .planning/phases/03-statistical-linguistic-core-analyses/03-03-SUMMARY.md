# 03-03 Summary

## Scope
Implemented lexical feature and n-gram analyses stratified by evasiveness label.

## Completed
- Added lexical module `src/analysis/lexical.py`.
- Added CLI entrypoint `scripts/analyze_lexical.py`.
- Added tests `tests/test_lexical_analysis.py`.
- Outputs include lexical summary tables, per-label unigram/bigram rankings, plots, and consolidated JSON.

## Verification
- `python scripts/analyze_lexical.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/analysis/phase3`
- `pytest -q tests/test_lexical_analysis.py`
- `python -m json.tool artifacts/analysis/phase3/lexical/top_ngrams.json >/dev/null`

## Notes
- N-gram ranking uses stable tie-breaking (`count desc`, `ngram asc`) for determinism.
