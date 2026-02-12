# Model Optimization Guide (Phase 8)

Phase 8 adds a reproducible optimization workflow for model selection under class imbalance.

## Objective

- Primary metric: `macro_f1`
- Accuracy floor: `0.6431560071727436`
- Evaluation protocol: stratified holdout + stratified 5-fold CV on the training partition

## Run

```bash
make optimize-model
```

Or directly:

```bash
python scripts/run_model_optimization.py \
  --input data/processed/evasionbench_prepared.parquet \
  --output-root artifacts/models/phase8 \
  --families all \
  --cv-folds 5 \
  --selection-metric f1_macro \
  --accuracy-floor 0.6431560071727436
```

## Outputs

- `artifacts/models/phase8/optimization_trials.csv`
- `artifacts/models/phase8/cv_summary.json`
- `artifacts/models/phase8/selected_model.json`
- `artifacts/models/phase8/holdout_metrics.json`
- Winner artifacts under `artifacts/models/phase8/<family>/`

## Serving default

Model loading now resolves default model in this order:

1. `EVASION_MODEL_NAME` environment variable
2. `artifacts/models/phase8/selected_model.json`
3. fallback to `boosting`

## Integrity check

Use artifact integrity check before publishing report metrics:

```bash
make verify-artifacts
```
