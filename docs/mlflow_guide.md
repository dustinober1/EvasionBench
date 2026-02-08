# MLflow Guide

Run script-first experiments and inspect tracked provenance locally.

## Canonical Runs

Single-family TF-IDF + Logistic Regression:

```bash
python scripts/run_experiment.py \
  --data data/processed/evasionbench_prepared.parquet \
  --output-root artifacts/models/phase5/logreg \
  --tracking-uri file:./mlruns \
  --experiment-name evasionbench-baselines
```

All Phase-5 classical families:

```bash
python scripts/run_classical_baselines.py \
  --input data/processed/evasionbench_prepared.parquet \
  --output-root artifacts/models/phase5 \
  --families all \
  --compare \
  --tracking-uri file:./mlruns \
  --experiment-name evasionbench-classical-baselines
```

Each family run logs:
- params (`model_family`, `target_col`, split sizes, random state, feature/model configs)
- metrics (`accuracy`, `f1_macro`, `precision_macro`, `recall_macro`)
- tags (`pipeline_stage`, `git_sha`, dataset provenance when provided)
- artifacts:
  - `metrics.json`
  - `classification_report.json`
  - `confusion_matrix.json`
  - `run_metadata.json`

## Open the UI

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Visit `http://localhost:5000` and open the latest runs under `evasionbench-baselines` or `evasionbench-classical-baselines`.

## Troubleshooting

- No runs visible: verify `--tracking-uri` matches `mlflow ui --backend-store-uri`.
- Missing provenance tags: confirm `data/contracts/evasionbench_manifest.json` exists when using `run_experiment.py`.
- Artifact logging issues: ensure the run has write access to the tracking directory.
