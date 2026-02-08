# MLflow Guide

Run script-first experiments and inspect tracked provenance locally.

## Canonical Run

```bash
python scripts/run_experiment.py \
  --tracking-uri file:./mlruns \
  --experiment-name evasionbench-baselines
```

The run logs:
- params (`model_type`, `target_col`, split sizes, random state)
- metrics (`accuracy`, `f1_macro`, `precision_macro`, `recall_macro`)
- tags (`dataset_checksum`, `dataset_revision`, `dataset_id`, `dataset_split`, `git_sha`)
- evaluation artifacts (classification report and confusion matrix)

## Open the UI

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Visit `http://localhost:5000` and open the latest run under `evasionbench-baselines`.

## Troubleshooting

- No runs visible: verify `--tracking-uri` matches `mlflow ui --backend-store-uri`.
- Missing provenance tags: confirm `data/contracts/evasionbench_manifest.json` exists.
- Artifact logging issues: ensure the run has write access to the tracking directory.
