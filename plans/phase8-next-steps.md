# Phase 8 Next Steps Plan (Handoff)

## Goal
Improve serving quality after selecting the relaxed-floor winner (`logreg_0003`) while keeping the workflow reproducible and safe for CI.

## Current baseline (as of this handoff)
- Winner artifact: `artifacts/models/phase8/selected_model.json`
- Selected model: `logreg_0003`
- Holdout metrics:
  - accuracy: `0.6389719067543336`
  - macro_f1: `0.54469754724448`
  - precision_macro: `0.5582350324499816`
  - recall_macro: `0.5343824216923796`
- Selection floor used: `0.63`

## Scope
- In scope:
  - lock serving to selected winner
  - remove solver deprecation risk
  - improve accuracy without losing macro-F1 gains
  - validate stability across seeds
- Out of scope:
  - transformer tuning
  - label relabeling workflow

## Phase A: Lock Serving to Current Winner
1. Verify the runtime reads `artifacts/models/phase8/selected_model.json` and defaults to that winner.
2. Run API and dashboard smoke checks using the selected model.
3. Record observed health output and one sample prediction payload.

### Commands
```bash
python - <<'PY'
from src.inference import load_model
p = load_model()
print(type(p).__name__, getattr(p, "model_name", None))
PY

pytest tests/test_smoke.py -v
```

### Acceptance criteria
- `load_model()` resolves to `logreg`
- `tests/test_smoke.py` passes

## Phase B: Remove Solver Deprecation Risk (liblinear)
1. Update optimization candidate solver to `lbfgs` (or `saga` with adequate `max_iter`) for multiclass safety.
2. Add/adjust `max_iter` to avoid convergence warnings.
3. Re-run the pruned sweep and compare metrics.

### Required code changes
- File: `scripts/run_model_optimization.py`
- Change:
  - `solver_options` from `liblinear` to `lbfgs`
  - add explicit `max_iter` candidate setting (`2000` baseline)

### Commands
```bash
black scripts/run_model_optimization.py
ruff check scripts/run_model_optimization.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_model_optimization.py \
  --input data/processed/evasionbench_prepared.parquet \
  --output-root artifacts/models/phase8 \
  --families all \
  --selection-metric f1_macro \
  --accuracy-floor 0.63
```

### Acceptance criteria
- no solver deprecation warnings for selected logreg candidates
- new selected model written to `artifacts/models/phase8/selected_model.json`

## Phase C: Targeted Accuracy Recovery Around Winning Region
1. Keep candidate count small and focused around winner-like settings.
2. Evaluate only logreg neighborhood with small grid:
  - `C`: `[0.2, 0.3, 0.5, 0.8]`
  - `class_weight`: `balanced` and `{direct:1.0, intermediate:1.2, fully_evasive:2.2}`
  - `min_df`: `[1, 2, 3]`
  - `max_features`: `[8000, 10000, 15000]`
3. Keep `cv_folds=3` for runtime control.

### Commands
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_model_optimization.py \
  --input data/processed/evasionbench_prepared.parquet \
  --output-root artifacts/models/phase8 \
  --families logreg \
  --selection-metric f1_macro \
  --accuracy-floor 0.63 \
  --cv-folds 3
```

### Acceptance criteria
- macro-F1 remains `>= 0.544`
- accuracy improves to `>= 0.640` (target) with same or better macro-F1

## Phase D: Stability Check Across Seeds
1. Run identical sweep for seeds `[21, 42, 84]`.
2. Save each run under distinct roots:
  - `artifacts/models/phase8_seed21`
  - `artifacts/models/phase8_seed42`
  - `artifacts/models/phase8_seed84`
3. Compare selected metrics variance.

### Commands
```bash
for seed in 21 42 84; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_model_optimization.py \
    --input data/processed/evasionbench_prepared.parquet \
    --output-root artifacts/models/phase8_seed${seed} \
    --families logreg \
    --selection-metric f1_macro \
    --accuracy-floor 0.63 \
    --cv-folds 3 \
    --random-state ${seed}
done
```

### Acceptance criteria
- std-dev of macro-F1 across selected runs `< 0.015`
- no run drops below accuracy floor `0.63`

## Phase E: Finalize and Publish
1. Copy best run into canonical `artifacts/models/phase8` if best run was produced in seed-specific root.
2. Ensure `selected_model.json` paths are canonical (`artifacts/models/phase8/...`).
3. Run CI checks.

### Commands
```bash
make ci-check
```

### Acceptance criteria
- `make ci-check` passes
- canonical winner is resolvable by `src/inference.py`

## Risks and mitigations
- Risk: long-running sweeps in constrained environment.
  - Mitigation: keep pruned grid + 3-fold CV + BLAS thread caps.
- Risk: accuracy/macro-F1 tradeoff swings winner unexpectedly.
  - Mitigation: use explicit floor and seed stability gate.

## Deliverables checklist
- [ ] Updated optimization solver settings in `scripts/run_model_optimization.py`
- [ ] Refreshed `artifacts/models/phase8/*` from successful run
- [ ] Seed stability artifacts (`phase8_seed21/42/84`)
- [ ] Passing `make ci-check`
- [ ] Brief summary note in next context with final selected metrics
