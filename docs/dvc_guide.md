# DVC Guide

This project uses a script-first DVC pipeline for reproducible data lineage.

## Stage Graph

`download_data -> validate_data -> prepare_data`

- `download_data`: downloads raw data with explicit source controls.
- `validate_data`: fails fast if checksum/schema/row-count diverge from manifest.
- `prepare_data`: generates deterministic prepared parquet consumed by experiments.

## Canonical Commands

1. Fetch raw data and write manifest:

   ```bash
   make data-fetch
   ```

2. Validate dataset contract:

   ```bash
   make data-validate
   ```

3. Reproduce full lineage with DVC:

   ```bash
   dvc repro
   ```

4. Optional prep-only run:

   ```bash
   make data-prepare
   ```

## Validation Failures

- `Row count mismatch`: data file changed; re-run `make data-fetch`.
- `Checksum mismatch`: file drifted or source revision changed; re-fetch and regenerate contract.
- `Schema mismatch`: upstream schema changed; update contract only after intentional review.
