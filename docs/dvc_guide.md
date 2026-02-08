# DVC Guide

Use DVC to track data and model artifacts.

Note: If `dvc` is not installed in your environment, install it (e.g., `pip install dvc`) before running the commands below.

1. Initialize DVC in the repository:
   ```bash
   dvc init
   ```

2. Add dataset (after running `scripts/download_data.py` or `dvc repro`):
   ```bash
   dvc add data/raw/evasionbench.parquet
   git add data/raw/evasionbench.parquet.dvc .gitignore
   git commit -m "Add raw EvasionBench dataset with DVC"
   ```

3. Or reproduce the provided pipeline stage (recommended):
   ```bash
   dvc repro dvc.yaml
   ```

4. Push to remote storage:
   ```bash
   dvc remote add -d storage s3://my-bucket/path
   dvc push
   ```

5. Reproduce pipeline:
   ```bash
   dvc repro
   ```
