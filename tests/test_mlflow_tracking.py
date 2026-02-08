from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient


def _prepared_data(path: Path) -> None:
    rows = []
    for idx in range(20):
        label = "evasive" if idx % 2 == 0 else "non_evasive"
        rows.append(
            {
                "question": f"Question {idx}",
                "answer": f"Answer text {idx} with context",
                "label": label,
                "answer_length": 10 + idx,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _manifest(path: Path, data_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/write_data_manifest.py",
            "--data",
            str(data_path),
            "--output",
            str(path),
            "--revision",
            "test-revision",
        ],
        check=True,
    )


def test_mlflow_run_logs_params_metrics_tags(tmp_path: Path) -> None:
    data_path = tmp_path / "prepared.parquet"
    manifest_path = tmp_path / "manifest.json"
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = f"file:{tracking_dir}"

    _prepared_data(data_path)
    _manifest(manifest_path, data_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "--data",
            str(data_path),
            "--contract",
            str(manifest_path),
            "--tracking-uri",
            tracking_uri,
            "--experiment-name",
            "evasionbench-test",
        ],
        check=True,
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("evasionbench-test")
    assert experiment is not None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    assert runs
    run = runs[0]

    assert "model_type" in run.data.params
    assert "accuracy" in run.data.metrics
    assert run.data.tags.get("dataset_revision") == "test-revision"
    assert run.data.tags.get("dataset_checksum")
