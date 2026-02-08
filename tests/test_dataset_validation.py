from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data.parquet"
    manifest = tmp_path / "manifest.json"
    frame = pd.DataFrame(
        [
            {"question": "Q1", "answer": "A1", "label": "evasive"},
            {"question": "Q2", "answer": "A2", "label": "non_evasive"},
        ]
    )
    frame.to_parquet(data, index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/write_data_manifest.py",
            "--data",
            str(data),
            "--output",
            str(manifest),
            "--revision",
            "fixture",
        ],
        check=True,
    )
    return data, manifest


def test_validator_success(tmp_path: Path) -> None:
    data, manifest = _fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_dataset.py",
            "--data",
            str(data),
            "--contract",
            str(manifest),
        ]
    )
    assert result.returncode == 0


def test_validator_fails_for_row_mismatch(tmp_path: Path) -> None:
    data, manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["row_count"] = 999
    manifest.write_text(json.dumps(payload))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_dataset.py",
            "--data",
            str(data),
            "--contract",
            str(manifest),
        ]
    )
    assert result.returncode != 0


def test_validator_fails_for_checksum_mismatch(tmp_path: Path) -> None:
    data, manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["checksum_sha256"] = "deadbeef"
    manifest.write_text(json.dumps(payload))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_dataset.py",
            "--data",
            str(data),
            "--contract",
            str(manifest),
        ]
    )
    assert result.returncode != 0
