from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _fixture_parquet(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"question": "Q1", "answer": "A1", "label": "evasive"},
            {"question": "Q2", "answer": "A2", "label": "non_evasive"},
        ]
    )
    frame.to_parquet(path, index=False)


def test_write_manifest_has_required_fields(tmp_path: Path) -> None:
    data_path = tmp_path / "sample.parquet"
    manifest_path = tmp_path / "manifest.json"
    _fixture_parquet(data_path)

    cmd = [
        sys.executable,
        "scripts/write_data_manifest.py",
        "--data",
        str(data_path),
        "--output",
        str(manifest_path),
        "--revision",
        "pinned-rev",
    ]
    subprocess.run(cmd, check=True)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["dataset"]["revision"] == "pinned-rev"
    assert manifest["row_count"] == 2
    assert manifest["checksum_sha256"]
    assert manifest["schema"] == [
        {"name": "question", "dtype": "object"},
        {"name": "answer", "dtype": "object"},
        {"name": "label", "dtype": "object"},
    ]


def test_write_manifest_is_stable_for_same_input(tmp_path: Path) -> None:
    data_path = tmp_path / "sample.parquet"
    first_manifest = tmp_path / "manifest_one.json"
    second_manifest = tmp_path / "manifest_two.json"
    _fixture_parquet(data_path)

    base = [
        sys.executable,
        "scripts/write_data_manifest.py",
        "--data",
        str(data_path),
        "--revision",
        "abc123",
    ]
    subprocess.run([*base, "--output", str(first_manifest)], check=True)
    subprocess.run([*base, "--output", str(second_manifest)], check=True)

    assert first_manifest.read_text() == second_manifest.read_text()


def test_write_manifest_fails_for_missing_parquet(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"
    out = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/write_data_manifest.py",
            "--data",
            str(missing),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
