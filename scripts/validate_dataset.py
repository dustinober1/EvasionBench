"""Validate parquet data against the stored manifest contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import sha256_file


class ValidationError(RuntimeError):
    pass


def _schema_from_frame(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"name": col, "dtype": str(dtype)} for col, dtype in frame.dtypes.items()]


def validate_dataset(data_path: Path, contract_path: Path) -> None:
    if not data_path.exists():
        raise ValidationError(f"Data file not found: {data_path}")
    if not contract_path.exists():
        raise ValidationError(f"Contract file not found: {contract_path}")

    manifest = json.loads(contract_path.read_text())
    frame = pd.read_parquet(data_path)

    expected_rows = manifest.get("row_count")
    actual_rows = int(len(frame))
    if expected_rows != actual_rows:
        raise ValidationError(
            f"Row count mismatch: expected {expected_rows}, actual {actual_rows}"
        )

    expected_checksum = manifest.get("checksum_sha256")
    actual_checksum = sha256_file(data_path)
    if expected_checksum != actual_checksum:
        raise ValidationError(
            "Checksum mismatch: "
            f"expected {expected_checksum}, actual {actual_checksum}"
        )

    expected_schema = manifest.get("schema")
    actual_schema = _schema_from_frame(frame)
    if expected_schema != actual_schema:
        raise ValidationError(
            f"Schema mismatch: expected {expected_schema}, actual {actual_schema}"
        )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Parquet data path")
    parser.add_argument("--contract", required=True, help="Manifest contract JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_dataset(Path(args.data), Path(args.contract))
    except ValidationError as exc:
        print(f"VALIDATION FAILED: {exc}")
        return 1

    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
