"""Generate dataset manifest containing lineage and integrity contract."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import DEFAULT_DATASET_ID, DEFAULT_REVISION, DEFAULT_SPLIT, sha256_file


def _stable_generated_timestamp(data_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    mtime = datetime.fromtimestamp(data_path.stat().st_mtime, tz=timezone.utc)
    return mtime.isoformat().replace("+00:00", "Z")


def generate_manifest(
    data_path: Path,
    output_path: Path,
    dataset_id: str,
    split: str,
    revision: str,
    generated_at_utc: str | None = None,
) -> Dict[str, Any]:
    frame = pd.read_parquet(data_path)
    manifest = {
        "dataset": {
            "dataset_id": dataset_id,
            "split": split,
            "revision": revision,
        },
        "generated_at_utc": _stable_generated_timestamp(data_path, generated_at_utc),
        "row_count": int(len(frame)),
        "checksum_sha256": sha256_file(data_path),
        "schema": [
            {"name": column, "dtype": str(dtype)}
            for column, dtype in frame.dtypes.items()
        ],
        "source_file": str(data_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Manifest output path")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--generated-at-utc",
        default=None,
        help="Optional explicit timestamp for deterministic fixtures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate_manifest(
        data_path=Path(args.data),
        output_path=Path(args.output),
        dataset_id=args.dataset_id,
        split=args.split,
        revision=args.revision,
        generated_at_utc=args.generated_at_utc,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
