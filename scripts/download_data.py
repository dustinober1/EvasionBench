"""Download EvasionBench dataset deterministically and save to parquet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    DEFAULT_DATASET_ID,
    DEFAULT_OUTPUT,
    DEFAULT_REVISION,
    DEFAULT_SPLIT,
    download_evasionbench,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Parquet output path")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = download_evasionbench(
        output_path=Path(args.output),
        dataset_id=args.dataset_id,
        split=args.split,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    print("Download complete with deterministic source controls:")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
