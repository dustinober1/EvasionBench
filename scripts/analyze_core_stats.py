"""Generate phase-3 core statistics artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.core_stats import run_core_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--sections",
        default="quality,lengths",
        help="Comma-separated sections: quality,lengths",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    sections = [part.strip() for part in args.sections.split(",") if part.strip()]
    generated = run_core_stats(frame, args.output_root, source_data=args.input, sections=sections)
    print(f"Generated {len(generated)} core-stats artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
