"""Generate readability, POS, and discourse artifacts for phase 3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.linguistic_quality import run_linguistic_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--sections",
        default="readability,pos,discourse",
        help="Comma-separated sections: readability,pos,discourse",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    sections = [part.strip() for part in args.sections.split(",") if part.strip()]
    generated = run_linguistic_quality(
        frame,
        args.output_root,
        source_data=args.input,
        sections=sections,
    )
    print(f"Generated {len(generated)} linguistic-quality artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
