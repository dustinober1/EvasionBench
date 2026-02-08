"""Generate phase-4 Q-A semantic similarity artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.qa_semantic import run_qa_semantic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--emit-hypothesis-summary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    generated = run_qa_semantic(
        frame,
        args.output_root,
        source_data=args.input,
        emit_hypothesis_summary=args.emit_hypothesis_summary or True,
    )
    print(f"Generated {len(generated)} semantic artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
