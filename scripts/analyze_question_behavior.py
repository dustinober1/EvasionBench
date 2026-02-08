"""Generate phase-4 question-behavior artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.question_behavior import run_question_behavior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--emit-assignments", action="store_true")
    parser.add_argument("--emit-summary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    generated = run_question_behavior(
        frame,
        args.output_root,
        source_data=args.input,
        emit_assignments=args.emit_assignments,
        emit_summary=args.emit_summary or True,
    )
    print(f"Generated {len(generated)} question-behavior artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
