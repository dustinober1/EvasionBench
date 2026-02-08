"""Generate phase-4 topic-modeling artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.topic_modeling import run_topic_modeling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--topics", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emit-summary", action="store_true")
    parser.add_argument(
        "--no-emit-summary",
        action="store_true",
        help="Disable summary artifact generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    generated = run_topic_modeling(
        frame,
        args.output_root,
        source_data=args.input,
        topics=args.topics,
        seed=args.seed,
        emit_summary=not args.no_emit_summary,
    )
    print(f"Generated {len(generated)} topic-modeling artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
