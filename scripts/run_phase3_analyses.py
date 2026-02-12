"""Run all phase-3 analyses using script-first entrypoints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.artifacts import PHASE3_FAMILIES, ensure_phase3_layout

SCRIPT_BY_FAMILY = {
    "core_stats": Path("scripts/analyze_core_stats.py"),
    "lexical": Path("scripts/analyze_lexical.py"),
    "linguistic_quality": Path("scripts/analyze_linguistic_quality.py"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/evasionbench_prepared.parquet",
        help="Prepared parquet input for all analyses.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/analysis/phase3",
        help="Root directory for phase-3 artifacts.",
    )
    parser.add_argument(
        "--families",
        default="all",
        help="Comma separated analysis families, or 'all'.",
    )
    return parser.parse_args()


def _selected_families(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(PHASE3_FAMILIES)
    families = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(families) - set(PHASE3_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown families: {', '.join(unknown)}")
    return families


def run() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_root)

    ensure_phase3_layout(output_root)
    families = _selected_families(args.families)

    for family in families:
        script_path = SCRIPT_BY_FAMILY[family]
        command = [
            sys.executable,
            str(script_path),
            "--input",
            str(input_path),
            "--output-root",
            str(output_root),
        ]
        print(f"[phase3] running {family}: {' '.join(command)}")
        subprocess.run(command, check=True)

    print(f"Completed families: {', '.join(families)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
