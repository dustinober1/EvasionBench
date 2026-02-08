"""Deterministically prepare raw EvasionBench parquet for modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = ["question", "answer", "label"]


def prepare_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if "label" not in prepared.columns and "eva4b_label" in prepared.columns:
        prepared["label"] = prepared["eva4b_label"]

    for column in EXPECTED_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = "" if column != "label" else "unknown"

    prepared["question"] = prepared["question"].fillna("").astype(str)
    prepared["answer"] = prepared["answer"].fillna("").astype(str)
    prepared["label"] = prepared["label"].fillna("unknown").astype(str)
    prepared["answer_length"] = prepared["answer"].str.len()
    prepared = prepared.sort_index().reset_index(drop=True)
    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw parquet input")
    parser.add_argument("--output", required=True, help="Prepared parquet output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    frame = pd.read_parquet(input_path)
    prepared = prepare_dataframe(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(output_path, index=False)
    print(f"Prepared {len(prepared)} rows at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
