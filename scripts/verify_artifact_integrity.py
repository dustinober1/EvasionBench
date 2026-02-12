"""Validate model artifact integrity for production-like reporting runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase5-root",
        default="artifacts/models/phase5",
        help="Path containing phase5 family directories",
    )
    parser.add_argument(
        "--families",
        default="logreg,tree,boosting",
        help="Comma-separated families to validate",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=1000,
        help="Minimum expected train_rows for non-test artifacts",
    )
    parser.add_argument(
        "--expected-label-count",
        type=int,
        default=3,
        help="Expected minimum number of labels in run_metadata",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifacts(
    *,
    phase5_root: Path,
    families: list[str],
    min_train_rows: int,
    expected_label_count: int,
) -> list[str]:
    errors: list[str] = []

    for family in families:
        family_dir = phase5_root / family
        metadata_path = family_dir / "run_metadata.json"
        metrics_path = family_dir / "metrics.json"

        if not family_dir.exists():
            errors.append(f"Missing family directory: {family_dir}")
            continue
        if not metadata_path.exists():
            errors.append(f"Missing run_metadata.json: {metadata_path}")
            continue
        if not metrics_path.exists():
            errors.append(f"Missing metrics.json: {metrics_path}")
            continue

        metadata = _load_json(metadata_path)
        train_rows = int(metadata.get("train_rows", 0))
        labels = metadata.get("labels", [])

        if train_rows < min_train_rows:
            errors.append(
                f"{family}: train_rows={train_rows} below min_train_rows={min_train_rows}"
            )

        if not isinstance(labels, list) or len(labels) < expected_label_count:
            errors.append(
                f"{family}: labels={labels} below expected_label_count={expected_label_count}"
            )

    return errors


def main() -> int:
    args = parse_args()
    phase5_root = Path(args.phase5_root)
    families = [name.strip() for name in args.families.split(",") if name.strip()]

    errors = validate_artifacts(
        phase5_root=phase5_root,
        families=families,
        min_train_rows=args.min_train_rows,
        expected_label_count=args.expected_label_count,
    )

    if errors:
        print("Artifact integrity check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Artifact integrity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
