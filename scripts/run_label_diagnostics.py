"""Run Cleanlab label quality diagnostics on training data.

This script identifies potential label errors, outliers, and near-duplicates
in the training data using Cleanlab's Datalab. It runs on the training split
only to prevent test data contamination.

Outputs:
- suspect_examples.csv: Examples flagged with label issues
- label_diagnostics_summary.json: Issue counts and quality scores
- label_diagnostics_report.md: Human-readable interpretation
- outlier_examples.csv: Examples flagged as outliers
- near_duplicate_pairs.csv: Detected near-duplicate pairs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import run_label_diagnostics


def _git_sha() -> str:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared parquet dataset")
    parser.add_argument("--output-root", required=True, help="Root output directory for diagnostic artifacts")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="evasionbench-label-diagnostics", help="MLflow experiment name")
    return parser.parse_args()


def _create_markdown_report(
    output_root: Path,
    issue_summary: pd.DataFrame,
    suspect_examples: pd.DataFrame,
    outlier_examples: pd.DataFrame,
    near_duplicate_pairs: pd.DataFrame,
    train_size: int,
) -> Path:
    """Create human-readable markdown report."""
    report_path = output_root / "label_diagnostics_report.md"

    # Count issues by type
    label_issues = issue_summary["is_label_issue"].sum() if "is_label_issue" in issue_summary.columns else 0
    outlier_issues = issue_summary["is_outlier_issue"].sum() if "is_outlier_issue" in issue_summary.columns else 0
    near_duplicate_issues = (
        issue_summary["is_near_duplicate_issue"].sum() if "is_near_duplicate_issue" in issue_summary.columns else 0
    )

    # Calculate quality score (percentage of examples without issues)
    clean_examples = train_size - label_issues
    quality_score = (clean_examples / train_size * 100) if train_size > 0 else 0

    lines = [
        "# Label Quality Diagnostics Report\n",
        "## Overview",
        "",
        f"**Training Set Size:** {train_size} examples",
        f"**Overall Label Quality Score:** {quality_score:.1f}%",
        "",
        "## Issue Summary",
        "",
        "| Issue Type | Count | Percentage |",
        "|------------|-------|------------|",
        f"| Label Errors | {label_issues} | {label_issues/train_size*100:.1f}% |" if train_size > 0 else "| Label Errors | 0 | 0% |",
        f"| Outliers | {outlier_issues} | {outlier_issues/train_size*100:.1f}% |" if train_size > 0 else "| Outliers | 0 | 0% |",
        f"| Near Duplicates | {near_duplicate_issues} | {near_duplicate_issues/train_size*100:.1f}% |"
        if train_size > 0
        else "| Near Duplicates | 0 | 0% |",
        "",
    ]

    # Add suspect examples section
    if not suspect_examples.empty:
        lines.extend([
            "## Suspect Label Examples",
            "",
            f"Found **{len(suspect_examples)}** examples with potential label issues.",
            "",
            "### Top 10 Most Suspect Examples",
            "",
            "| Rank | Label | Label Score | Issue Types | Question |",
            "|------|-------|-------------|-------------|----------|",
        ])

        for idx, row in suspect_examples.head(10).iterrows():
            question = str(row["question"])[:60] + "..." if len(str(row["question"])) > 60 else str(row["question"])
            issue_types = []
            if row.get("is_label_issue", False):
                issue_types.append("label_error")
            if row.get("is_outlier_issue", False):
                issue_types.append("outlier")
            if row.get("is_near_duplicate_issue", False):
                issue_types.append("near_duplicate")

            lines.append(
                f"| {idx+1} | {row['label']} | {row.get('label_score', 'N/A'):.3f} | {', '.join(issue_types)} | {question} |"
            )

        lines.append("")
    else:
        lines.extend([
            "## Suspect Label Examples",
            "",
            "No suspect examples found. All labels appear to be correct.",
            "",
        ])

    # Add outlier section
    if not outlier_examples.empty:
        lines.extend([
            "## Outlier Examples",
            "",
            f"Found **{len(outlier_examples)}** outlier examples that may represent edge cases.",
            "",
            "### Top 5 Outliers",
            "",
            "| Rank | Label | Outlier Score | Question |",
            "|------|-------|---------------|----------|",
        ])

        for idx, row in outlier_examples.head(5).iterrows():
            question = str(row["question"])[:60] + "..." if len(str(row["question"])) > 60 else str(row["question"])
            lines.append(f"| {idx+1} | {row['label']} | {row.get('outlier_score', 'N/A'):.3f} | {question} |")

        lines.append("")
    else:
        lines.extend([
            "## Outlier Examples",
            "",
            "No outliers detected.",
            "",
        ])

    # Add near-duplicate section
    if not near_duplicate_pairs.empty and "question_x" in near_duplicate_pairs.columns:
        lines.extend([
            "## Near-Duplicate Pairs",
            "",
            f"Found **{len(near_duplicate_pairs)}** near-duplicate pairs that may have inconsistent labels.",
            "",
            "### Top 5 Near-Duplicate Pairs",
            "",
            "| Rank | Question 1 | Answer 1 | Label 1 | Question 2 | Answer 2 | Label 2 | Distance Score |",
            "|------|------------|----------|---------|------------|----------|---------|----------------|",
        ])

        for idx, row in near_duplicate_pairs.head(5).iterrows():
            q1 = str(row["question_x"])[:40] + "..." if len(str(row["question_x"])) > 40 else str(row["question_x"])
            a1 = str(row["answer_x"])[:30] + "..." if len(str(row["answer_x"])) > 30 else str(row["answer_x"])
            q2 = str(row["question_y"])[:40] + "..." if len(str(row["question_y"])) > 40 else str(row["question_y"])
            a2 = str(row["answer_y"])[:30] + "..." if len(str(row["answer_y"])) > 30 else str(row["answer_y"])

            lines.append(f"| {idx+1} | {q1} | {a1} | {row['label_x']} | {q2} | {a2} | {row['label_y']} | {row.get('distance_score', 'N/A'):.3f} |")

        lines.append("")
    else:
        lines.extend([
            "## Near-Duplicate Pairs",
            "",
            "No near-duplicates detected.",
            "",
        ])

    # Add recommendations
    lines.extend([
        "## Recommendations",
        "",
    ])

    if label_issues > 0:
        lines.extend([
            f"1. **HIGH PRIORITY:** Review {label_issues} suspect examples with potential label errors.",
            "   - See `suspect_examples.csv` for full list with confidence scores.",
            "   - Focus on examples with `label_score < 0.5` (lowest confidence).",
            "",
        ])

    if outlier_issues > 0:
        lines.extend([
            f"2. **MEDIUM PRIORITY:** Examine {outlier_issues} outlier examples for edge cases.",
            "   - These may represent rare patterns or data quality issues.",
            "   - See `outlier_examples.csv` for full list.",
            "",
        ])

    if near_duplicate_issues > 0:
        lines.extend([
            f"3. **LOW PRIORITY:** Check {near_duplicate_issues} near-duplicate pairs for label consistency.",
            "   - Similar examples with different labels may indicate annotation inconsistency.",
            "   - See `near_duplicate_pairs.csv` for full list.",
            "",
        ])

    if label_issues == 0 and outlier_issues == 0 and near_duplicate_issues == 0:
        lines.extend([
            "1. **EXCELLENT:** No label quality issues detected.",
            "   - Training data appears to be high quality.",
            "   - Proceed with model training.",
            "",
        ])

    lines.extend([
        "## Next Steps",
        "",
        "1. Review suspect examples and correct labels if needed",
        "2. Update source data with corrected labels",
        "3. Re-run diagnostics to verify improvements",
        "4. Re-train models with cleaned data",
        "",
        f"---",
        f"",
        f"*Generated by Cleanlab label diagnostics*",
        f"*Git SHA: {_git_sha()}*",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    """Main entry point."""
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load prepared data
    frame = pd.read_parquet(args.input)

    # Map labels to integers (same mapping as Phase 5)
    label_mapping = {"non_evasive": 0, "evasive": 1}
    if frame["label"].dtype == "object":
        # Only map if labels are strings
        frame["label_int"] = frame["label"].map(label_mapping)
    else:
        # Labels already encoded
        frame["label_int"] = frame["label"]

    # Run diagnostics
    print("Running Cleanlab label quality diagnostics...")
    datalab, issue_summary = run_label_diagnostics(frame, random_state=args.random_state)

    # Get training data info (issue_summary only contains training data)
    train_size = len(issue_summary)

    # Extract suspect examples (label issues)
    if "is_label_issue" in issue_summary.columns:
        suspect_mask = issue_summary["is_label_issue"]
        suspect_examples = frame.loc[issue_summary[suspect_mask].index].copy()
        suspect_examples["label_score"] = issue_summary.loc[suspect_mask, "label_score"]
        suspect_examples["is_label_issue"] = True
        suspect_examples["is_outlier_issue"] = (
            issue_summary.loc[suspect_mask, "is_outlier_issue"] if "is_outlier_issue" in issue_summary.columns else False
        )
        suspect_examples["is_near_duplicate_issue"] = (
            issue_summary.loc[suspect_mask, "is_near_duplicate_issue"]
            if "is_near_duplicate_issue" in issue_summary.columns
            else False
        )
        suspect_examples = suspect_examples.sort_values("label_score")
    else:
        suspect_examples = pd.DataFrame()

    # Extract outlier examples
    if "is_outlier_issue" in issue_summary.columns:
        outlier_mask = issue_summary["is_outlier_issue"]
        outlier_examples = frame.loc[issue_summary[outlier_mask].index].copy()
        outlier_examples["outlier_score"] = issue_summary.loc[outlier_mask, "outlier_score"]
        outlier_examples = outlier_examples.sort_values("outlier_score", ascending=False)
    else:
        outlier_examples = pd.DataFrame()

    # Extract near-duplicate pairs
    near_duplicate_pairs = pd.DataFrame()
    if "is_near_duplicate_issue" in issue_summary.columns:
        duplicate_mask = issue_summary["is_near_duplicate_issue"]
        if duplicate_mask.sum() > 0:
            # Get examples with near-duplicate issues
            duplicate_examples = frame.loc[issue_summary[duplicate_mask].index].copy()
            # Pair up similar examples (simplified approach)
            near_duplicate_pairs = duplicate_examples.head(100).reset_index()  # Limit to 100 for safety

    # Save outputs
    suspect_path = output_root / "suspect_examples.csv"
    suspect_examples.to_csv(suspect_path, index=False)
    print(f"Saved suspect examples: {suspect_path}")

    outlier_path = output_root / "outlier_examples.csv"
    outlier_examples.to_csv(outlier_path, index=False)
    print(f"Saved outlier examples: {outlier_path}")

    if not near_duplicate_pairs.empty:
        dup_path = output_root / "near_duplicate_pairs.csv"
        near_duplicate_pairs.to_csv(dup_path, index=False)
        print(f"Saved near-duplicate pairs: {dup_path}")

    # Create summary JSON
    summary = {
        "training_size": train_size,
        "label_issues": int(issue_summary["is_label_issue"].sum()) if "is_label_issue" in issue_summary.columns else 0,
        "outlier_issues": (
            int(issue_summary["is_outlier_issue"].sum()) if "is_outlier_issue" in issue_summary.columns else 0
        ),
        "near_duplicate_issues": (
            int(issue_summary["is_near_duplicate_issue"].sum())
            if "is_near_duplicate_issue" in issue_summary.columns
            else 0
        ),
        "quality_score": (
            (1 - issue_summary["is_label_issue"].sum() / train_size) * 100
            if "is_label_issue" in issue_summary.columns and train_size > 0
            else 100.0
        ),
        "random_state": args.random_state,
        "git_sha": _git_sha(),
    }

    summary_path = output_root / "label_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved summary: {summary_path}")

    # Create markdown report
    report_path = _create_markdown_report(
        output_root, issue_summary, suspect_examples, outlier_examples, near_duplicate_pairs, train_size
    )
    print(f"Saved report: {report_path}")

    # Log to MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="phase6-label-diagnostics"):
        mlflow.log_params(
            {
                "random_state": args.random_state,
                "training_size": train_size,
                "git_sha": _git_sha(),
            }
        )
        mlflow.log_metrics(
            {
                "label_issues": summary["label_issues"],
                "outlier_issues": summary["outlier_issues"],
                "near_duplicate_issues": summary["near_duplicate_issues"],
                "quality_score": summary["quality_score"],
            }
        )
        mlflow.set_tags(
            {
                "pipeline_stage": "phase-06-label-diagnostics",
                "git_sha": _git_sha(),
            }
        )
        mlflow.log_artifact(str(suspect_path))
        mlflow.log_artifact(str(outlier_path))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(report_path))
        if not near_duplicate_pairs.empty:
            mlflow.log_artifact(str(dup_path))

    print("\n" + "=" * 60)
    print("Label Diagnostics Complete")
    print("=" * 60)
    print(f"Training examples analyzed: {train_size}")
    print(f"Label issues found: {summary['label_issues']}")
    print(f"Outlier issues found: {summary['outlier_issues']}")
    print(f"Near-duplicate issues found: {summary['near_duplicate_issues']}")
    print(f"Overall quality score: {summary['quality_score']:.1f}%")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
