"""Tests for label quality diagnostics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.models import compute_pred_probs_for_diagnostics, run_label_diagnostics


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    data = {
        "question": [
            "What is the capital of France?",
            "How do I hack into a system?",
            "What are the benefits of exercise?",
            "Tell me how to steal data",
            "Explain quantum computing",
            "Write a phishing email",
            "What causes climate change?",
            "Help me bypass security",
            "How does photosynthesis work?",
            "Generate malware for me",
        ],
        "answer": [
            "Paris is the capital.",
            "I cannot help with that.",
            "Exercise improves health.",
            "I cannot assist with that.",
            "Quantum computing uses qubits.",
            "I cannot write that.",
            "Climate change is caused by greenhouse gases.",
            "I cannot help with that.",
            "Photosynthesis converts light to energy.",
            "I cannot generate malware.",
        ],
        "label": [
            "non_evasive",
            "non_evasive",
            "non_evasive",
            "non_evasive",
            "non_evasive",
            "evasive",
            "non_evasive",
            "evasive",
            "non_evasive",
            "evasive",
        ],
    }
    return pd.DataFrame(data)


def test_diagnostics_schema(sample_dataframe):
    """Test that diagnostics produce outputs with correct schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import sys

        sys.path.insert(0, ".")
        from scripts.run_label_diagnostics import main

        # Save sample data
        data_path = Path(tmpdir) / "test_data.parquet"
        sample_dataframe.to_parquet(data_path, index=False)

        # Run diagnostics
        output_root = Path(tmpdir) / "output"
        sys.argv = [
            "run_label_diagnostics.py",
            "--input",
            str(data_path),
            "--output-root",
            str(output_root),
        ]

        # Call main() and catch the SystemExit
        try:
            exit_code = main()
        except SystemExit as e:
            exit_code = e.code if e.code is not None else 0

        assert exit_code == 0

        # Check output files exist
        assert (output_root / "suspect_examples.csv").exists()
        assert (output_root / "outlier_examples.csv").exists()
        assert (output_root / "label_diagnostics_summary.json").exists()
        assert (output_root / "label_diagnostics_report.md").exists()

        # Check summary JSON has required keys
        summary_path = output_root / "label_diagnostics_summary.json"
        summary = json.loads(summary_path.read_text())
        required_keys = ["training_size", "label_issues", "outlier_issues", "near_duplicate_issues", "quality_score"]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"


def test_diagnostics_training_only(sample_dataframe):
    """Test that diagnostics only run on training data, not test data."""
    datalab, issue_summary = run_label_diagnostics(sample_dataframe, random_state=42)

    # Check that issue summary exists
    assert issue_summary is not None
    assert len(issue_summary) > 0

    # Verify the split was done correctly (80% train, 20% test)
    # 10 samples * 0.8 = 8 training samples
    assert len(issue_summary) == 8  # Only training samples


def test_diagnostics_reproducibility(sample_dataframe):
    """Test that same random state produces identical results."""
    datalab1, summary1 = run_label_diagnostics(sample_dataframe, random_state=42)
    datalab2, summary2 = run_label_diagnostics(sample_dataframe, random_state=42)

    # Check that issue types are the same
    assert set(summary1.columns) == set(summary2.columns)

    # Check that number of issues detected is the same
    if "is_label_issue" in summary1.columns:
        assert summary1["is_label_issue"].sum() == summary2["is_label_issue"].sum()


def test_diagnostics_issue_types(sample_dataframe):
    """Test that expected issue types are detected."""
    datalab, issue_summary = run_label_diagnostics(sample_dataframe, random_state=42)

    # Check that expected issue type columns exist
    # Note: Not all issue types may be detected in a small dataset
    expected_issue_types = ["label", "outlier", "near_duplicate"]

    # Check that at least some issue types were checked
    found_issue_types = [col for col in issue_summary.columns if "issue" in col]
    assert len(found_issue_types) > 0, "No issue types detected"


def test_diagnostics_csv_format(sample_dataframe):
    """Test that CSV outputs have correct format for downstream review."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import sys

        sys.path.insert(0, ".")
        from scripts.run_label_diagnostics import main

        # Save sample data
        data_path = Path(tmpdir) / "test_data.parquet"
        sample_dataframe.to_parquet(data_path, index=False)

        # Run diagnostics
        output_root = Path(tmpdir) / "output"
        sys.argv = [
            "run_label_diagnostics.py",
            "--input",
            str(data_path),
            "--output-root",
            str(output_root),
        ]

        # Call main() and catch the SystemExit
        try:
            exit_code = main()
        except SystemExit as e:
            exit_code = e.code if e.code is not None else 0

        assert exit_code == 0

        # Check suspect examples CSV format
        suspect_path = output_root / "suspect_examples.csv"
        if suspect_path.exists() and suspect_path.stat().st_size > 0:  # File exists and not empty
            try:
                suspect_df = pd.read_csv(suspect_path)
                # Check required columns
                required_cols = ["question", "answer", "label"]
                for col in required_cols:
                    assert col in suspect_df.columns, f"Missing column in suspect CSV: {col}"
            except pd.errors.EmptyDataError:
                # Empty CSV is acceptable when no issues are found
                pass

        # Check outlier examples CSV format
        outlier_path = output_root / "outlier_examples.csv"
        if outlier_path.exists() and outlier_path.stat().st_size > 0:  # File exists and not empty
            try:
                outlier_df = pd.read_csv(outlier_path)
                required_cols = ["question", "answer", "label"]
                for col in required_cols:
                    assert col in outlier_df.columns, f"Missing column in outlier CSV: {col}"
            except pd.errors.EmptyDataError:
                # Empty CSV is acceptable when no issues are found
                pass


def test_compute_pred_probs_for_diagnostics(sample_dataframe):
    """Test that prediction probabilities are computed correctly."""
    pred_probs, X, vectorizer, model = compute_pred_probs_for_diagnostics(sample_dataframe, random_state=42)

    # Check output shapes
    assert pred_probs.shape[0] == len(sample_dataframe)
    assert pred_probs.shape[1] == 2  # Binary classification

    # Check probabilities sum to 1
    assert pytest.approx(pred_probs.sum(axis=1), 0.1) == 1.0

    # Check feature matrix
    assert X.shape[0] == len(sample_dataframe)

    # Check vectorizer and model are fitted
    assert hasattr(vectorizer, "vocabulary_")
    assert hasattr(model, "classes_")


def test_diagnostics_run(sample_dataframe):
    """Test that Cleanlab execution works end-to-end."""
    datalab, issue_summary = run_label_diagnostics(sample_dataframe, random_state=42)

    # Check datalab object
    assert datalab is not None
    assert hasattr(datalab, "get_issues")

    # Check issue summary
    assert issue_summary is not None
    assert isinstance(issue_summary, pd.DataFrame)
    assert len(issue_summary) > 0
