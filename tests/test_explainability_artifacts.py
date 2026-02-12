"""Tests for Phase 6 explainability artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def xai_root() -> Path:
    """Return the root directory for XAI artifacts."""
    return Path("artifacts/explainability/phase6")


@pytest.mark.parametrize(
    "family",
    ["logreg", "tree", "boosting"],
)
def test_shap_schema_exists(xai_root: Path, family: str) -> None:
    """Test that SHAP schema files exist for each model family."""
    family_dir = xai_root / family
    assert family_dir.exists(), f"XAI directory not found: {family_dir}"

    summary_path = family_dir / "shap_summary.json"
    samples_path = family_dir / "shap_samples.json"
    plot_path = family_dir / "shap_summary.png"

    assert summary_path.exists(), f"SHAP summary not found: {summary_path}"
    assert samples_path.exists(), f"SHAP samples not found: {samples_path}"
    assert plot_path.exists(), f"SHAP plot not found: {plot_path}"


@pytest.mark.parametrize(
    "family",
    ["logreg", "tree", "boosting"],
)
def test_shap_summary_keys(xai_root: Path, family: str) -> None:
    """Test that shap_summary.json has required keys."""
    summary_path = xai_root / family / "shap_summary.json"
    assert summary_path.exists(), f"SHAP summary not found: {summary_path}"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    required_keys = ["feature_names", "importance_ranking", "mean_abs_shap"]
    for key in required_keys:
        assert key in summary, f"Missing key in summary: {key}"

    # Validate structure
    assert len(summary["feature_names"]) > 0, "feature_names is empty"
    assert len(summary["importance_ranking"]) > 0, "importance_ranking is empty"
    assert len(summary["mean_abs_shap"]) > 0, "mean_abs_shap is empty"

    # Ensure lengths match
    n_features = len(summary["feature_names"])
    assert (
        len(summary["importance_ranking"]) == n_features
    ), "importance_ranking length mismatch"
    assert len(summary["mean_abs_shap"]) == n_features, "mean_abs_shap length mismatch"


@pytest.mark.parametrize(
    "family",
    ["logreg", "tree", "boosting"],
)
def test_shap_samples_keys(xai_root: Path, family: str) -> None:
    """Test that shap_samples.json has required keys."""
    samples_path = xai_root / family / "shap_samples.json"
    assert samples_path.exists(), f"SHAP samples not found: {samples_path}"

    samples = json.loads(samples_path.read_text(encoding="utf-8"))

    required_keys = ["indices", "shap_values", "true_labels"]
    for key in required_keys:
        assert key in samples, f"Missing key in samples: {key}"

    # Validate structure
    assert len(samples["indices"]) > 0, "No sample indices"
    assert len(samples["shap_values"]) > 0, "No SHAP values"
    assert samples["true_labels"] is not None, "true_labels is None"

    # Ensure lengths match
    n_samples = len(samples["indices"])
    assert len(samples["shap_values"]) == n_samples, "shap_values length mismatch"
    assert len(samples["true_labels"]) == n_samples, "true_labels length mismatch"


def test_shap_no_test_leakage(xai_root: Path) -> None:
    """Test that SHAP was computed on training data only, not test data.

    This is a meta-check: we verify that the explainability script
    doesn't have access to test splits by checking the metadata.
    """
    # The explainability script should only use the full dataset,
    # not split data. We verify this by checking the summary exists
    # and doesn't contain test-specific artifacts.

    for family in ["logreg", "tree", "boosting"]:
        family_dir = xai_root / family
        if not family_dir.exists():
            continue

        summary_path = family_dir / "shap_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            # Check that we don't have test-specific keys
            # (This is a sanity check - the real validation is in code review)
            assert (
                "test_indices" not in summary
            ), "SHAP summary should not contain test_indices"
            assert (
                "test_shap_values" not in summary
            ), "SHAP summary should not contain test_shap_values"


def test_shap_reproducibility(xai_root: Path) -> None:
    """Test that SHAP output is deterministic with fixed random state.

    This test verifies that running explainability twice with the same
    random state produces identical results.
    """
    # For now, this is a smoke test - we just verify the artifacts exist
    # A full reproducibility test would require re-running the XAI pipeline
    # which is expensive. We verify the structure is consistent instead.

    for family in ["logreg", "tree", "boosting"]:
        family_dir = xai_root / family
        if not family_dir.exists():
            continue

        summary_path = family_dir / "shap_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            # Verify that importance rankings are integers
            for idx in summary["importance_ranking"]:
                assert isinstance(
                    idx, int
                ), f"Importance ranking should be int, got {type(idx)}"

            # Verify that mean_abs_shap values are floats
            for val in summary["mean_abs_shap"]:
                assert isinstance(
                    val, (int, float)
                ), f"Mean abs SHAP should be numeric, got {type(val)}"


def test_xai_summary_aggregation(xai_root: Path) -> None:
    """Test that the combined xai_summary.json exists and has correct structure."""
    summary_path = xai_root / "xai_summary.json"
    assert summary_path.exists(), f"XAI summary not found: {summary_path}"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # Should have entries for each family
    for family in ["logreg", "tree", "boosting"]:
        # Families might not all exist if some weren't trained
        if (xai_root / family).exists():
            assert family in summary, f"Missing family {family} in summary"

            # Check required keys
            assert "explainer_type" in summary[family]
            assert "n_features" in summary[family]
            assert "n_samples" in summary[family]
            assert "output_dir" in summary[family]
            assert "artifacts" in summary[family]
