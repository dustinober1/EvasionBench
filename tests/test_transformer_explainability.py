"""Tests for transformer explainability with Captum."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer

from src.explainability import explain_transformer_batch, explain_transformer_instance


def test_captum_single_sample() -> None:
    """Test explain_transformer_instance returns tokens and attributions."""
    # Create a simple model for testing
    from transformers import AutoModelForSequenceClassification

    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test with sample text
    text = "This is a test question about programming."
    true_label = "evasive"

    result = explain_transformer_instance(
        model=model,
        tokenizer=tokenizer,
        text=text,
        true_label=true_label,
        device="cpu",
    )

    # Verify output structure
    assert "tokens" in result
    assert "attributions" in result
    assert "predicted_label" in result
    assert "attribution_sum" in result
    assert "true_label" in result

    # Verify tokens and attributions align
    assert len(result["tokens"]) == len(result["attributions"])
    assert len(result["tokens"]) > 0

    # Verify no special tokens in output
    special_tokens = {"[CLS]", "[SEP]", "[PAD]"}
    for token in result["tokens"]:
        assert token not in special_tokens

    # Verify attributions are numeric
    assert isinstance(result["attributions"], np.ndarray)
    assert isinstance(result["attribution_sum"], float)

    # Verify true label is preserved
    assert result["true_label"] == true_label


def test_captum_token_alignment() -> None:
    """Test that attribution array length matches token count."""
    from transformers import AutoModelForSequenceClassification

    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = "Another test with different words here."
    result = explain_transformer_instance(
        model=model,
        tokenizer=tokenizer,
        text=text,
        true_label="non_evasive",
        device="cpu",
    )

    # Token-attribution alignment
    assert len(result["tokens"]) == len(result["attributions"])
    assert result["attributions"].dtype == np.float64


def test_captum_batch_output(tmp_path: Path) -> None:
    """Test explain_transformer_batch generates required JSON output."""
    from transformers import AutoModelForSequenceClassification

    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create test data
    test_data = pd.DataFrame(
        {
            "text": [
                "Test question one about programming.",
                "Test question two about security.",
                "Test question three about algorithms.",
            ],
            "label": ["evasive", "non_evasive", "evasive"],
        }
    )

    output_dir = tmp_path / "xai_output"
    result = explain_transformer_batch(
        model=model,
        tokenizer=tokenizer,
        data=test_data,
        output_dir=output_dir,
        n_samples=3,
        device="cpu",
    )

    # Verify JSON outputs exist
    xai_json = output_dir / "transformer_xai.json"
    summary_json = output_dir / "transformer_xai_summary.json"

    assert xai_json.exists(), "transformer_xai.json should exist"
    assert summary_json.exists(), "transformer_xai_summary.json should exist"

    # Verify JSON structure
    explanations = json.loads(xai_json.read_text(encoding="utf-8"))
    assert isinstance(explanations, list)
    assert len(explanations) == 3

    # Check first explanation has required fields
    exp = explanations[0]
    required_keys = {
        "sample_id",
        "text",
        "true_label",
        "predicted_label",
        "tokens",
        "attributions",
        "attribution_sum",
    }
    assert set(exp.keys()) == required_keys

    # Verify summary
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert "n_samples" in summary
    assert "avg_attribution_sum" in summary
    assert summary["n_samples"] == 3


def test_captum_html_output(tmp_path: Path) -> None:
    """Test that HTML visualization is generated."""
    from transformers import AutoModelForSequenceClassification

    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_data = pd.DataFrame(
        {
            "text": [
                "Test question for HTML generation.",
                "Another question for visualization.",
            ],
            "label": ["evasive", "non_evasive"],
        }
    )

    output_dir = tmp_path / "xai_html"
    explain_transformer_batch(
        model=model,
        tokenizer=tokenizer,
        data=test_data,
        output_dir=output_dir,
        n_samples=2,
        device="cpu",
    )

    html_file = output_dir / "transformer_xai.html"
    assert html_file.exists(), "HTML file should be generated"

    # Verify HTML contains expected content
    html_content = html_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_content
    assert "Transformer Explainability" in html_content
    assert "sample" in html_content.lower()
    assert "attribution" in html_content.lower()


def test_captum_reproducibility(tmp_path: Path) -> None:
    """Test that same model instance produces same predictions (deterministic behavior)."""
    from transformers import AutoModelForSequenceClassification, set_seed

    model_name = "distilbert-base-uncased"
    set_seed(42)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = "Reproducibility test question."

    # Run explanation twice on the same model
    result1 = explain_transformer_instance(
        model=model,
        tokenizer=tokenizer,
        text=text,
        true_label="evasive",
        device="cpu",
    )

    result2 = explain_transformer_instance(
        model=model,
        tokenizer=tokenizer,
        text=text,
        true_label="evasive",
        device="cpu",
    )

    # Predictions should be consistent
    assert result1["predicted_label"] == result2["predicted_label"]

    # Attributions should be exactly the same for the same model
    np.testing.assert_array_equal(
        result1["attributions"],
        result2["attributions"],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
