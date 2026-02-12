"""Explainability utilities for classical and transformer models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def explain_classical_model(
    model_dir: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    *,
    random_state: int = 42,
    n_samples: int = 100,
    top_k: int = 20,
    samples_per_class: int = 5,
) -> dict[str, Any]:
    """Generate SHAP explanations for a classical model.

    Args:
        model_dir: Directory containing the trained model artifacts
        data_path: Path to prepared dataset parquet file
        output_dir: Directory to write SHAP artifacts
        random_state: Random seed for reproducibility
        n_samples: Number of background samples for SHAP (uses training data only)
        top_k: Number of top features to include in global importance
        samples_per_class: Number of representative samples per class for local explanations

    Returns:
        Dictionary with feature_names, shap_values array, and sample indices

    Raises:
        ValueError: If model directory doesn't exist or model family is unsupported
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model metadata to determine model family
    metadata_path = model_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Model metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_family = metadata.get("model_family")

    if model_family not in ("logreg", "tree", "boosting"):
        raise ValueError(f"Unsupported model family: {model_family}")

    # Load the trained model and associated components
    model_data, vectorizer = _load_model(model_dir, model_family)

    # Handle different return types from _load_model
    if model_family == "logreg":
        model = model_data  # model_data is the Pipeline
    else:
        # For tree/boosting, model_data is a dict with 'model', 'vectorizer', 'svd'
        model_bundle = model_data
        model = model_bundle["model"]
        vectorizer = model_bundle["vectorizer"]

    # Load training data (IMPORTANT: use training data only, NOT test data)
    df = pd.read_parquet(data_path)

    # Reconstruct features using the same logic as training
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("Dataset must contain 'question' and 'answer' columns")

    features = (
        df["question"].fillna("").astype(str)
        + " [SEP] "
        + df["answer"].fillna("").astype(str)
    ).astype(str)

    # Get label column if available
    target_col = "label" if "label" in df.columns else None
    labels = df[target_col] if target_col else None

    # For Pipeline models (logreg), extract the fitted components
    if isinstance(model, Pipeline):
        classifier = model.named_steps["clf"]

        # Transform features to get the training matrix
        X_train = model.named_steps["tfidf"].transform(features)

        # Select appropriate explainer for the model type
        if isinstance(classifier, LogisticRegression):
            explainer = shap.LinearExplainer(
                classifier, X_train, feature_perturbation="interventional"
            )
        else:
            raise ValueError(f"Unsupported classifier in Pipeline: {type(classifier)}")

        shap_values = explainer.shap_values(X_train)
        feature_names = model.named_steps["tfidf"].get_feature_names_out().tolist()

    # For tree-based models
    elif isinstance(model, (RandomForestClassifier, HistGradientBoostingClassifier)):
        # Use the loaded vectorizer from the model bundle
        X_train = vectorizer.transform(features)

        # For boosting models with SVD, apply dimensionality reduction
        if model_family == "boosting":
            svd = model_bundle.get("svd")
            if svd is None:
                raise ValueError("Boosting model bundle missing SVD component")

            X_train_reduced = svd.transform(X_train)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train_reduced)

            # For SVD components, feature names are the component indices
            feature_names = [f"component_{i}" for i in range(X_train_reduced.shape[1])]

            # Note: shap_values for binary classification returns list of two arrays
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
        else:
            # RandomForest
            explainer = shap.TreeExplainer(model)
            # Convert sparse matrix to dense for TreeExplainer
            X_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
            shap_values = explainer.shap_values(X_dense)

            # TreeExplainer returns list for binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class

            feature_names = vectorizer.get_feature_names_out().tolist()

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    # Ensure shap_values is 2D array
    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(-1, 1)

    # Compute global feature importance (mean absolute SHAP)
    # Handle different shap_values structures
    if isinstance(shap_values, list):
        # Multi-class: use positive class
        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_vals = shap_values

    # Ensure shap_vals is 2D
    if len(shap_vals.shape) == 1:
        shap_vals = shap_vals.reshape(-1, 1)

    # Compute mean absolute SHAP across samples
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    # Handle multi-output case (e.g., multi-class SHAP)
    if len(mean_abs_shap.shape) > 1:
        # Take mean across classes
        mean_abs_shap = np.mean(mean_abs_shap, axis=1)

    # Ensure top_k doesn't exceed number of features
    actual_top_k = min(top_k, len(mean_abs_shap))
    # Get top k indices in descending order
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    top_indices = sorted_indices[:actual_top_k]

    # Convert to list of Python ints for JSON serialization
    top_indices_list = [int(i) for i in top_indices.flatten()]

    global_importance = {
        "feature_names": [feature_names[i] for i in top_indices_list],
        "importance_ranking": top_indices_list,
        "mean_abs_shap": mean_abs_shap[top_indices_list].tolist(),
    }

    # Select representative samples for local explanations
    sample_indices = _select_representative_samples(
        shap_values,
        labels,
        n_samples=samples_per_class * 2 if labels is not None else n_samples,
    )

    local_explanations = {
        "indices": sample_indices.tolist(),
        "shap_values": shap_values[sample_indices].tolist(),
        "true_labels": labels.iloc[sample_indices].tolist()
        if labels is not None
        else None,
    }

    # Generate summary plot
    plt.figure(figsize=(10, 6))

    # Create SHAP Features object for plotting
    if hasattr(X_train, "toarray"):
        X_plot = X_train.toarray()
    else:
        X_plot = X_train

    # Create a simplified summary plot using matplotlib directly
    # shap.summary_plot can be problematic in headless environments
    plt.figure(figsize=(10, 6))

    # Select top features for visualization
    top_indices = global_importance["importance_ranking"][
        : min(20, len(global_importance["importance_ranking"]))
    ]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = global_importance["mean_abs_shap"]

    # Handle multi-class importance (take mean across classes if needed)
    if isinstance(top_importance[0], list):
        top_importance = [np.mean(vals) for vals in top_importance]

    # Create horizontal bar chart
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_importance, align="center")
    plt.yticks(y_pos, top_features)
    plt.xlabel("Mean |SHAP Value|")
    plt.title(f"Global Feature Importance - {model_family.upper()} Model")
    plt.tight_layout()

    summary_plot_path = output_dir / "shap_summary.png"
    plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Write artifacts
    summary_path = output_dir / "shap_summary.json"
    samples_path = output_dir / "shap_samples.json"

    summary_path.write_text(
        json.dumps(global_importance, indent=2) + "\n", encoding="utf-8"
    )
    samples_path.write_text(
        json.dumps(local_explanations, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "feature_names": feature_names,
        "shap_values": shap_values,
        "sample_indices": sample_indices,
        "explainer_type": type(explainer).__name__,
    }


def _load_model(model_dir: Path, model_family: str) -> tuple[Any, Any]:
    """Load a trained model from the artifact directory.

    Args:
        model_dir: Directory containing model artifacts
        model_family: Model family (logreg, tree, or boosting)

    Returns:
        Tuple of (model, vectorizer) or (model_bundle_dict, vectorizer)

    Raises:
        ValueError: If model file doesn't exist or can't be loaded
    """
    import pickle

    if model_family == "logreg":
        model_file = model_dir / "model.pkl"
        if not model_file.exists():
            raise ValueError(f"Model file not found: {model_file}")

        with open(model_file, "rb") as f:
            model = pickle.load(f)

        return model, None  # Pipeline contains vectorizer

    elif model_family in ("tree", "boosting"):
        bundle_file = model_dir / "model_bundle.pkl"
        if not bundle_file.exists():
            raise ValueError(f"Model bundle file not found: {bundle_file}")

        with open(bundle_file, "rb") as f:
            bundle = pickle.load(f)

        return bundle, bundle["vectorizer"]

    else:
        raise ValueError(f"Unsupported model family: {model_family}")


def _select_representative_samples(
    shap_values: np.ndarray,
    labels: pd.Series | None,
    n_samples: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """Select representative samples for local explanations.

    Args:
        shap_values: SHAP values array (n_samples, n_features)
        labels: True labels (optional)
        n_samples: Number of samples to select
        random_state: Random seed

    Returns:
        Array of sample indices
    """
    np.random.seed(random_state)
    total_samples = len(shap_values)

    if labels is not None:
        # Select samples balanced across classes
        unique_labels = labels.unique()
        samples_per_label = n_samples // len(unique_labels)

        selected_indices = []
        for label in unique_labels:
            label_indices = labels[labels == label].index
            n_select = min(samples_per_label, len(label_indices))
            selected = np.random.choice(label_indices, n_select, replace=False)
            selected_indices.extend(selected)

        return np.array(selected_indices[:n_samples])
    else:
        # Random selection if no labels
        return np.random.choice(
            total_samples, min(n_samples, total_samples), replace=False
        )


# =============================================================================
# Transformer Explainability with Captum
# =============================================================================


def explain_transformer_instance(
    model: Any,
    tokenizer: Any,
    text: str,
    true_label: str,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Generate Captum attributions for a single transformer prediction.

    Args:
        model: Trained transformer model (must be in eval mode)
        tokenizer: Tokenizer matching the model
        text: Input text to explain
        true_label: Ground truth label
        device: Device to run attribution on

    Returns:
        Dictionary with:
            - tokens: list of token strings (special tokens filtered)
            - attributions: numpy array of attribution scores
            - predicted_label: string (evasive/non_evasive)
            - attribution_sum: sum of attributions (sanity check)
            - true_label: ground truth label

    Raises:
        ValueError: If model doesn't have embedding layer or prediction fails
    """
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        raise ImportError(
            "Captum is required for transformer explainability. "
            "Install with: pip install captum"
        )

    # Ensure model is in eval mode
    model.eval()
    model = model.to(device)

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class_id = logits.argmax(dim=-1).item()

    # Get label mapping from model config
    if hasattr(model, "config") and hasattr(model.config, "id2label"):
        id2label = model.config.id2label
        predicted_label = id2label.get(
            predicted_class_id, f"class_{predicted_class_id}"
        )
    else:
        predicted_label = f"class_{predicted_class_id}"

    # Find embedding layer for DistilBERT/BERT-like models
    if hasattr(model, "distilbert"):
        target_layer = model.distilbert.embeddings
    elif hasattr(model, "bert"):
        target_layer = model.bert.embeddings
    elif hasattr(model, "roberta"):
        target_layer = model.roberta.embeddings
    else:
        raise ValueError(
            "Unsupported model architecture. "
            "Expected DistilBERT, BERT, or RoBERTa-like model with embedding layer."
        )

    # Create LayerIntegratedGradients explainer
    def forward_func(input_ids, attention_mask=None):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    lig = LayerIntegratedGradients(forward_func, target_layer)

    # Generate attributions for the predicted class
    with torch.no_grad():
        # Use zero tensor as baseline for IG
        baseline = torch.zeros_like(input_ids)

    # Compute attributions
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline,
        additional_forward_args=(attention_mask,),
        target=predicted_class_id,
        return_convergence_delta=False,
    )

    # Convert to numpy and squeeze batch dimension
    attributions_np = attributions.squeeze().cpu().detach().numpy()

    # Get tokens (remove special tokens for interpretability)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().tolist())

    # Filter out special tokens ([CLS], [SEP], [PAD])
    special_tokens = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
    filtered_tokens = []
    filtered_attributions = []

    for token, attr in zip(tokens, attributions_np):
        if token not in special_tokens:
            filtered_tokens.append(token)
            # Handle both scalar and array attributions
            attr_value = float(attr) if np.isscalar(attr) else float(np.sum(attr))
            filtered_attributions.append(attr_value)

    # Sanity check: attribution sum should be close to logit difference
    attribution_sum = float(np.sum(filtered_attributions))

    return {
        "tokens": filtered_tokens,
        "attributions": np.array(filtered_attributions),
        "predicted_label": predicted_label,
        "attribution_sum": attribution_sum,
        "true_label": true_label,
    }


def explain_transformer_batch(
    model: Any,
    tokenizer: Any,
    data: pd.DataFrame,
    output_dir: str | Path,
    *,
    text_col: str = "text",
    label_col: str = "label",
    n_samples: int = 20,
    random_state: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Generate Captum attributions for representative samples.

    Args:
        model: Trained transformer model
        tokenizer: Tokenizer matching the model
        data: DataFrame with text and label columns
        output_dir: Directory to write XAI artifacts
        text_col: Column name containing text
        label_col: Column name containing labels
        n_samples: Number of samples to explain
        random_state: Random seed for sample selection
        device: Device to run attribution on

    Returns:
        Dictionary with sample explanations and metadata

    Raises:
        ValueError: If required columns missing or model prediction fails
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate input
    if text_col not in data.columns:
        raise ValueError(f"Text column '{text_col}' not found in data")
    if label_col not in data.columns:
        raise ValueError(f"Label column '{label_col}' not found in data")

    # Get model predictions for all samples
    model.eval()
    model = model.to(device)

    texts = data[text_col].tolist()
    true_labels = data[label_col].tolist()

    # Predict on all samples to identify correct/incorrect predictions
    predicted_labels = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_class_id = outputs.logits.argmax(dim=-1).item()

        if hasattr(model, "config") and hasattr(model.config, "id2label"):
            id2label = model.config.id2label
            predicted_label = id2label.get(
                predicted_class_id, f"class_{predicted_class_id}"
            )
        else:
            predicted_label = f"class_{predicted_class_id}"

        predicted_labels.append(predicted_label)

    # Select representative samples
    np.random.seed(random_state)

    # Group by (true_label, correct_prediction)
    data_with_preds = data.copy()
    data_with_preds["predicted_label"] = predicted_labels
    data_with_preds["correct"] = (
        data_with_preds[label_col] == data_with_preds["predicted_label"]
    )

    selected_indices = []

    # Try to get balanced samples across groups
    for true_label in data_with_preds[label_col].unique():
        for correct in [True, False]:
            group = data_with_preds[
                (data_with_preds[label_col] == true_label)
                & (data_with_preds["correct"] == correct)
            ]

            if len(group) > 0:
                n_from_group = max(
                    1, n_samples // (len(data_with_preds[label_col].unique()) * 2)
                )
                available = min(n_from_group, len(group))
                selected = np.random.choice(
                    group.index.tolist(), available, replace=False
                )
                selected_indices.extend(selected)

    # If we don't have enough samples, fill randomly
    if len(selected_indices) < n_samples:
        remaining = n_samples - len(selected_indices)
        remaining_indices = [
            i for i in data_with_preds.index.tolist() if i not in selected_indices
        ]
        if remaining_indices:
            additional = np.random.choice(
                remaining_indices, min(remaining, len(remaining_indices)), replace=False
            )
            selected_indices.extend(additional.tolist())

    # Truncate to n_samples
    selected_indices = selected_indices[:n_samples]

    # Generate attributions for selected samples
    explanations = []
    top_tokens_all = []

    for idx in selected_indices:
        text = data.loc[idx, text_col]
        true_label = data.loc[idx, label_col]

        result = explain_transformer_instance(
            model=model,
            tokenizer=tokenizer,
            text=text,
            true_label=true_label,
            device=device,
        )

        result["sample_id"] = int(idx)
        explanations.append(result)

        # Track top tokens for summary
        if len(result["tokens"]) > 0:
            top_token_idx = int(np.argmax(np.abs(result["attributions"])))
            top_tokens_all.append(
                {
                    "token": result["tokens"][top_token_idx],
                    "attribution": float(result["attributions"][top_token_idx]),
                    "sample_id": int(idx),
                }
            )

    # Compute summary statistics
    attribution_sums = [exp["attribution_sum"] for exp in explanations]
    avg_attribution_sum = float(np.mean(attribution_sums))

    # Write JSON output
    xai_json = output_dir / "transformer_xai.json"
    serializable_explanations = []
    for exp in explanations:
        serializable_explanations.append(
            {
                "sample_id": exp["sample_id"],
                "text": texts[exp["sample_id"]],
                "true_label": exp["true_label"],
                "predicted_label": exp["predicted_label"],
                "tokens": exp["tokens"],
                "attributions": exp["attributions"].tolist(),
                "attribution_sum": exp["attribution_sum"],
            }
        )

    xai_json.write_text(
        json.dumps(serializable_explanations, indent=2) + "\n", encoding="utf-8"
    )

    # Write summary
    summary = {
        "n_samples": len(explanations),
        "avg_attribution_sum": avg_attribution_sum,
        "top_tokens": top_tokens_all[:20],  # Top 20 tokens across all samples
        "sample_indices": [int(idx) for idx in selected_indices],
    }

    summary_path = output_dir / "transformer_xai_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Generate HTML visualization
    _generate_transformer_html(explanations, output_dir)

    return {
        "explanations": explanations,
        "summary": summary,
        "n_samples": len(explanations),
    }


def _generate_transformer_html(explanations: list[dict], output_dir: Path) -> None:
    """Generate HTML visualization with highlighted text by attribution strength.

    Args:
        explanations: List of explanation dictionaries from explain_transformer_instance
        output_dir: Directory to write HTML file
    """
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>Transformer Explainability - Captum Attributions</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
        "    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }",
        "    h1 { color: #333; }",
        "    .sample { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }",
        "    .sample-header { font-weight: bold; margin-bottom: 10px; }",
        "    .label-correct { color: green; }",
        "    .label-incorrect { color: red; }",
        "    .tokens { line-height: 1.8; }",
        "    .token { margin-right: 4px; padding: 2px 4px; border-radius: 3px; display: inline-block; }",
        "    .pos { background: linear-gradient(to right, rgba(255, 100, 100, 0.3), rgba(255, 100, 100, 0.8)); }",
        "    .neg { background: linear-gradient(to right, rgba(100, 100, 255, 0.3), rgba(100, 100, 255, 0.8)); }",
        "    .neutral { background: #f0f0f0; }",
        "    .legend { margin: 20px 0; padding: 10px; background: #e8e8e8; border-radius: 5px; }",
        "    .legend-item { display: inline-block; margin-right: 20px; }",
        "    .attribution-info { font-size: 0.9em; color: #666; margin-top: 5px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class='container'>",
        "    <h1>Transformer Explainability Analysis</h1>",
        "    <p>Word-level attributions generated using Captum LayerIntegratedGradients.</p>",
        "",
        "    <div class='legend'>",
        "      <div class='legend-item'>",
        "        <span class='token pos' style='background: rgba(255, 100, 100, 0.5);'>Positive</span>: Pushes toward prediction",
        "      </div>",
        "      <div class='legend-item'>",
        "        <span class='token neg' style='background: rgba(100, 100, 255, 0.5);'>Negative</span>: Pushes away from prediction",
        "      </div>",
        "    </div>",
        "",
    ]

    for exp in explanations:
        predicted = exp["predicted_label"]
        true = exp["true_label"]
        is_correct = predicted == true

        # Generate HTML for highlighted tokens
        tokens_html = []
        for token, attr in zip(exp["tokens"], exp["attributions"]):
            # Normalize attribution for color intensity
            max_attr = max([abs(a) for e in explanations for a in e["attributions"]])
            intensity = min(1.0, abs(attr) / (max_attr + 1e-9))

            if attr > 0:
                css_class = "pos"
                style = f"background: rgba(255, 100, 100, {0.2 + intensity * 0.6});"
            elif attr < 0:
                css_class = "neg"
                style = f"background: rgba(100, 100, 255, {0.2 + intensity * 0.6});"
            else:
                css_class = "neutral"
                style = ""

            # Handle special tokens that should stay together
            display_token = (
                token.replace("##", "") if token.startswith("##") else f" {token}"
            )

            tokens_html.append(
                f"<span class='token {css_class}' style='{style}' title='Attribution: {attr:.4f}'>{display_token}</span>"
            )

        html_parts.extend(
            [
                "    <div class='sample'>",
                f"      <div class='sample-header'>",
                f"        Sample {exp['sample_id']} | ",
                f"        True: <span class='{'label-correct' if is_correct else 'label-incorrect'}'>{true}</span> | ",
                f"        Predicted: {predicted}",
                f"      </div>",
                f"      <div class='tokens'>{''.join(tokens_html)}</div>",
                f"      <div class='attribution-info'>Attribution sum: {exp['attribution_sum']:.4f}</div>",
                "    </div>",
            ]
        )

    html_parts.extend(
        [
            "  </div>",
            "</body>",
            "</html>",
        ]
    )

    html_path = output_dir / "transformer_xai.html"
    html_path.write_text("\n".join(html_parts), encoding="utf-8")
