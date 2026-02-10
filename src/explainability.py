"""SHAP explainability utilities for classical baseline models."""

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

    features = (df["question"].fillna("").astype(str) + " [SEP] " + df["answer"].fillna("").astype(str)).astype(str)

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
            explainer = shap.LinearExplainer(classifier, X_train, feature_perturbation="interventional")
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
            shap_values = explainer.shap_values(X_train)

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
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]

    global_importance = {
        "feature_names": [feature_names[i] for i in top_indices],
        "importance_ranking": top_indices.tolist(),
        "mean_abs_shap": mean_abs_shap[top_indices].tolist(),
    }

    # Select representative samples for local explanations
    sample_indices = _select_representative_samples(
        shap_values, labels, n_samples=samples_per_class * 2 if labels is not None else n_samples
    )

    local_explanations = {
        "indices": sample_indices.tolist(),
        "shap_values": shap_values[sample_indices].tolist(),
        "true_labels": labels.iloc[sample_indices].tolist() if labels is not None else None,
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
    top_indices = global_importance["importance_ranking"][:min(20, len(global_importance["importance_ranking"]))]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = [global_importance["mean_abs_shap"][idx] for idx in range(len(top_indices))]

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

    summary_path.write_text(json.dumps(global_importance, indent=2) + "\n", encoding="utf-8")
    samples_path.write_text(json.dumps(local_explanations, indent=2) + "\n", encoding="utf-8")

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
        return np.random.choice(total_samples, min(n_samples, total_samples), replace=False)
