# Explainability Guide: SHAP for Classical Baseline Models

## Overview

Phase 6 generates SHAP (SHapley Additive exPlanations) values for all Phase 5 classical baseline models (logistic regression, random forest, and histogram gradient boosting). SHAP values provide feature importance and local explanations for model predictions.

## What are SHAP Values?

SHAP values measure the contribution of each feature to a specific prediction:

- **Positive SHAP value**: Feature pushes the prediction toward the positive class (evasive behavior)
- **Negative SHAP value**: Feature pushes the prediction toward the negative class (non-evasive behavior)
- **Magnitude**: Larger absolute values indicate stronger influence on the prediction

SHAP values have a solid theoretical foundation in game theory and provide consistent feature importance measurements across different model types.

## Per-Model Explainer Types

### Logistic Regression (LinearExplainer)

- **Model**: TF-IDF + Logistic Regression
- **Explainer**: `shap.LinearExplainer`
- **Interpretation**: SHAP values represent the contribution of each TF-IDF feature to the log-odds of the prediction
- **Feature Space**: Bag-of-words (n-grams) from Q+A text
- **Example**: A high positive SHAP value for "financial" means the presence of this word increases the probability of evasive classification

### Random Forest (TreeExplainer)

- **Model**: TF-IDF + RandomForestClassifier
- **Explainer**: `shap.TreeExplainer`
- **Interpretation**: SHAP values measure how much each feature contributes to the prediction across all trees in the forest
- **Feature Space**: Bag-of-words (n-grams) from Q+A text
- **Note**: TreeExplainer is exact for tree-based models and runs in polynomial time

### Histogram Gradient Boosting (TreeExplainer)

- **Model**: TF-IDF + TruncatedSVD + HistGradientBoostingClassifier
- **Explainer**: `shap.TreeExplainer`
- **Interpretation**: SHAP values are computed on the SVD-reduced feature space (components, not original words)
- **Feature Space**: 80 SVD components (latent features)
- **Caveat**: Feature importance is at the component level, not word level. Components represent combinations of words.

## Artifact Structure

### Per-Family Artifacts

Each model family has three artifacts in `artifacts/explainability/phase6/{family}/`:

#### 1. `shap_summary.json` - Global Feature Importance

```json
{
  "feature_names": ["in", "have", "financial", "can", ...],
  "importance_ranking": [42, 15, 7, 123, ...],
  "mean_abs_shap": [0.0234, 0.0198, 0.0156, ...]
}
```

- **feature_names**: List of feature names (words/n-grams or components)
- **importance_ranking**: Indices of top features sorted by importance
- **mean_abs_shap**: Mean absolute SHAP value for each feature (measures global importance)

**Interpretation**: Higher `mean_abs_shap` = more important feature globally.

#### 2. `shap_samples.json` - Local Explanations

```json
{
  "indices": [42, 157, 893, ...],
  "shap_values": [[0.012, -0.034, 0.008, ...], ...],
  "true_labels": ["direct", "evasive", "intermediate", ...]
}
```

- **indices**: Sample indices from the training set
- **shap_values**: SHAP values for each feature for each sample (matches feature order in summary)
- **true_labels**: Ground truth labels for each sample

**Interpretation**: For a specific sample, positive SHAP values indicate features that pushed toward evasive, negative values toward non-evasive.

#### 3. `shap_summary.png` - Visualization

Horizontal bar chart showing the top 20 features by mean absolute SHAP value. Useful for reports and presentations.

### Combined Artifact

#### 4. `xai_summary.json` - Cross-Family Summary

```json
{
  "logreg": {
    "explainer_type": "LinearExplainer",
    "n_features": 268,
    "n_samples": 10,
    "output_dir": "artifacts/explainability/phase6/logreg",
    "artifacts": ["shap_summary.json", "shap_samples.json", "shap_summary.png"]
  },
  "tree": {...},
  "boosting": {...}
}
```

## Usage Examples

### Load and Analyze Global Feature Importance

```python
import json
import pandas as pd

# Load summary for a specific model family
with open("artifacts/explainability/phase6/logreg/shap_summary.json") as f:
    summary = json.load(f)

# Create DataFrame for analysis
df_importance = pd.DataFrame({
    "feature": summary["feature_names"],
    "importance": summary["mean_abs_shap"]
}).sort_values("importance", ascending=False)

print("Top 10 features:")
print(df_importance.head(10))
```

### Load and Analyze Local Explanations

```python
import json
import numpy as np

# Load local explanations
with open("artifacts/explainability/phase6/logreg/shap_samples.json") as f:
    samples = json.load(f)

# Analyze a specific sample
sample_idx = 0
sample_index = samples["indices"][sample_idx]
shap_vals = samples["shap_values"][sample_idx]
true_label = samples["true_labels"][sample_idx]

print(f"Sample {sample_index} (true label: {true_label})")
print(f"Top 5 contributing features:")

# Get top features for this sample
top_indices = np.argsort(np.abs(shap_vals))[::-1][:5]
for idx in top_indices:
    print(f"  Feature {idx}: SHAP = {shap_vals[idx]:.4f}")
```

### Load Feature Names for Mapping

```python
import json

# Load summary to get feature names
with open("artifacts/explainability/phase6/logreg/shap_summary.json") as f:
    summary = json.load(f)

feature_names = summary["feature_names"]

# Now you can map SHAP values to feature names
# shap_values[i] corresponds to feature_names[i]
```

## Generating Explainability Artifacts

### Using DVC

```bash
dvc repro phase6_xai_classical
```

### Using Make

```bash
make xai-classical
```

### Direct Script

```bash
python scripts/run_explainability_analysis.py \
  --data data/processed/evasionbench_prepared.parquet \
  --models-root artifacts/models/phase5 \
  --output-root artifacts/explainability/phase6 \
  --families all
```

### Select Specific Families

```bash
# Only logistic regression
python scripts/run_explainability_analysis.py \
  --data data/processed/evasionbench_prepared.parquet \
  --models-root artifacts/models/phase5 \
  --output-root artifacts/explainability/phase6 \
  --families logreg

# Only tree-based models
python scripts/run_explainability_analysis.py \
  --data data/processed/evasionbench_prepared.parquet \
  --models-root artifacts/models/phase5 \
  --output-root artifacts/explainability/phase6 \
  --families tree
```

## Important Limitations

### 1. Training Data Only

SHAP values are computed on the **training data only**, not test data. This prevents data leakage and ensures explanations are based on the model's learned patterns, not test set characteristics.

**Implication**: SHAP values reflect how features influence predictions on the training distribution. Test set predictions may have different SHAP distributions.

### 2. Not Post-Hoc on Test Predictions

These artifacts are **not** SHAP values for test set predictions. They are computed on the full dataset (which includes training samples) using the fitted model.

If you need SHAP values for specific test predictions, you must compute them separately using the same explainers.

### 3. Boosting Model Component Space

For the boosting model, SHAP values are computed on the SVD-reduced feature space (80 components), not the original TF-IDF features (2000+ words).

**Implication**: You cannot directly interpret which words are important for boosting. You can only see which SVD components (latent features) are important.

### 4. Binary Classification

SHAP values are computed for the positive class (evasive). For multi-class problems (if the label column has > 2 classes), SHAP values would be computed per class.

### 5. Computational Cost

SHAP computation can be expensive:
- **Logistic Regression**: Fast (LinearExplainer is O(n_features))
- **Random Forest**: Moderate (TreeExplainer is O(n_trees * n_features))
- **Boosting**: Moderate to Slow (depends on n_iterations)

## Integration with Reports

### Including in Report Figures

1. **Global Feature Importance**: Use `shap_summary.png` directly or recreate from `shap_summary.json`
2. **Local Explanations**: Plot top features for specific samples using `shap_samples.json`
3. **Cross-Model Comparison**: Compare `mean_abs_shap` across families using `xai_summary.json`

### Example: Feature Importance Table

```python
import json
import pandas as pd

families = ["logreg", "tree", "boosting"]
results = {}

for family in families:
    with open(f"artifacts/explainability/phase6/{family}/shap_summary.json") as f:
        summary = json.load(f)
    results[family] = pd.DataFrame({
        "feature": summary["feature_names"][:10],
        "importance": summary["mean_abs_shap"][:10]
    })

# Combine into comparison table
comparison = pd.concat(
    {family: df.set_index("feature")["importance"] for family, df in results.items()},
    axis=1
)
print(comparison.head(10))
```

## Troubleshooting

### Error: "Model file not found"

**Cause**: Phase 5 models were not saved during training.

**Solution**: Re-run Phase 5 training with the updated runner that saves models:

```bash
make model-phase5
```

### Error: "Cannot cast ufunc 'isnan' input from dtype('O')"

**Cause**: TreeExplainer requires dense matrices, but sparse matrix was provided.

**Solution**: This is handled in the code. If you see this error, ensure you're using the latest version of `src/explainability.py`.

### Empty SHAP Values

**Cause**: Model has no feature importance (all predictions are the same).

**Solution**: Check model performance metrics. If accuracy is ~50% for binary classification, the model may not be learning patterns.

## References

- [SHAP Documentation](https://shap.readthedocs.io/)
- [TreeExplainer for Scikit-Learn](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
- [LinearExplainer Documentation](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html)
- [SHAP Values for Machine Learning Explainability](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ab48)

---

**Last Updated**: 2026-02-09
**Phase**: 06 - Transformer, Explainability & Label Diagnostics
**Plan**: 02 - Classical Model Explainability
