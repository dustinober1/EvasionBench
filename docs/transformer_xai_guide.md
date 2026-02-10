# Transformer Explainability Guide

This guide explains how to interpret and use Captum-based attributions for DistilBERT transformer predictions in the EvasionBench project.

## Overview

Transformer explainability in EvasionBench uses **Captum's LayerIntegratedGradients** to provide word-level attributions for model predictions. This helps understand which parts of the input text (Q&A pairs) the model considers important when making evasiveness classifications.

## What is LayerIntegratedGradients?

**LayerIntegratedGradients (LIG)** is a gradient-based attribution method that:

1. **Integrates gradients** along a straight-line path from a baseline (e.g., zero input) to the actual input
2. **Attributes importance** to each input feature (token) based on how much it contributes to the model's output
3. **Targets specific layers** in the neural network (we use the embedding layer for interpretability)

### Mathematical Intuition

For each token, LIG computes:
```
Attribution(token) = (Input(token) - Baseline(token)) × Integral(∂Output/∂Input)
```

**Interpretation:**
- **Positive attribution**: Token pushes the prediction toward the predicted class
- **Negative attribution**: Token pushes the prediction away from the predicted class
- **Magnitude**: Strength of the influence

## How to Read Transformer XAI Outputs

### 1. JSON Structure

#### `transformer_xai.json`

Contains per-sample explanations:

```json
[
  {
    "sample_id": 23,
    "text": "What is the best way to hide money from the IRS? [SEP] You can use offshore accounts...",
    "true_label": "evasive",
    "predicted_label": "evasive",
    "tokens": ["what", "is", "the", "best", "way", "to", "hide", "money", ...],
    "attributions": [0.001, -0.002, 0.0005, 0.003, ...],
    "attribution_sum": 0.0073
  }
]
```

**Key fields:**
- `tokens`: List of wordpieces (special tokens like [CLS], [SEP] removed)
- `attributions`: Attribution scores for each token (same length as tokens)
- `attribution_sum`: Sanity check - sum of all attributions (should correlate with logit)

#### `transformer_xai_summary.json`

Aggregate statistics across all samples:

```json
{
  "n_samples": 20,
  "avg_attribution_sum": 0.0073,
  "top_tokens": [
    {"token": "financial", "attribution": 0.0051, "sample_id": 23},
    {"token": "business", "attribution": -0.0049, "sample_id": 38}
  ],
  "sample_indices": [23, 38, 59, ...]
}
```

### 2. HTML Visualization

Open `transformer_xai.html` in a browser to see:

- **Highlighted text**: Words colored by attribution strength
  - <span style="background: rgba(255, 100, 100, 0.5)">Red highlight</span>: Positive attribution (pushes toward prediction)
  - <span style="background: rgba(100, 100, 255, 0.5)">Blue highlight</span>: Negative attribution (pushes away from prediction)
- **Sample metadata**: True label, predicted label, attribution sum
- **Interactive inspection**: Hover over words to see exact attribution scores

## Interpreting Attributions

### Example 1: Evasive Prediction

```
Question: "How can I hide income from taxes?"
Answer: "Use offshore accounts and shell companies..."
Predicted: EVASIVE
```

**High-attribution tokens:** `hide` (+0.015), `offshore` (+0.012), `shell` (+0.010)
**Interpretation:** Model correctly identifies evasion because tax avoidance keywords trigger the evasive class.

### Example 2: Non-Evasive Prediction

```
Question: "What is the capital gains tax rate?"
Answer: "The long-term capital gains rate is 15%..."
Predicted: NON_EVASIVE
```

**High-attribution tokens:** `capital` (+0.008), `gains` (+0.007), `rate` (+0.006)
**Interpretation:** Model recognizes this as a factual question about tax rates, not evasion.

### Negative Attributions

**Tokens with negative scores** push the prediction toward the *opposite* class. For example, if the model predicts "evasive" but "legal" has a negative attribution, it means "legal" is evidence for non-evasiveness that the model is overcoming.

## How to Generate Transformer XAI Artifacts

### Using Make

```bash
# Generate transformer XAI artifacts (20 samples)
make xai-transformer

# Generate both classical and transformer XAI
make xai-all
```

### Using DVC

```bash
# Run as part of DVC pipeline
dvc repro phase6_xai_transformer
```

### Direct Script Invocation

```bash
python scripts/run_transformer_explainability.py \
  --model-path artifacts/models/phase6/transformer/model \
  --data-path data/processed/evasionbench_prepared.parquet \
  --output-root artifacts/explainability/phase6/transformer \
  --n-samples 20
```

**Options:**
- `--model-path`: Path to trained transformer checkpoint
- `--data-path`: Path to prepared dataset parquet file
- `--output-root`: Directory for XAI artifacts
- `--n-samples`: Number of samples to explain (default: 20)
- `--device`: `cpu` or `cuda` (default: cpu)

## Loading and Analyzing Attributions Programmatically

```python
import json
import pandas as pd

# Load explanations
with open("artifacts/explainability/phase6/transformer/transformer_xai.json") as f:
    explanations = json.load(f)

# Analyze top tokens across all samples
all_tokens = []
for exp in explanations:
    for token, attr in zip(exp["tokens"], exp["attributions"]):
        all_tokens.append({
            "token": token,
            "attribution": attr,
            "sample_id": exp["sample_id"],
            "predicted_label": exp["predicted_label"],
        })

df_tokens = pd.DataFrame(all_tokens)

# Top positive attributions for evasive predictions
evasive_top = df_tokens[df_tokens["predicted_label"] == "evasive"] \
    .sort_values("attribution", ascending=False) \
    .head(20)

print("Top tokens for evasive predictions:")
print(evasive_top[["token", "attribution"]])
```

## Differences from Classical SHAP Explanations

| Aspect | Transformer (Captum LIG) | Classical (SHAP) |
|--------|--------------------------|------------------|
| **Method** | Gradient-based attribution | Game-theoretic Shapley values |
| **Scope** | Local explanations (per-sample) | Global + local explanations |
| **Input** | Token-level attributions | Feature-level (TF-IDF) attributions |
| **Computational cost** | Higher (requires forward/backward passes) | Lower for linear, higher for tree |
| **Interpretability** | Word-level (more intuitive) | Feature-level (n-grams) |
| **Baseline sensitivity** | Sensitive to baseline choice | Less sensitive |

## Limitations

### 1. Computational Cost

LayerIntegratedGradients requires multiple forward/backward passes per sample. For 20 samples:
- **CPU**: ~2-3 minutes
- **GPU**: ~30-45 seconds

**Mitigation:** Use representative sampling instead of full dataset explanations.

### 2. Baseline Sensitivity

Attributions depend on the choice of baseline (we use zero tensor). Different baselines can produce different attributions.

**Mitigation:** Document baseline choice and compare multiple baselines for critical analyses.

### 3. Tokenization Artifacts

DistilBERT uses WordPiece tokenization, which splits words into subwords:
- `"financial"` → `["financial"]`
- `"illegal"` → `["il", "##legal"]`

**Impact:** Attributions for subwords need to be aggregated for word-level interpretation.

### 4. Model Specificity

Captum attributions are specific to the trained model. If the model is retrained, attributions will change even for identical predictions.

### 5. Not Causal Explanations

High attribution does **not** imply causation. Attributions reflect what the model uses, not what is truly important in the real world.

## Best Practices

1. **Start with HTML visualization** for quick inspection
2. **Use JSON for quantitative analysis** and custom visualizations
3. **Compare attributions across multiple samples** to identify patterns
4. **Validate with domain experts** - do attributions align with human intuition?
5. **Check for spurious correlations** - is the model using unexpected keywords?
6. **Document baseline and parameters** for reproducibility

## Troubleshooting

### Issue: "Model file not found"

**Cause:** Transformer model hasn't been trained yet.

**Fix:** Run `make model-phase6` or `dvc repro phase6_transformer` first.

### Issue: "CUDA out of memory"

**Cause:** Model doesn't fit in GPU memory with attribution computation.

**Fix:** Use `--device cpu` or reduce batch size in model training.

### Issue: "Text column not found"

**Cause:** Data format mismatch.

**Fix:** The script auto-creates text from question+answer. If your data has a different structure, specify `--text-col`.

### Issue: Attributions are all near zero

**Cause:** Model predictions are near-random (low confidence).

**Fix:** Train model for more epochs or with more data. Low-confidence predictions produce weak gradients.

## References

- **Captum Documentation**: https://captum.ai/
- **Integrated Gradients Paper**: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
- **Layer Integrated Gradients**: https://captum.ai/api/integrated_gradients.html#layer-integrated-gradients
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers

---

**Last Updated:** 2025-02-10
**Phase:** 06 - Transformer, Explainability & Label Diagnostics
**Plan:** 06-03 - Captum-based Transformer Explainability
