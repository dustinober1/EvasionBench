# Phase 6: Transformer, Explainability & Label Diagnostics - Research

**Researched:** 2026-02-09
**Domain:** Transformer fine-tuning, explainability (SHAP/Captum/LIME), label quality diagnostics (Cleanlab), MLflow model registry
**Confidence:** HIGH

## Summary

Phase 6 requires implementing three distinct but interconnected capabilities: (1) transformer-based classifier training and evaluation using Hugging Face ecosystem, (2) explainability artifact generation for both classical and transformer models, and (3) label quality diagnostics using Cleanlab to identify noisy/ambiguous examples. The phase must build on Phase 5's classical baselines while maintaining script-first workflows, MLflow tracking integration, and artifact contracts suitable for downstream reporting.

**Primary recommendation:** Use DistilBERT for transformer training (best speed/accuracy tradeoff for binary classification), implement SHAP for classical model explainability and Captum/LIME for transformer explainability, leverage Cleanlab's Datalab API for comprehensive label diagnostics, and adopt MLflow's model registry with proper versioning metadata for all model artifacts. All implementations should follow the project's existing artifact contract patterns from Phase 5.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | 4.x+ | Hugging Face transformer models and Trainer API | Industry standard for transformer fine-tuning with comprehensive model hub and training utilities |
| torch | 2.x+ | PyTorch backend for transformer models | Required dependency for transformers, dynamic execution for easier debugging |
| cleanlab | 2.x+ | Label quality diagnostics and noise detection | State-of-the-art automated label error detection, recently acquired by Handshake (2026), active development |
| shap | 0.40+ | Feature attribution for classical models | Standard library for SHAP values, optimized support for various model architectures including text |
| captum | 0.x+ | Model interpretability for PyTorch/transformers | Meta's official interpretability library for PyTorch, native attribution methods for transformer models |
| mlflow | 3.x+ | Experiment tracking and model registry | Already integrated in Phase 5, native pytorch and transformers flavors for model versioning |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| lime | 0.x+ | Local explanations for transformer predictions | Alternative to Captum for local explainability, LIME-LLM variants show improved performance for NLP |
| datasets | 2.x+ | Hugging Face datasets for efficient data loading | When working with large datasets requiring memory-efficient processing |
| bertviz | 0.x+ | Attention visualization for transformer models | For visualizing attention patterns in BERT-based models (optional, supplemental) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| DistilBERT | BERT-base, RoBERTa | BERT-base: 5% better accuracy but 60% slower; RoBERTa: better performance but larger model |
| SHAP | LIME, ELI5 | LIME: faster but less theoretically rigorous; ELI5: simpler but less comprehensive for text |
| Captum | Transformer SHAP, Alibi | Transformer SHAP: research methods not production-ready; Alibi: less mature ecosystem |
| Cleanlab Datalab | Manual label inspection, snorkel | Manual: infeasible at scale; Snorkel: more complex for simple label error detection |

**Installation:**
```bash
pip install transformers torch datasets cleanlab shap captum lime
# Already installed: mlflow scikit-learn pandas numpy
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── models/
│   ├── transformers.py      # Transformer training functions
│   └── label_quality.py      # Cleanlab wrapper utilities
├── explainability/
│   ├── classical.py          # SHAP for logreg/tree models
│   └── transformer.py        # Captum/LIME for transformer models
└── evaluation.py             # (existing) expand for transformer metrics

scripts/
├── run_transformer_baselines.py      # Main transformer training entrypoint
├── run_explainability_analysis.py    # Generate XAI artifacts for all models
└── run_label_diagnostics.py          # Cleanlab label quality analysis

tests/
├── test_transformer_baseline.py
├── test_explainability_artifacts.py
└── test_label_diagnostics.py
```

### Pattern 1: Transformer Fine-tuning with Hugging Face Trainer
**What:** Use the `Trainer` API for fine-tuning pre-trained transformer models on binary classification task. Provides optimized training loop, mixed precision, gradient accumulation, and automatic evaluation.

**When to use:** All transformer model training scenarios. Required for reproducible training with proper checkpointing and evaluation.

**Example:**
```python
# Source: https://huggingface.co/docs/transformers/training
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2  # Binary classification
)

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Adjust based on GPU VRAM
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# Train
trainer.train()
```

### Pattern 2: SHAP Explanations for Classical Text Classifiers
**What:** Generate SHAP values for TF-IDF + LogisticRegression and tree-based models to understand feature importance and word-level contributions.

**When to use:** XAI-01 requirement for classical model explainability. Provides global feature importance and local explanations for individual predictions.

**Example:**
```python
# Source: https://h1ros.github.io/posts/sentiment-analysis-by-shap-with-logistic-regression/
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Assume model and vectorizer are already fitted
# model: Pipeline with TfidfVectorizer + LogisticRegression
vectorizer = model.named_steps['tfidf']
classifier = model.named_steps['clf']

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Create SHAP explainer
explainer = shap.LinearExplainer(classifier, vectorizer.transform(X_train))
shap_values = explainer.shap_values(vectorizer.transform(X_test))

# Global feature importance
shap.summary_plot(shap_values, feature_names=feature_names)

# Local explanation for single instance
shap.force_plot(
    explainer.expected_value[0],
    shap_values[0][0],
    feature_names=feature_names
)
```

### Pattern 3: Captum Attribution for Transformer Models
**What:** Use Captum's LayerIntegratedGradients or LIME to generate attributions for transformer predictions, providing word-level importance scores.

**When to use:** XAI-02 requirement for transformer explainability. Captum is PyTorch-native and works well with transformer architectures.

**Example:**
```python
# Source: Captum documentation and Medium tutorials on transformer explainability
from captum.attr import LayerIntegratedGradients
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Define attribution function
lig = LayerIntegratedGradients(model, model.distilbert.embeddings)

def explain_instance(text, true_label="evasive"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    pred_label = "evasive" if model(**inputs).logits[0].argmax().item() == 1 else "non_evasive"

    # Compute attributions
    attributions = lig.attribute(
        inputs['input_ids'],
        additional_forward_args=(inputs['attention_mask']),
        target=int(true_label == "evasive")
    )

    return attributions, pred_label
```

### Pattern 4: Cleanlab Label Quality Diagnostics
**What:** Use Cleanlab's Datalab API to automatically detect label errors, outliers, near-duplicates, and other data issues in the training dataset.

**When to use:** XAI-03 requirement for label quality diagnostics. Essential for identifying noisy labels and ambiguous examples that may affect model performance.

**Example:**
```python
# Source: https://docs.cleanlab.ai/stable/tutorials/datalab/datalab_quickstart.html
import pandas as pd
from cleanlab import Datalab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load prepared data
df = pd.read_parquet("data/processed/evasionbench_prepared.parquet")

# Create features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['question'] + " [SEP] " + df['answer'])
y = df['label'].map({'evasive': 1, 'non_evasive': 0}).values

# Train a model to get pred_probs
model = LogisticRegression()
model.fit(X, y)
pred_probs = model.predict_proba(X)

# Initialize Datalab
lab = Datalab(data={'text': df['question'] + " " + df['answer'], 'label': y}, label_name='label')

# Find issues
lab.find_issues(pred_probs=pred_probs, features=X.toarray())

# Get report
report = lab.get_report()
issue_summary = lab.get_issue_summary()

# Access specific issue types
label_issues = lab.get_issues('label')
outlier_issues = lab.get_issues('outlier')

# Save problematic examples for review
problematic_indices = label_issues[label_issues['is_label_issue']].index
df.iloc[problematic_indices].to_csv('artifacts/label_diagnostics/suspect_examples.csv')
```

### Pattern 5: MLflow Model Registration for Transformers
**What:** Log transformer models with MLflow's transformers flavor, register best models in the model registry with versioned metadata.

**When to use:** MODL-05 requirement for model versioning and registry. Essential for tracking model lineage and managing deployments.

**Example:**
```python
# Source: https://mlflow.org/docs/latest/python_api/mlflow.transformers.html
import mlflow
import mlflow.transformers

# Set tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("evasionbench-transformers")

# Enable autologging
mlflow.transformers.autolog()

# Training happens within MLflow run
with mlflow.start_run(run_name="distilbert-binary-classifier"):
    # ... training code ...

    # Log custom metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1_macro": f1,
        "precision": precision,
        "recall": recall
    })

    # Log model explicitly
    model_info = mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="transformer_model",
        task="text-classification",
        input_example="Sample question text [SEP] Sample answer text",
    )

    # Register model
    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name="evasionbench-transformer",
        tags={"stage": "production", "version": "1.0.0"}
    )
```

### Anti-Patterns to Avoid
- **Training transformers without gradient checkpointing:** For GPU-constrained environments, use `gradient_checkpointing=True` in TrainingArguments to reduce memory usage
- **Using raw class indices for binary classification:** Map string labels to integers consistently (e.g., "evasive" → 1, "non_evasive" → 0) to avoid confusion
- **Ignoring sequence length limits:** BERT-based models have 512 token limits; either truncate or use long-sequence models for longer texts
- **Running Cleanlab on test set:** Label diagnostics should only use training data to avoid data leakage
- **Manual model serialization:** Use `save_pretrained()` and `mlflow.transformers.log_model()` instead of `torch.save()` for compatibility

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Transformer training loop | Custom PyTorch training loops | Hugging Face `Trainer` API | Handles mixed precision, gradient accumulation, checkpointing, logging automatically |
| Tokenization | Manual token encoding | `AutoTokenizer.from_pretrained()` | Handles special tokens, padding, attention masks correctly for each model |
| Feature attribution for classical models | Custom permutation importance | SHAP library | Mathematically rigorous, supports text models, provides both global and local explanations |
| Transformer attribution | Custom gradient computation | Captum library | Native PyTorch integration, tested attribution methods, handles complex architectures |
| Label error detection | Manual review or heuristics | Cleanlab Datalab | Probabilistic confident learning, handles multiple issue types, production-tested |
| Model versioning | Custom model serialization schemes | MLflow Model Registry | Standard MLOps practice, automatic lineage tracking, stage management |

**Key insight:** Building custom solutions for model interpretability and label quality is error-prone. Established libraries like SHAP, Captum, and Cleanlab incorporate years of research and handle edge cases (e.g., model-agnostic support, proper normalization, batch processing) that custom implementations typically miss.

## Common Pitfalls

### Pitfall 1: Insufficient GPU Memory for Transformer Training
**What goes wrong:** BERT-base models require 12GB+ VRAM for fine-tuning with batch_size=16. Running out of memory causes training to fail silently or produce OOM errors.

**Why it happens:** Transformer models store activations for backpropagation, and memory scales with sequence_length² × batch_size × hidden_dim.

**How to avoid:**
- Use DistilBERT instead of BERT-base (40% smaller, 60% faster, minimal accuracy loss)
- Enable `gradient_checkpointing=True` in TrainingArguments
- Reduce `per_device_train_batch_size` to 8 or 4 if needed
- Use `fp16=True` for mixed precision training
- For CPU-only environments, use much smaller models (e.g., `distilbert-base-uncased` with batch_size=4)

**Warning signs:** CUDA out-of-memory errors, extremely slow training on CPU, kernel crashes during training

### Pitfall 2: Label Leakage in Explainability Analysis
**What goes wrong:** Computing SHAP values or feature importance on test data before model finalization, or including test information in feature engineering.

**Why it happens:** Convenience—running explainability on all available data without respecting train/test split.

**How to avoid:**
- Only compute explainability artifacts on training data for model interpretation
- Generate separate explainability for test set only after model is finalized
- Clearly label artifacts as "train_explainability" vs "test_explainability"

**Warning signs:** Suspiciously high feature importance on test-specific patterns, explanations that don't generalize

### Pitfall 3: Data Contamination in Label Diagnostics
**What goes wrong:** Running Cleanlab on the entire dataset including test split, or using test predictions to identify label issues.

**Why it happens:** Wanting to maximize data for diagnostics, not realizing this leaks test information into the training process.

**How to avoid:**
- Only run label diagnostics on training data
- If issues are found and labels are corrected, re-split before test evaluation
- Document which data was used for diagnostics in run metadata

**Warning signs:** Test performance improves after label "cleaning" without model changes

### Pitfall 4: Inconsistent Model Loading Between Training and Inference
**What goes wrong:** Model saved with `save_pretrained()` but loaded with `torch.load()`, or tokenizer/model version mismatch.

**Why it happens:** Different serialization methods have different expectations for directory structure and file formats.

**How to avoid:**
- Always use `save_pretrained()` and `from_pretrained()` from Hugging Face
- Save both model and tokenizer to same directory
- Log model configuration with MLflow for exact reproducibility
- Include model card/instruction in MLflow run description

**Warning signs:** "Key mismatch" errors when loading, tokenizer vocabulary size mismatches

### Pitfall 5: Over-interpreting Attention Weights in Transformers
**What goes wrong:** Treating attention visualization as definitive explanations, despite research showing attention weights don't always correlate with feature importance.

**Why it happens:** Attention visualization is intuitive and easy to generate, but recent research (2025) questions its explanatory value.

**How to avoid:**
- Use attention visualization as supplementary, not primary explanation
- Rely on gradient-based attribution (Captum) or perturbation-based methods (LIME) for more reliable explanations
- Be transparent about limitations when reporting attention-based insights

**Warning signs:** Attention patterns don't align with domain knowledge, explanations contradict model predictions

## Code Examples

Verified patterns from official sources:

### Training DistilBERT for Binary Classification
```python
# Source: https://huggingface.co/docs/transformers/tasks/sequence_classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate

# Load model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Prepare dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Metrics
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")

# Training
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
```

### SHAP for Logistic Regression Pipeline
```python
# Source: https://h1ros.github.io/posts/sentiment-analysis-by-shap-with-logistic-regression/
import shap
from sklearn.pipeline import Pipeline

# Assume `model` is a fitted Pipeline: TfidfVectorizer + LogisticRegression
vectorizer = model.named_steps['tfidf']
classifier = model.named_steps['clf']

# Create explainer
explainer = shap.LinearExplainer(classifier, vectorizer.transform(X_train))

# Compute SHAP values for test set
X_test_tfidf = vectorizer.transform(X_test)
shap_values = explainer.shap_values(X_test_tfidf)

# Summary plot (global importance)
shap.summary_plot(shap_values, X_test_tfidf, feature_names=vectorizer.get_feature_names_out())

# Force plot for individual prediction
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    feature_names=vectorizer.get_feature_names_out(),
    matplotlib=True
)
```

### Cleanlab Label Diagnostics
```python
# Source: https://docs.cleanlab.ai/stable/tutorials/clean_learning/text.html
from cleanlab import Datalab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Prepare data
texts = df['question'] + " [SEP] " + df['answer']
labels = df['label'].map({'evasive': 1, 'non_evasive': 0})

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)
pred_probs = model.predict_proba(X)

# Run diagnostics
lab = Datalab(
    data={'text': texts, 'label': labels},
    label_name='label'
)
lab.find_issues(pred_probs=pred_probs, features=X.toarray())

# Get results
label_issues = lab.get_issues('label')
outlier_issues = lab.get_issues('outlier')

# Export suspect examples
suspect_idx = label_issues[label_issues['is_label_issue']].index.tolist()
df.iloc[suspect_idx].to_csv('label_issues.csv', index=False)
```

### MLflow Transformers Integration
```python
# Source: https://mlflow.org/docs/latest/python_api/mlflow.transformers.html
import mlflow
import mlflow.transformers

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("evasionbench-transformers")

# Autologging captures parameters, metrics, and models
mlflow.transformers.autolog()

with mlflow.start_run():
    # Training code here
    trainer.train()

    # Explicit model logging for registration
    model_info = mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer
        },
        artifact_path="model",
        task="text-classification",
    )

    # Register to Model Registry
    mlflow.register_model(
        model_uri=model_info.model_uri,
        name="evasionbench-distilbert",
        tags={"phase": "06", "task": "binary-classification"}
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| BERT-base for all tasks | DistilBERT for efficiency-critical applications | 2024-2025 | 60% faster training, 40% smaller models with <5% accuracy loss |
| Manual label review | Automated label error detection with Cleanlab | 2023-2024 | Systematic identification of label noise at scale |
| Attention-based explanations | Gradient-based attribution (Captum) + perturbation methods (LIME) | 2024-2025 | More reliable explanations, attention visualization now considered supplementary |
| Single model file | Model + tokenizer + config directory structure | Transformers 4.x+ | Improved compatibility, easier model sharing via Hub |
| Basic SHAP values | Multiple explanation methods (SHAP, LIME, Captum) for triangulation | 2025 | Triangulating explanations from multiple methods increases confidence |

**Deprecated/outdated:**
- Using `torch.save()` for transformer models (replaced by `save_pretrained()`)
- Interpreting attention weights as definitive explanations (research shows limited correlation with true feature importance)
- Manual label error detection based on loss thresholds (replaced by confident learning approaches in Cleanlab)
- Training transformers without mixed precision (FP16/BF16 now standard practice)

## Open Questions

1. **GPU availability for transformer training**
   - What we know: Project has `transformers`, `torch` in requirements.txt but no GPU detected in research environment
   - What's unclear: Whether production/training environment has GPU access, which affects model choice and batch size
   - Recommendation: Start with DistilBERT (smallest viable model), implement CPU fallback with small batch sizes, add GPU detection logic to adjust batch size automatically

2. **Best transformer model for financial QA evasion detection**
   - What we know: Binary classification on Q&A pairs, domain-specific language (financial earnings calls)
   - What's unclear: Whether domain-specific pre-training (e.g., FinBERT) outperforms general-purpose models
   - Recommendation: Test DistilBERT as baseline, optionally experiment with FinBERT if resources allow, document domain adaptation approach

3. **Explainability artifact format for reporting**
   - What we know: Phase 5 established artifact contract (metrics.json, classification_report.json, etc.)
   - What's unclear: What format explainability artifacts should take (JSON plots, CSV summaries, HTML reports)
   - Recommendation: Follow artifact contract pattern—JSON for machine-readable summaries, matplotlib plots for visualizations, CSV for example-level explanations

4. **Label error handling workflow**
   - What we know: Cleanlab can identify suspect labels
   - What's unclear: Whether to correct labels, remove examples, or flag for manual review
   - Recommendation: Phase 6 should identify and flag issues, not modify labels. Document findings for potential Phase 7 data cleanup iteration

## Sources

### Primary (HIGH confidence)
- [Hugging Face Transformers Training Documentation](https://huggingface.co/docs/transformers/training) - Official guide on Trainer API, fine-tuning workflows, hyperparameters
- [Hugging Face Sequence Classification Task Guide](https://huggingface.co/docs/transformers/tasks/sequence_classification) - Official binary classification tutorial with DistilBERT
- [Cleanlab Official Documentation](https://docs.cleanlab.ai/) - Label quality diagnostics, Datalab API, text classification tutorials
- [MLflow PyTorch Integration](https://mlflow.org/docs/latest/ml/deep-learning/pytorch/) - MLflow autologging, model registry, PyTorch workflows
- [MLflow Transformers API](https://mlflow.org/docs/latest/python_api/mlflow.transformers.html) - Transformers flavor, model logging, registration

### Secondary (MEDIUM confidence)
- [SHAP with Logistic Regression Tutorial](https://h1ros.github.io/posts/sentiment-analysis-by-shap-with-logistic-regression/) - Verified implementation pattern for text classification
- [DistilBERT vs BERT Comparison (Zilliz, 2024)](https://zilliz.com/learn/distilbert-distilled-version-of-bert) - Performance benchmarks showing 60% speed improvement, 40% size reduction
- [BERT VRAM Requirements (Official GitHub)](https://github.com/google-research/bert/blob/master/README.md) - 12GB minimum VRAM for BERT-base fine-tuning
- [Captum for Transformer Feature Attribution (June 2025)](https://eureka.patsnap.com/article/captum-for-pytorch-feature-attribution-for-transformer-models) - Recent tutorial on Captum for transformers
- [MLflow Model Registry Documentation](https://mlflow.org/docs/latest/ml/model-registry/) - Version control, stage management, deployment workflows
- [Cleanlab Text Classification Tutorial](https://docs.cleanlab.ai/stable/tutorials/clean_learning/text.html) - 5-minute quickstart for text label error detection
- [Data Preprocessing with Cleanlab (Analytics Vidhya, April 2025)](https://www.analyticsvidhya.com/blog/2025/04/data-preprocessing-using-cleanlab/) - Updated tutorial with 2025 best practices

### Tertiary (LOW confidence - marked for validation)
- [Sentiment Analysis by SHAP with Logistic Regression](https://h1ros.github.io/posts/sentiment-analysis-by-shap-with-logistic-regression/) - Single-source implementation, verified against SHAP docs
- [Visualize LLM Attention with Captum (Medium, 2025)](https://medium.com/@cbrackeen05/visualize-llm-attention-layers-with-captum-a-deep-dive-d82e05f06f35) - Practical tutorial but requires validation for binary classification
- [Explaining BERT with LIME (Blog, March 2025)](https://omseeth.github.io/blog/2025/Explaining_BERT/) - Single implementation example, verify with LIME documentation
- [Which LIME Should I Trust? (arXiv, March 2025)](https://arxiv.org/html/2503.24365v1) - Research paper discussing LIME limitations, requires practical validation
- [DistilBERT Performance Studies (ResearchGate, 2025)](https://www.researchgate.net/publication/397567285_Comparing_BERTBase_DistilBERT_and_RoBERTa_in_Sentiment_Analysis) - Academic comparison, needs domain-specific validation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries are industry standards with official documentation verified
- Architecture: HIGH - Hugging Face Trainer API and Cleanlab patterns are well-documented and production-tested
- Explainability: MEDIUM - SHAP/Captum patterns are established, but transformer explainability is active research area with some debate
- Pitfalls: HIGH - Based on common failure modes documented in official docs and community discussions
- Hardware requirements: MEDIUM - VRAM requirements verified from multiple sources, but specific environment unknown

**Research date:** 2026-02-09
**Valid until:** 2026-03-09 (30 days - stable domain, but transformer ecosystem moves quickly)
