# EvasionBench Research Summary

## Research Question

**Can we automatically detect evasion patterns in earnings-call Q&A transcripts?**

Corporate executives sometimes evade analyst questions during earnings calls — providing indirect, vague, or off-topic responses. This project builds and evaluates machine learning models to classify Q&A pairs into three evasion categories: **direct**, **intermediate**, and **fully_evasive**.

## Dataset

- **Source:** [FutureMa/EvasionBench](https://huggingface.co/datasets/FutureMa/EvasionBench) on HuggingFace
- **Domain:** Financial earnings call transcripts
- **Labels:** Three-class evasion taxonomy (direct, intermediate, fully_evasive)

## Methodology

### Feature Engineering
- TF-IDF vectorization on combined `question [SEP] answer` text
- SVD dimensionality reduction for tree-based models
- Transformer tokenization for BERT-based models

### Models Evaluated

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| **Logistic Regression** | **0.643** | **0.534** | Best overall; TF-IDF + sklearn Pipeline |
| XGBoost (Boosting) | 0.609 | 0.450 | TF-IDF + SVD + XGBClassifier |
| Decision Tree | 0.587 | 0.436 | TF-IDF + DecisionTreeClassifier |

### Explainability Analysis
- **SHAP** values computed for all classical models to identify key features driving predictions
- **LIME** explanations for individual predictions
- **Captum** integrated gradients for transformer model interpretability

### Label Quality Analysis
- **Cleanlab** used to identify potentially mislabeled examples
- Near-duplicate detection to find redundant training examples
- Suspect example analysis with confidence scoring

## Key Findings

1. **Logistic Regression outperforms tree-based models** on this task (F1: 0.534 vs 0.450/0.436), suggesting the decision boundary is relatively linear in TF-IDF space.
2. **The intermediate class is hardest to classify** — models struggle to distinguish partially evasive responses from direct or fully evasive ones.
3. **SHAP analysis reveals** that hedging language, topic shifts, and response length are strong predictive features.
4. **Label quality analysis** identified potential annotation inconsistencies in the intermediate category, partially explaining model difficulty.

## Limitations

- Three-class taxonomy may oversimplify evasion patterns
- TF-IDF features lose word order and semantic nuance
- Dataset is domain-specific to financial earnings calls — generalization unknown
- Macro F1 scores indicate room for improvement, particularly on minority classes

## Future Work

- Fine-tune larger language models (e.g., FinBERT) for domain-specific transfer learning
- Explore multi-task learning with evasion intensity as a continuous variable
- Investigate cross-domain transfer to other Q&A contexts (press conferences, congressional hearings)
- Augment features with discourse structure and pragmatic analysis
