# EvasionBench Case Study

## Problem

Executives sometimes avoid directly answering analyst questions during earnings calls. This project frames evasion detection as a 3-class NLP classification task (`direct`, `intermediate`, `fully_evasive`) on Q&A transcript pairs.

## Approach

1. Build reproducible data and analysis pipeline (scripts + DVC artifacts)
2. Train baseline model families on combined `question [SEP] answer` text
3. Compare models on macro-F1 and class-level behavior
4. Add explainability and label diagnostics to validate model behavior

## Model Selection

Best-performing classical baseline:

- Logistic Regression: accuracy `0.643`, macro-F1 `0.534`
- Boosting: accuracy `0.609`, macro-F1 `0.450`
- Decision Tree: accuracy `0.587`, macro-F1 `0.436`

Why Logistic Regression won:

1. TF-IDF features appear linearly separable for many evasion patterns
2. The model is stable and fast to train
3. Coefficients are interpretable for stakeholder review

## Error Analysis

Primary failure mode: `intermediate` class ambiguity.

Observed pattern:

1. `intermediate` answers often share lexical features with both `direct` and `fully_evasive`
2. This label boundary is less consistent in annotation than the two extremes
3. Misclassifications cluster around hedging language and partial disclosures

## Explainability and Data Quality Checks

1. SHAP/LIME/Captum are used to inspect decision drivers
2. Cleanlab diagnostics identify likely suspect labels and difficult examples
3. Results Explorer surfaces these artifacts for manual review

## Engineering Outcomes

1. End-to-end script-first workflow in `scripts/` with reusable `src/` modules
2. Artifact contracts enforced in tests for phase outputs
3. CI checks: format, lint, tests, and coverage
4. FastAPI + Streamlit interfaces for practical model interaction

## What I Would Improve Next

1. Calibrated probability outputs for better confidence reliability
2. More robust intermediate-class annotation policy and adjudication
3. Domain-adapted transformer fine-tuning with error-focused sampling
