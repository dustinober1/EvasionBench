# EvasionBench Hiring Manager Brief

## What This Project Demonstrates

This project demonstrates production-minded data science, not just notebook experimentation.

## Core Skills Shown

1. NLP modeling:
   - Multi-model benchmarking (linear, tree, boosting, transformer)
   - Class-level evaluation and error analysis
2. ML engineering:
   - Script-first architecture with tested module boundaries
   - Reproducible artifacts and phase contracts
3. Model governance:
   - Explainability outputs (SHAP/LIME/Captum)
   - Label quality diagnostics (Cleanlab)
4. Delivery:
   - FastAPI inference endpoint
   - Streamlit dashboard for stakeholder-facing exploration
   - CI-enforced quality gate

## Notable Outcomes

1. Best baseline: Logistic Regression (`0.643` accuracy, `0.534` macro-F1)
2. Diagnosed hardest class (`intermediate`) with targeted error-analysis narrative
3. End-to-end pipeline from raw data to report artifacts

## Engineering Decisions and Tradeoffs

1. Chose interpretable classical baselines first to establish reliable benchmark behavior
2. Used phase-based contracts to prevent silent artifact drift across the pipeline
3. Prioritized testability and reproducibility over notebook-only velocity

## Interview Discussion Prompts This Repo Supports

1. How model choice changed with data representation and class imbalance
2. How to detect and mitigate label noise in supervised NLP tasks
3. How to make research outputs reproducible and reviewable in CI
4. How to package DS work for non-DS stakeholders (API + dashboard + report)
