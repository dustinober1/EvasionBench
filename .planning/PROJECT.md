# EvasionBench Script-First Research Portfolio

## What This Is

EvasionBench is a brownfield migration from a notebook-centric NLP research project to a script-first Python system that produces publication-style academic reporting artifacts. It will preserve the original research ambition (EDA, linguistic analysis, Q&A interaction analysis, baseline models, transformer training, explainability, API, dashboard, and MLOps) while replacing notebook workflows with reproducible CLI/script execution. The target user is an academic reporting audience that expects methodological rigor, traceability, and high-quality charts with clear explanations.

## Core Value

A single command reproducibly generates a rigorous, publication-quality research report with charts and explanations from the EvasionBench dataset.

## Requirements

### Validated

- ✓ Dataset ingestion and local data pipeline scaffolding exist via `scripts/download_data.py`, `data/`, and `dvc.yaml` — existing
- ✓ Reusable Python module boundaries exist for data, features, modeling, evaluation, visualization, and explainability in `src/` — existing
- ✓ API serving foundation exists with FastAPI in `api/main.py` — existing
- ✓ Dashboard foundation exists in `dashboard/app.py` (currently Streamlit) — existing
- ✓ Basic CI/testing/linting foundation exists via GitHub Actions and pytest/black checks — existing
- ✓ Experiment/versioning tools are already present in stack and docs (`mlflow`, `dvc`) — existing

### Active

- [ ] Replace notebook-driven research execution with script/CLI-driven pipelines
- [ ] Implement one-command orchestration that runs data, analysis, modeling, evaluation, and report generation end to end
- [ ] Generate both Markdown and HTML/PDF research reports automatically with publication-style rigor
- [ ] Rebuild dashboard as a professional React frontend integrated with backend outputs
- [ ] Expand and harden FastAPI endpoints for batch and single prediction plus research artifact access
- [ ] Ensure full test coverage strategy (unit + integration + smoke) and reliable CI pass criteria
- [ ] Keep DVC + MLflow as first-class reproducibility and experiment-tracking workflow

### Out of Scope

- Continuing notebook parity or notebook maintenance — old notebooks can be removed per product direction
- Mobile-native application — not required for research portfolio goals
- Multi-node/distributed training infrastructure — unnecessary for v1 scope and local development constraints
- Enterprise auth/tenant management for API/dashboard — not required for a portfolio/research delivery

## Context

This repository already contains a research-oriented Python codebase with established layers for ingestion, features, models, evaluation, explainability, API serving, and dashboarding. The previous plan emphasized many notebooks for analysis and model work; the new direction is to remove notebook dependence entirely and move to script-first reproducible workflows. The user runs on Apple Silicon (Mac M1, 16 GB memory), accepts GPU acceleration where available, and wants one-command execution. The report emphasis is academic reporting quality rather than casual portfolio storytelling.

## Constraints

- **Execution model**: Script-first and automatable in one command — required to replace notebooks and improve reproducibility
- **Output format**: Must produce both Markdown and HTML/PDF reports — required for academic reporting delivery
- **Tracking stack**: DVC + MLflow required in v1 — chosen explicitly for data/model/experiment traceability
- **Platform**: Must run on Mac M1 (16 GB unified memory) — local development and validation environment
- **Dashboard direction**: React-based professional UI — replaces existing Streamlit approach for final presentation layer
- **Quality bar**: Tests must pass and system must work end to end — explicit success definition from user

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Migrate from notebook-first to script-first architecture | Improve reproducibility, automation, and maintainability | — Pending |
| Keep full original scope (analysis + ML + transformers + API/dashboard/MLOps) in v1 planning | Preserve ambition and avoid silent scope cuts | — Pending |
| Require one-command pipeline execution | Aligns with core value and operational simplicity | — Pending |
| Generate Markdown plus HTML/PDF reports | Supports both source readability and publication-grade output | — Pending |
| Use DVC + MLflow from day one | Ensures data/model lineage and experiment traceability | — Pending |
| Replace Streamlit dashboard with React frontend | Meet professional UI expectation | — Pending |
| Prioritize publication-style academic rigor in outputs | Matches target audience and evaluation criteria | — Pending |

---
*Last updated: 2026-02-08 after initialization*
