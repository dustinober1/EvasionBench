# Requirements: EvasionBench Script-First Research Portfolio

**Defined:** 2026-02-08
**Core Value:** A single command reproducibly generates a rigorous, publication-quality research report with charts and explanations from the EvasionBench dataset.

## v1 Requirements

### Data & Reproducibility

- [x] **DATA-01**: User can download and cache the EvasionBench dataset using a documented script command
- [x] **DATA-02**: User can verify dataset schema, row count, and checksum in an automated validation step
- [x] **DATA-03**: User can reproduce data preparation artifacts through DVC stages
- [x] **DATA-04**: User can run experiments with tracked parameters and metrics in MLflow

### Statistical & Linguistic Analysis

- [x] **ANLY-01**: User can generate class distribution and data quality statistics as report-ready tables and figures
- [x] **ANLY-02**: User can generate answer/question length distribution analysis with statistical significance tests
- [x] **ANLY-03**: User can compute lexical and n-gram feature comparisons by evasiveness label
- [x] **ANLY-04**: User can compute readability, POS, and discourse marker analyses by label
- [x] **ANLY-05**: User can run question-answer semantic similarity analysis and include findings in the report
- [x] **ANLY-06**: User can run topic and question-type analyses tied to evasion behavior hypotheses

### Modeling

- [x] **MODL-01**: User can train and evaluate TF-IDF + Logistic Regression baseline models
- [x] **MODL-02**: User can train and evaluate tree/boosting baseline models for comparison
- [ ] **MODL-03**: User can train and evaluate at least one transformer-based classifier
- [x] **MODL-04**: User can output per-class metrics and confusion matrices for each model family
- [ ] **MODL-05**: User can register best-performing model artifacts with versioned metadata

### Explainability & Label Quality

- [ ] **XAI-01**: User can generate explainability artifacts for classical models (feature importance/SHAP where applicable)
- [ ] **XAI-02**: User can generate explainability artifacts for transformer predictions on representative samples
- [ ] **XAI-03**: User can run label quality diagnostics and summarize potential noise/ambiguity cases

### Report Generation

- [ ] **RPT-01**: User can run one command that executes the full pipeline from data through reporting
- [ ] **RPT-02**: User can generate a structured Markdown research report with charts and methodological explanations
- [ ] **RPT-03**: User can generate HTML/PDF outputs from the same report source
- [ ] **RPT-04**: User can trace every figure/table in the report back to a reproducible script stage

### Serving & Visualization

- [ ] **API-01**: User can request single-example evasion predictions from FastAPI endpoints
- [ ] **API-02**: User can request batch evasion predictions from FastAPI endpoints
- [ ] **API-03**: User can retrieve model metadata and version information through API endpoints
- [ ] **UI-01**: User can view core research charts and model comparisons in a React dashboard
- [ ] **UI-02**: User can inspect sample-level predictions and explanations in the dashboard

### Engineering Quality

- [ ] **QUAL-01**: User can run unit tests for data, feature, model, and reporting modules
- [ ] **QUAL-02**: User can run integration/smoke tests that validate end-to-end pipeline and API health
- [x] **QUAL-03**: User can rely on CI to run linting and tests on every change
- [ ] **QUAL-04**: User can build and run the API/dashboard stack in Docker for reproducible dev execution

## v2 Requirements

### Advanced Research

- **RSCH-01**: User can run expanded ablation studies across prompt/model variants
- **RSCH-02**: User can run cross-domain transfer experiments beyond the primary dataset
- **RSCH-03**: User can export manuscript-ready appendix tables automatically

### Productization

- **PROD-01**: User can authenticate dashboard/API usage with role-based access
- **PROD-02**: User can deploy dashboard/API to managed cloud environments with IaC
- **PROD-03**: User can monitor production drift and alerting policies automatically

## Out of Scope

| Feature | Reason |
|---------|--------|
| Notebook-based analysis workflow | Explicitly replaced by script-first architecture |
| Mobile application | Not required for academic reporting portfolio outcomes |
| Multi-tenant enterprise access control | Adds complexity beyond v1 research deliverable goals |
| Distributed training cluster orchestration | Not required for current dataset scale and local-first workflow |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 2 | Complete |
| DATA-02 | Phase 2 | Complete |
| DATA-03 | Phase 2 | Complete |
| DATA-04 | Phase 2 | Complete |
| ANLY-01 | Phase 3 | Complete |
| ANLY-02 | Phase 3 | Complete |
| ANLY-03 | Phase 3 | Complete |
| ANLY-04 | Phase 3 | Complete |
| ANLY-05 | Phase 4 | Complete |
| ANLY-06 | Phase 4 | Complete |
| MODL-01 | Phase 5 | Complete |
| MODL-02 | Phase 5 | Complete |
| MODL-03 | Phase 6 | Pending |
| MODL-04 | Phase 5 | Complete |
| MODL-05 | Phase 6 | Pending |
| XAI-01 | Phase 6 | Pending |
| XAI-02 | Phase 6 | Pending |
| XAI-03 | Phase 6 | Pending |
| RPT-01 | Phase 7 | Pending |
| RPT-02 | Phase 7 | Pending |
| RPT-03 | Phase 7 | Pending |
| RPT-04 | Phase 7 | Pending |
| API-01 | Phase 8 | Pending |
| API-02 | Phase 8 | Pending |
| API-03 | Phase 8 | Pending |
| UI-01 | Phase 8 | Pending |
| UI-02 | Phase 8 | Pending |
| QUAL-01 | Phase 9 | Pending |
| QUAL-02 | Phase 9 | Pending |
| QUAL-03 | Phase 1 | Complete |
| QUAL-04 | Phase 9 | Pending |

**Coverage:**
- v1 requirements: 31 total
- Mapped to phases: 31
- Unmapped: 0 ✓

**Phase Status:**
- Phase 5: Complete (MODL-01, MODL-02, MODL-04)
- Phase 6: Planned (MODL-03, MODL-05, XAI-01, XAI-02, XAI-03)

---
*Requirements defined: 2026-02-08*
*Last updated: 2026-02-09 after completing Phase 5 and planning Phase 6*
