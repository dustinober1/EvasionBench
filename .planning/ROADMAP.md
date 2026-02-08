# Roadmap: EvasionBench Script-First Research Portfolio

**Created:** 2026-02-08
**Source requirements:** `.planning/REQUIREMENTS.md`

## Coverage Summary

- v1 requirements: 31
- Mapped requirements: 31
- Unmapped requirements: 0
- Phases: 9

## Phases

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | Delivery Foundation & CI Baseline | Establish reliable project execution scaffolding and CI policy for script-first development | QUAL-03 | 3 |
| 2 | Data Lineage & Experiment Backbone | Make data and experiment tracking reproducible with DVC + MLflow | DATA-01, DATA-02, DATA-03, DATA-04 | 4 |
| 3 | Statistical & Linguistic Core Analyses | Implement publication-grade core EDA and linguistic analysis scripts | ANLY-01, ANLY-02, ANLY-03, ANLY-04 | 5 |
| 4 | Q-A Interaction Research Analyses | Add semantic, topic, and question-behavior analyses tied to evasion hypotheses | ANLY-05, ANLY-06 | 4 |
| 5 | Classical Baseline Modeling | Deliver strong traditional ML baselines and comparison metrics | MODL-01, MODL-02, MODL-04 | 4 |
| 6 | Transformer, Explainability & Label Diagnostics | Train transformer models and produce explainability plus label-quality insights | MODL-03, MODL-05, XAI-01, XAI-02, XAI-03 | 5 |
| 7 | One-Command Research Reporting Pipeline | Build the end-to-end orchestration that outputs Markdown + HTML/PDF with traceable artifacts | RPT-01, RPT-02, RPT-03, RPT-04 | 5 |
| 8 | FastAPI + React Research Experience | Expose inference and research outputs through API and professional React UI | API-01, API-02, API-03, UI-01, UI-02 | 5 |
| 9 | Test Hardening & Reproducible Runtime Packaging | Ensure robust unit/integration/smoke validation and reproducible containerized execution | QUAL-01, QUAL-02, QUAL-04 | 4 |

## Phase Status

| Phase | Status | Verified |
|-------|--------|----------|
| 1 | Complete | 2026-02-08 |
| 2 | Complete | 2026-02-08 |
| 3 | Complete | 2026-02-08 |
| 4-9 | Pending | - |

## Phase Details

### Phase 1: Delivery Foundation & CI Baseline

**Goal:** Establish reliable project execution scaffolding and CI policy for script-first development.

**Requirements:** QUAL-03

**Success criteria:**
1. CI workflow runs on each push/PR and reports clear pass/fail status for core checks.
2. Script-first project entrypoints and task conventions are documented and executable.
3. Repository structure reflects notebook removal direction and script-first ownership boundaries.

### Phase 2: Data Lineage & Experiment Backbone

**Goal:** Make data and experiment tracking reproducible with DVC + MLflow.

**Requirements:** DATA-01, DATA-02, DATA-03, DATA-04

**Success criteria:**
1. A documented script command downloads and caches EvasionBench deterministically.
2. Automated validation checks schema, row count, and checksum with failing exit codes on mismatch.
3. `dvc repro` reproduces data-prep artifacts from tracked stages.
4. Training/evaluation runs emit parameters and metrics to MLflow with run metadata.

### Phase 3: Statistical & Linguistic Core Analyses

**Goal:** Implement publication-grade core EDA and linguistic analysis scripts.

**Requirements:** ANLY-01, ANLY-02, ANLY-03, ANLY-04

**Success criteria:**
1. Scripts generate labeled class distribution and data-quality tables/figures suitable for publication.
2. Length distribution analyses include statistical test outputs and interpretation-ready summaries.
3. Lexical and n-gram analyses are reproducible and stratified by evasiveness label.
4. Readability/POS/discourse marker analyses are generated with consistent chart/table formatting.
5. All generated artifacts are versioned and consumed by report build steps.

### Phase 4: Q-A Interaction Research Analyses

**Goal:** Add semantic, topic, and question-behavior analyses tied to evasion hypotheses.

**Requirements:** ANLY-05, ANLY-06

**Success criteria:**
1. Q-A semantic similarity computation runs reproducibly and exports analysis-ready outputs.
2. Topic modeling and question-type analyses produce reproducible tables and charts.
3. Findings include explicit hypothesis framing and interpretation notes for report inclusion.
4. Artifacts are integrated into the common outputs directory for downstream report generation.

### Phase 5: Classical Baseline Modeling

**Goal:** Deliver strong traditional ML baselines and comparison metrics.

**Requirements:** MODL-01, MODL-02, MODL-04

**Success criteria:**
1. TF-IDF + Logistic Regression baseline trains and evaluates from scripts.
2. At least one tree/boosting baseline trains and evaluates from scripts.
3. Per-class metrics and confusion matrices are generated for every classical model run.
4. Outputs are stored with run metadata and consumable by reporting/dashboard layers.

### Phase 6: Transformer, Explainability & Label Diagnostics

**Goal:** Train transformer models and produce explainability plus label-quality insights.

**Requirements:** MODL-03, MODL-05, XAI-01, XAI-02, XAI-03

**Success criteria:**
1. Transformer classifier training/evaluation runs reproducibly on available local hardware.
2. Best-performing model artifacts are versioned with metadata suitable for registry usage.
3. Explainability artifacts are produced for both classical and transformer model families.
4. Label-quality diagnostics produce actionable noise/ambiguity findings with examples.
5. All outputs are exportable into final report sections without manual notebook intervention.

### Phase 7: One-Command Research Reporting Pipeline

**Goal:** Build the end-to-end orchestration that outputs Markdown + HTML/PDF with traceable artifacts.

**Requirements:** RPT-01, RPT-02, RPT-03, RPT-04

**Success criteria:**
1. A single command executes full pipeline stages from data through report outputs.
2. Markdown report generation includes methodological explanations and linked figures/tables.
3. HTML/PDF are generated from the same report source with stable formatting.
4. Every report figure/table includes source stage metadata for reproducibility.
5. Pipeline failures are surfaced with actionable logs and non-zero exit status.

### Phase 8: FastAPI + React Research Experience

**Goal:** Expose inference and research outputs through API and professional React UI.

**Requirements:** API-01, API-02, API-03, UI-01, UI-02

**Success criteria:**
1. FastAPI serves single-example inference endpoint with validated request/response schemas.
2. FastAPI serves batch inference endpoint with robust error handling.
3. API exposes model/version metadata and run provenance details.
4. React dashboard presents core research charts and model comparison views.
5. React dashboard supports sample-level prediction + explanation inspection workflows.

### Phase 9: Test Hardening & Reproducible Runtime Packaging

**Goal:** Ensure robust unit/integration/smoke validation and reproducible containerized execution.

**Requirements:** QUAL-01, QUAL-02, QUAL-04

**Success criteria:**
1. Unit test suites cover key data/feature/model/report modules with meaningful assertions.
2. Integration and smoke tests validate pipeline execution and API health paths.
3. Docker workflows can build and run API/dashboard services reproducibly.
4. CI gating incorporates relevant test suites for merge confidence.

---
*Last updated: 2026-02-08 after completing Phase 3*
