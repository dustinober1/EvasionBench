# EvasionBench Portfolio Improvement Plan

> **10 actionable improvements** to transform EvasionBench from a solid research repo into a standout portfolio project.  
> Prioritized by **portfolio impact** — what will impress a hiring manager or collaborator the most.

---

## 1. Wire the Dashboard & API to Real Models (Critical)

### Problem

Both `dashboard/app.py` and `api/main.py` are hardcoded stubs returning `"prediction: direct (confidence 0.9)"`. The project trains real models (LogReg, Tree, Boosting, Transformer) but never actually serves them. A recruiter who clicks "Analyze" sees a fake result and immediately loses interest.

### Implementation

1. Create `src/inference.py` — a model-loading and prediction utility:
   - Load serialized model artifacts from `artifacts/models/phase5/` or `artifacts/models/phase6/`
   - Accept raw Q&A text, tokenize/featurize, and return `{prediction, confidence, probabilities}`
   - Support model selection (LogReg, Tree Boosting, Transformer)
2. Update **`api/main.py`** to call `src/inference.py` at startup
3. Update **`dashboard/app.py`** to call the same inference module
4. Add a dropdown to the dashboard for model selection

### Files to Create/Modify

| Action | File |
|--------|------|
| Create | `src/inference.py` |
| Modify | `api/main.py` |
| Modify | `dashboard/app.py` |

### Why It Matters

A working live demo is **the single biggest differentiator** between a code-dump repo and a compelling portfolio piece. This is the highest-ROI improvement.

---

## 2. Add a Results Explorer Dashboard Page

### Problem

The project generates rich artifacts — SHAP explanations, confusion matrices, topic models, label diagnostics — but they're all buried in JSON files inside `artifacts/`. No one is going to `cat artifacts/explainability/phase6/xai_summary.json` to appreciate your work.

### Implementation

1. Create `dashboard/pages/2_Results_Explorer.py` (Streamlit multi-page app):
   - **Model Comparison tab** — interactive table/charts comparing accuracy, F1, precision across all models
   - **Explainability Viewer** — render SHAP summaries from `artifacts/explainability/phase6/`
   - **Sample Predictions** — show correctly/incorrectly classified examples with explanations
   - **Label Diagnostics** — visualize suspect examples and near-duplicates from `artifacts/diagnostics/phase6/`
2. Use Plotly for interactivity (already in `requirements.txt`)

### Files to Create/Modify

| Action | File |
|--------|------|
| Create | `dashboard/pages/2_Results_Explorer.py` |
| Modify | `dashboard/app.py` (configure as multi-page app) |

### Why It Matters

This showcases the depth of your ML work — explainability, label quality analysis, model comparison — in an accessible visual format. Hiring managers can explore your analysis without reading JSON files.

---

## 3. Write a Proper README with Badges, Architecture Diagram & Results

### Problem

The current `README.md` is 41 lines of minimal setup instructions. It reads like internal tooling docs, not a portfolio piece. There are no badges, no architecture diagram, no results summary, no screenshots, and no clear "what this project does" narrative. The `pyproject.toml` still has the placeholder `authors = ["Your Name <you@example.com>"]`.

### Implementation

1. Rewrite `README.md` with these sections:
   - **Hero section** — project title, one-line description, badges (CI status, Python version, license)
   - **What is EvasionBench?** — 2–3 paragraph explanation of the research question and approach
   - **Key Results** — summary table of model performance (F1, accuracy) across all models
   - **Architecture diagram** — Mermaid or image showing the pipeline: Data → Analysis → Modeling → XAI → Report
   - **Screenshots** — dashboard demo, sample predictions, SHAP visualizations
   - **Quick Start** — streamlined setup instructions
   - **Project Structure** — annotated directory tree
   - **Citation / References** — link to the HuggingFace dataset
2. Fix `pyproject.toml` author field

### Files to Modify

| Action | File |
|--------|------|
| Rewrite | `README.md` |
| Fix | `pyproject.toml` (author placeholder) |

### Why It Matters

The README is the **first thing anyone sees** on GitHub. A polished README with results and visuals is the difference between "bookmark and read later" and "close tab immediately."

---

## 4. Add a GitHub Actions CI Workflow

### Problem

The project references CI in `docs/ci_baseline.md` (mentions `.github/workflows/ci.yml`) and has a `ci_check.sh` script — but there is no `.github/` directory. CI doesn't actually exist. The CONTRIBUTING.md tells contributors to run `make ci-check` before PRs, but there's nothing enforcing this.

### Implementation

1. Create `.github/workflows/ci.yml`:
   ```yaml
   name: CI
   on: [push, pull_request]
   jobs:
     check:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: { python-version: '3.11' }
         - run: pip install -r requirements.txt
         - run: bash scripts/ci_check.sh
   ```
2. Add a CI badge to `README.md`
3. Optionally add a separate workflow for DVC pipeline validation

### Files to Create

| Action | File |
|--------|------|
| Create | `.github/workflows/ci.yml` |
| Modify | `README.md` (add badge) |

### Why It Matters

A green CI badge signals engineering maturity. It also catches the fact that you already document CI — you just haven't set it up yet, which looks like an oversight.

---

## 5. Pin Dependency Versions in `requirements.txt`

### Problem

Every single dependency in `requirements.txt` is unpinned (e.g., `pandas`, `torch`, `transformers`). This means:
- Builds are **not reproducible** — `pip install` grabs whatever is latest
- A breaking `numpy` or `scikit-learn` update will silently break the project
- The `environment.yml` just points back to the same unpinned `requirements.txt`

### Implementation

1. Generate a pinned lockfile:
   ```bash
   pip freeze > requirements.lock
   ```
2. Update `requirements.txt` to pin major+minor versions (e.g., `pandas>=2.1,<2.3`)
3. Add `requirements.lock` as a fully pinned reference
4. Document the difference in the README (loose pins for development, lockfile for exact reproduction)

### Files to Modify

| Action | File |
|--------|------|
| Update | `requirements.txt` (add version pins) |
| Create | `requirements.lock` (fully pinned) |

### Why It Matters

Unpinned dependencies are a red flag in any project that claims to be production-quality. It signals that builds will break on a fresh clone — and they probably will.

---

## 6. Complete the Docker Story

### Problem

The `docker/Dockerfile` exists but is a bare-minimum 7-line file that only serves the API. There is no `docker-compose.yml`, no multi-service setup, no documentation on how to use it, and the `.gitignore` references `docker-compose.override.yml` (suggesting one was planned but never created).

### Implementation

1. Enhance `docker/Dockerfile`:
   - Add a multi-stage build (builder stage for deps, slim runtime stage)
   - Add health checks
   - Set proper user permissions (don't run as root)
2. Create `docker-compose.yml` at project root:
   - `api` service (FastAPI)
   - `dashboard` service (Streamlit)
   - Shared volume for model artifacts
3. Add `make docker-up` and `make docker-build` targets
4. Optionally add a `docker/Dockerfile.dashboard` for the Streamlit app
5. Document usage in `README.md`

### Files to Create/Modify

| Action | File |
|--------|------|
| Rewrite | `docker/Dockerfile` |
| Create | `docker-compose.yml` |
| Modify | `Makefile` (add docker targets) |

### Why It Matters

A one-command `docker-compose up` demo is extremely impressive in a portfolio. It shows you think about deployment, not just research.

---

## 7. Add Type Hints and a Linter Beyond Black

### Problem

The project uses Black for formatting but has no static analysis (no `mypy`, `ruff`, `flake8`, or `pylint`). There are no type hints in the codebase. The `pre-commit-config.yaml` only runs Black and a structure check.

### Implementation

1. Add `ruff` to `requirements.txt` and configure in `pyproject.toml`:
   ```toml
   [tool.ruff]
   select = ["E", "F", "I", "UP", "B"]
   line-length = 88
   ```
2. Add `mypy` with basic config:
   ```toml
   [tool.mypy]
   python_version = "3.11"
   warn_return_any = true
   ```
3. Add type hints to the public API surfaces (`src/inference.py`, `src/models.py`, `src/evaluation.py`, `src/data.py`)
4. Add `ruff` and `mypy` hooks to `.pre-commit-config.yaml`
5. Add linting step to `ci_check.sh`

### Files to Modify

| Action | File |
|--------|------|
| Update | `pyproject.toml` (ruff + mypy config) |
| Update | `.pre-commit-config.yaml` |
| Update | `requirements.txt` (add ruff, mypy) |
| Update | `scripts/ci_check.sh` |
| Update | `src/*.py` (add type hints) |

### Why It Matters

Type hints + linting is standard practice in professional Python. Their absence suggests the code hasn't been through a production-readiness pass.

---

## 8. Add Test Coverage Reporting & Improve Coverage

### Problem

There are 26 test files in `tests/` — impressive breadth — but there's no coverage reporting configured and no visibility into what percentage of `src/` is actually exercised. Tests exist but there's no proof they're comprehensive.

### Implementation

1. Add `pytest-cov` to `requirements.txt`
2. Update `Makefile` test target:
   ```makefile
   test:
       pytest -q --cov=src --cov-report=term-missing --cov-report=html:artifacts/coverage
   ```
3. Add a coverage badge to README (use `coverage-badge` or shields.io)
4. Add a coverage threshold to CI:
   ```bash
   pytest -q --cov=src --cov-fail-under=80
   ```
5. Write tests for uncovered modules (e.g., `src/features.py`, `src/utils.py` are tiny but untested)

### Files to Modify

| Action | File |
|--------|------|
| Update | `requirements.txt` (add pytest-cov) |
| Update | `Makefile` (coverage target) |
| Update | `scripts/ci_check.sh` (coverage enforcement) |
| Update | `README.md` (coverage badge) |

### Why It Matters

Showing actual coverage numbers (especially if 80%+) demonstrates testing discipline. Without reporting, the 26 test files could be covering 20% or 95% — no one knows.

---

## 9. Add a Project Architecture / Design Document

### Problem

The `docs/` folder has 10 files, all focused on specific technical runbooks (DVC guide, MLflow guide, explainability guide). There's no high-level document that explains:
- The overall research question and methodology
- How the phases connect (data → analysis → models → XAI → report)
- Key design decisions (why script-first? why DVC? why this model selection?)
- Results and conclusions

The `papers/` directory is empty except for a placeholder README.

### Implementation

1. Create `docs/architecture.md`:
   - System architecture diagram (pipeline DAG)
   - Phase dependency graph
   - Technology choices and rationale
2. Create `docs/research_summary.md`:
   - Research question and hypothesis
   - Methodology overview
   - Key findings and model comparison
   - Limitations and future work
3. Consider writing an actual short paper in `papers/` summarizing the research

### Files to Create

| Action | File |
|--------|------|
| Create | `docs/architecture.md` |
| Create | `docs/research_summary.md` |

### Why It Matters

Hiring managers want to see **how you think**, not just how you code. A well-written architecture doc with clear decision rationale demonstrates senior-level thinking.

---

## 10. Deploy a Live Demo (Streamlit Cloud, HuggingFace Spaces, or Railway)

### Problem

The project can only be experienced by cloning the repo, installing 35+ dependencies, and running locally. The vast majority of people who discover this repo will never do that.

### Implementation

**Option A — Streamlit Community Cloud** (easiest):
1. Ensure `dashboard/app.py` works standalone (Improvement #1 is prerequisite)
2. Add a `dashboard/requirements.txt` with only the deps the dashboard needs
3. Deploy to [share.streamlit.io](https://share.streamlit.io)
4. Add the live demo link to the top of `README.md`

**Option B — HuggingFace Spaces** (best for ML):
1. Create a `spaces/` directory with a Gradio or Streamlit app
2. Push to HuggingFace Spaces
3. Automatically gets a free GPU if using Transformers

**Option C — Railway / Render** (for the API):
1. Use the Docker setup (Improvement #6)
2. Deploy both API + dashboard

### Files to Create

| Action | File |
|--------|------|
| Create | Deployment config (varies by platform) |
| Modify | `README.md` (add live demo link + badge) |

### Why It Matters

A **live demo link in the README** is the single most effective way to get someone to engage with your project. Most people will click a link; almost no one will `git clone`.

---

## Priority Matrix

| # | Improvement | Impact | Effort | Priority |
|---|------------|--------|--------|----------|
| 1 | Wire real models to dashboard/API | 🔴 Critical | Medium (4–6h) | **P0** |
| 3 | Rewrite README | 🔴 Critical | Low (2–3h) | **P0** |
| 10 | Deploy live demo | 🔴 Critical | Low (1–2h) | **P0** |
| 2 | Results Explorer dashboard page | 🟠 High | Medium (6–8h) | **P1** |
| 4 | Add GitHub Actions CI | 🟠 High | Low (1h) | **P1** |
| 5 | Pin dependency versions | 🟡 Medium | Low (0.5h) | **P1** |
| 9 | Architecture & research docs | 🟡 Medium | Medium (3–4h) | **P2** |
| 6 | Complete Docker setup | 🟡 Medium | Medium (2–3h) | **P2** |
| 8 | Test coverage reporting | 🟢 Nice | Low (1h) | **P2** |
| 7 | Type hints + linter | 🟢 Nice | Medium (4–6h) | **P3** |

## Recommended Execution Order

```
Week 1 (P0 — maximum portfolio impact):
  #1  Wire real models  ──→  #3  Rewrite README  ──→  #10  Deploy live demo

Week 2 (P1 — engineering credibility):
  #4  GitHub Actions CI  ──→  #5  Pin deps  ──→  #2  Results Explorer page

Week 3 (P2/P3 — polish):
  #9  Architecture docs  ──→  #6  Docker  ──→  #8  Coverage  ──→  #7  Type hints
```

---

## Summary

The core ML research in this repo is **strong** — 7 phases of analysis, multiple model families, SHAP explainability, label diagnostics, a reproducible DVC pipeline, and 26 test files. What's missing is the **presentation layer**: the README doesn't sell the work, the demo doesn't work, there's no CI badge, and there's no live link. The improvements above focus on making the existing excellent work **visible and accessible** to anyone who visits the repo.
