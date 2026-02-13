# GitHub Pages Publication Data Contract

## Goal

Define the exact artifact subset that can be published to GitHub Pages while preserving reproducibility and avoiding large model binaries.

## Script Entrypoint

Use `scripts/build_pages_dataset.py` as the single source of truth for publication payload assembly.

## Required Inputs

- `artifacts/reports/phase7/provenance_manifest.json`
- `artifacts/reports/phase7/report.md`
- `artifacts/reports/phase7/report.html`
- `artifacts/reports/phase7/report.pdf`
- `artifacts/reports/phase7/report_traceability.json`
- `artifacts/analysis/phase3/artifact_index.json`
- `artifacts/analysis/phase4/artifact_index.json`
- `artifacts/models/phase5/model_comparison/summary.json`
- `artifacts/diagnostics/phase6/label_diagnostics_summary.json`
- `artifacts/explainability/phase6/xai_summary.json`

Optional input:

- `artifacts/reports/phase7/run_summary.json`

## Publish Rules

### Include

- Report deliverables (`.md`, `.html`, `.pdf`, `.json`)
- Analysis and diagnostics figures (`.png`, `.svg`)
- Summary tables and metrics (`.csv`, `.json`)
- Explainability outputs needed for interpretation (`.png`, `.json`, `.html`)

### Exclude

- Model binaries (`.pkl`, `.bin`, `.pt`, `.safetensors`)
- Large intermediate datasets (`.parquet`)
- Any path outside the repository root

## Size Budgets

- Total publication payload: default `<= 250 MB`
- Single published file: default `<= 25 MB`

Budgets are enforced in `scripts/build_pages_dataset.py`.

## Output Contract

All outputs are written under `artifacts/publish/`.

- `artifacts/publish/assets/...`
  - Curated copy of publishable artifacts with repo-relative paths.
- `artifacts/publish/data/site_data.json`
  - Normalized content model consumed by the site renderer.
- `artifacts/publish/data/publication_manifest.json`
  - Per-file publication manifest with size metadata and budget summary.

No absolute filesystem paths are allowed in output JSON payloads.

## Rendering Contract

Use `scripts/render_github_pages.py` to generate static pages from `site_data.json`.

Expected pages in `artifacts/publish/site/`:

- `index.html`
- `methodology.html`
- `findings.html`
- `modeling.html`
- `explainability.html`
- `reproducibility.html`
- `static/site.css`
- `.nojekyll`

## Orchestration Contract

Use `scripts/run_pages_pipeline.py` as the one-command entrypoint.

- Stage 1: build publish dataset
- Stage 2: render static pages
- Stage 3: validate required outputs and write `pages_pipeline_summary.json`
