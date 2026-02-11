# Phase 7: One-Command Research Reporting Pipeline - Research

**Researched:** 2026-02-11
**Domain:** End-to-end scripted reporting orchestration, Markdown synthesis, HTML/PDF rendering, and artifact traceability
**Confidence:** HIGH

## User Constraints

No `*-CONTEXT.md` decisions were present for this phase at planning time.

## Summary

Phase 7 maps to `RPT-01`, `RPT-02`, `RPT-03`, and `RPT-04`, and must transform existing phase outputs (data validation, analyses, models, explainability, diagnostics) into one reproducible report pipeline. The repository already has script-first stage entrypoints and DVC wiring through Phase 6, which should be reused instead of reimplementing analysis/model logic.

**Primary recommendation:** plan this phase as 4 executable plans in 3 waves:
1. define report artifact contract + provenance manifest,
2. add one-command orchestration across all prerequisite stages,
3. generate canonical Markdown report and render HTML/PDF from the same source.

## Existing Foundation To Reuse

| Existing Asset | Reuse Value |
|---|---|
| `scripts/run_phase3_analyses.py` | Aggregates all phase-3 analysis artifacts deterministically |
| `scripts/run_phase4_analyses.py` | Aggregates all phase-4 analysis artifacts deterministically |
| `scripts/run_classical_baselines.py` | Produces model family outputs and comparison artifacts |
| `scripts/run_transformer_baselines.py` | Produces transformer artifacts and metadata |
| `scripts/run_explainability_analysis.py` + `scripts/run_transformer_explainability.py` | Supplies XAI outputs for report sections |
| `scripts/run_label_diagnostics.py` | Supplies label quality findings and report-ready summary |
| `dvc.yaml` + `Makefile` | Existing reproducibility/orchestration patterns |

## Standard Stack

### Core
| Library/Tool | Purpose | Why Standard |
|---|---|---|
| Python `pathlib`, `json`, `subprocess` | deterministic orchestration and manifest generation | zero extra runtime complexity |
| `jinja2` | structured Markdown templating | clean separation of data vs report text |
| `markdown` | Markdown-to-HTML rendering | lightweight, scriptable conversion |
| `weasyprint` | HTML-to-PDF rendering | deterministic PDF from same source HTML |
| DVC + Make | one-command reproducibility and automation | already project standard |

### Supporting
| Tool | Purpose | When to Use |
|---|---|---|
| `pytest` | report artifact contract and pipeline smoke tests | every plan |
| `rich` | clearer pipeline stage logs and failure summaries | orchestration plan |
| `pyyaml` | optional report section config map | if section ordering becomes configurable |

## Architecture Patterns

### Pattern 1: Report Build Context Object
Create one normalized context payload that aggregates metrics/tables/paths from all prior phases. Every renderer consumes this shared object.

### Pattern 2: Provenance-First Artifacts
Every figure/table entry in report metadata should include source script/stage, input artifact path, and generation timestamp.

### Pattern 3: Single Source -> Multi-Format Rendering
Generate Markdown once, then render HTML and PDF from that same Markdown (no hand-maintained duplicate templates).

### Pattern 4: Fail-Fast Orchestration with Stage Logs
Pipeline runner should execute prerequisite steps in deterministic order, emit per-stage logs, and return non-zero exit codes on first failure.

### Anti-Patterns To Avoid
- Manually curated report text disconnected from generated artifacts.
- Independent HTML/PDF templates that diverge from Markdown.
- Silent failures or partial success with exit code 0.
- Notebook-only rendering steps.

## Open Questions For Execution

1. Should PDF rendering default to `weasyprint` or support a `pandoc` fallback profile?
2. Should full pipeline rerun all upstream stages every time, or support `--from-phase` / `--skip-existing` controls?
3. Should report templates support multiple publication styles in Phase 7 or defer style variants to Phase 8?

## Planning Impact

- Use 4 plans in 3 waves to keep dependencies explicit while preserving parallelizable pieces.
- Enforce a report contract (`report.md`, `report.html`, `report.pdf`, `provenance_manifest.json`, `run_summary.json`).
- Add dedicated traceability tests to guarantee every reported figure/table maps to a reproducible source stage.
