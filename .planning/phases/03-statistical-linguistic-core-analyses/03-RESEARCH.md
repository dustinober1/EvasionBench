# Phase 3: Statistical & Linguistic Core Analyses - Research

**Researched:** 2026-02-08
**Domain:** Script-first publication-grade EDA and linguistic analysis pipeline
**Confidence:** HIGH

## User Constraints

No `*-CONTEXT.md` decisions were provided for this phase directory at planning time.

## Summary

Phase 3 must convert current notebook-first statistical and linguistic exploration into deterministic scripts that produce report-ready artifacts. The codebase has a reproducible data backbone from Phase 2 (`data/processed/evasionbench_prepared.parquet`, DVC stages, MLflow), but analysis modules are still minimal (`src/features.py` placeholder, simple plotting helper).

Notebook references (`notebooks/01_data_quality_and_statistics.ipynb`, `notebooks/02_linguistic_patterns.ipynb`) already encode target analyses: class distribution, data quality checks, answer/question length tests (Kruskal + Mann-Whitney), lexical/n-gram comparisons, readability metrics, hedging/discourse marker frequency, and POS patterns. Planning should preserve these hypotheses and outputs while enforcing script-first reproducibility and consistent artifact layout.

**Primary recommendation:** Implement Phase 3 as four executable plans: (1) shared analysis artifact contract + orchestration, (2) core EDA and length statistics, (3) lexical and n-gram comparisons, (4) readability/POS/discourse analyses. Keep plans parallel after shared foundations to accelerate execution while preserving consistent output schema.

## Standard Stack

### Core
| Library/Tool | Purpose | Why Standard |
|---|---|---|
| `pandas`/`numpy` | deterministic tabular stats + aggregations | already used in scripts and data pipeline |
| `scipy.stats` | non-parametric significance testing for length analyses | directly supports roadmap statistical-test requirement |
| `scikit-learn` vectorizers | n-gram frequency/TF-IDF extraction | already in dependencies and prior notebooks |
| `matplotlib` + `seaborn` | publication-style figures with deterministic render settings | already standard in repo and requirements |
| DVC (`dvc.yaml`) | reproducible analysis-stage execution with tracked outputs | needed so generated artifacts are versioned |

### Supporting
| Tool | Purpose | When to Use |
|---|---|---|
| `spacy` | POS proportions by evasiveness label | in readability/POS/discourse pipeline |
| `textstat` | readability indices | in readability analysis |
| `pytest` | unit/integration checks on generated artifact contracts | every plan should add or extend tests |
| JSON/CSV artifacts | stable data exchange to reporting phase | all analyses should emit table + metadata outputs |

## Architecture Patterns

### Pattern 1: Analysis Artifact Contract
**What:** Every analysis script writes outputs under a deterministic structure (for example `artifacts/analysis/phase3/...`) including tables (`.csv`/`.json`), figures (`.png`), and metadata.
**When to use:** All Phase 3 analyses.

### Pattern 2: One Script per Analysis Family
**What:** Separate CLI entrypoints by analysis family (core stats, lexical/n-gram, readability/POS/discourse) with shared utility modules.
**When to use:** Keep implementation focused and independently testable.

### Pattern 3: DVC-Tracked Analysis Stages
**What:** Add DVC stages for each analysis script using prepared parquet as dependency and artifact directories as outputs.
**When to use:** To satisfy reproducibility and artifact versioning requirements.

### Anti-Patterns to Avoid
- Reintroducing notebooks as primary execution path.
- Non-deterministic sampling/ordering in analysis outputs.
- Writing interpretation text only to stdout with no persisted summary artifacts.
- Mixing multiple analysis families into one monolithic script.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---|---|---|---|
| Statistical tests | custom hypothesis-testing implementation | `scipy.stats` (`kruskal`, `mannwhitneyu`) | trusted, tested implementation |
| POS tagging heuristics | regex-only POS approximation | `spacy` POS pipeline | robust POS tags with standard model |
| Artifact lineage | ad-hoc folder copies | DVC stage outs + deterministic paths | reproducibility and traceability |

## Common Pitfalls

### Pitfall 1: Statistical outputs not interpretation-ready
**What goes wrong:** p-values are generated but not tied to explicit effect direction or concise narrative summary.
**How to avoid:** Emit summary JSON/Markdown with test statistic, p-value, directional interpretation, and sample sizes.

### Pitfall 2: N-gram outputs unstable across runs
**What goes wrong:** token ordering or tie handling creates diff churn.
**How to avoid:** deterministic preprocessing, fixed vectorizer params, stable sorting on score + token.

### Pitfall 3: POS/readability pipeline fragile to missing models
**What goes wrong:** scripts fail silently or produce partial outputs when `spacy` model unavailable.
**How to avoid:** explicit model checks, actionable failure messages, and optional setup docs/commands.

## Open Questions

1. Should POS analysis use `en_core_web_sm` as mandatory dependency in phase execution, or should POS be gracefully skipped with explicit warning artifacts?
2. Should lexical analysis include both raw frequency and TF-IDF ranking tables, or only one canonical ranking for report simplicity?
3. Should DVC track full-resolution figures directly or only structured tables and regenerate figures on demand?

## Planning Impact

- Use 4 plans in 2 waves: wave 1 shared contract/orchestration, wave 2 parallel analysis families.
- Make each plan emit explicit verification commands that can run headless in CI/local.
- Include artifact-index output so downstream report phase can consume phase 3 outputs without additional discovery logic.
