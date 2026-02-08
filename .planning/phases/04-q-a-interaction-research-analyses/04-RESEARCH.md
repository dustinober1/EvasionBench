# Phase 4: Q-A Interaction Research Analyses - Research

**Researched:** 2026-02-08
**Domain:** Script-first semantic/topic/question-behavior analysis for evasion hypotheses
**Confidence:** HIGH

## User Constraints

No `*-CONTEXT.md` decisions were provided for this phase directory at planning time.

## Summary

Phase 4 should extend the now-stable Phase 3 analysis stack with interaction-level signals focused on question-answer relationships rather than answer text alone. The roadmap maps `ANLY-05` and `ANLY-06` to this phase and requires hypothesis-linked interpretation artifacts that can be consumed by report generation without manual notebook post-processing.

The existing codebase already provides a shared analysis artifact contract (`src/analysis/artifacts.py`), script-first execution patterns, DVC orchestration, and test conventions from Phase 3. Planning should reuse this infrastructure and add three focused analysis families: semantic similarity between question and answer pairs, unsupervised topic structure with label-linked contrasts, and question-type behavior analyses tied to evasiveness hypotheses.

**Primary recommendation:** Implement Phase 4 as four plans in two waves: (1) shared phase-4 orchestration/artifact contract updates, then (2) parallel semantic, topic, and question-behavior analysis plans. This preserves reproducibility while maximizing execution speed.

## Standard Stack

### Core
| Library/Tool | Purpose | Why Standard |
|---|---|---|
| `sentence-transformers` (or fallback embedding backend) | semantic similarity embeddings for Q-A pairs | robust semantic signal for `ANLY-05` |
| `scikit-learn` (`NMF`/`LDA`, vectorizers) | reproducible topic modeling and term extraction | stable API, already present dependency family |
| `pandas`/`numpy` | deterministic tabular aggregations and hypothesis metrics | existing analysis baseline |
| `matplotlib` + `seaborn` | publication-style topic and behavior charts | existing plotting conventions |
| DVC (`dvc.yaml`) | reproducible phase-4 execution and outputs | required for downstream report integration |

### Supporting
| Tool | Purpose | When to Use |
|---|---|---|
| `pytest` | unit/integration checks on analysis outputs | every plan should include deterministic schema tests |
| `spaCy` or rule-based heuristics | question-type classification features | question-behavior analysis |
| JSON/CSV/Markdown outputs | machine + human interpretation artifacts | report pipeline compatibility |

## Architecture Patterns

### Pattern 1: Per-Family Analysis Modules with Shared Artifact Contract
**What:** Separate modules/scripts for semantic, topic, and question-behavior analyses using common output/index helpers.
**When to use:** All Phase 4 implementations.

### Pattern 2: Hypothesis-First Output Schema
**What:** Every analysis writes both quantitative metrics and short interpretation notes explicitly mapped to hypotheses.
**When to use:** All summary outputs expected by report generation.

### Pattern 3: Phase-Level Orchestrator + DVC Stage
**What:** Add a single phase runner script (`run_phase4_analyses.py`) and DVC stages that call each family script.
**When to use:** Reproducible local/CI execution and artifact lineage.

### Anti-Patterns to Avoid
- Notebook-only analysis paths or manual interpretation steps.
- Topic model randomness without fixed seeds/config serialization.
- Semantic outputs without per-label distributions or hypothesis tie-ins.
- Free-form question-type labels that drift across runs.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---|---|---|---|
| Embedding model internals | custom semantic encoder training | pre-trained sentence embedding models | reliable semantic baseline, faster delivery |
| Topic discovery framework | ad-hoc clustering code | scikit-learn topic modeling with fixed seeds | reproducibility + maintainability |
| Report prose generation | manual one-off notes | deterministic summary markdown + JSON fields | supports automation and traceability |

## Common Pitfalls

### Pitfall 1: Semantic similarity is computed but not interpretable
**What goes wrong:** Only raw cosine scores are produced without per-label comparisons.
**How to avoid:** Export by-label distributions, effect-size comparisons, and concise interpretation notes.

### Pitfall 2: Topic model outputs are unstable across reruns
**What goes wrong:** Different random states and preprocessing create diff churn.
**How to avoid:** Fix random seeds, vectorizer params, vocabulary preprocessing, and deterministic top-term sorting.

### Pitfall 3: Question-type analysis drifts due to weak schema
**What goes wrong:** Labels are inconsistently defined or impossible to compare across analyses.
**How to avoid:** Define a stable taxonomy, persist classifier/rule config, and validate required output categories in tests.

## Open Questions

1. Should semantic analysis default to a local lightweight encoder for portability, with optional larger models via CLI flag?
2. Should topic analysis be answer-only, question-only, or both with paired comparisons (recommended: both if runtime allows)?
3. Should question-type detection prioritize deterministic rule-based taxonomy first, then optional learned classifier in later phases?

## Planning Impact

- Use 4 plans in 2 waves: wave 1 for shared orchestration/contract, wave 2 for parallel analysis families.
- Require all plans to emit reproducible artifacts and hypothesis-linked summaries.
- Ensure all outputs are integrated under common artifact paths and indexed for Phase 7 reporting.
