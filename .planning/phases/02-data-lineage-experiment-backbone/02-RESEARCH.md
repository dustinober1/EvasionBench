# Phase 2: Data Lineage & Experiment Backbone - Research

**Researched:** 2026-02-08
**Domain:** Reproducible data lineage with DVC and experiment tracking with MLflow
**Confidence:** HIGH

## User Constraints

No `*-CONTEXT.md` decisions were provided for this phase directory at planning time.

## Summary

Phase 2 must turn current ad-hoc dataset download and experimental code into a deterministic, script-first lineage backbone. The repository already includes a starter `dvc.yaml`, `scripts/download_data.py`, and MLflow guide docs, but the current setup lacks a strict data contract, robust validation failures, and run-level experiment metadata guarantees.

The roadmap maps `DATA-01..DATA-04` to this phase. Planning should enforce one canonical data command, explicit data integrity checks, a multi-stage DVC graph reproducible via `dvc repro`, and MLflow logging wired into script-based train/eval execution. A plan split that keeps data foundation first and parallelizes DVC hardening with MLflow instrumentation gives good execution speed without sacrificing traceability.

**Primary recommendation:** Implement Phase 2 as three plans: (1) deterministic dataset acquisition + lineage manifest, (2) DVC-backed validation and data-prep reproducibility, (3) MLflow-tracked experiment execution with explicit run metadata.

## Standard Stack

### Core
| Library/Tool | Purpose | Why Standard |
|---|---|---|
| `datasets` (HuggingFace) | Source dataset acquisition | Already used in `src/data.py`; supports revision pinning for reproducibility |
| DVC | Data lineage and stage reproducibility | Already present (`dvc.yaml`) and directly required by success criteria |
| MLflow | Parameter/metric tracking and run metadata | Directly required by `DATA-04`; works with local file backend first |
| `pandas` + `pyarrow` | Schema and checksum-aware data I/O | Existing parquet flow relies on it |

### Supporting
| Tool | Purpose | When to Use |
|---|---|---|
| `hashlib` | File checksum generation for data integrity | During manifest and validation steps |
| `pytest` | Validation script behavioral tests | Enforcing fail-fast behavior and schema contract checks |
| Makefile targets | Script-first command surface | Keep command UX explicit and consistent with CI/local runs |

## Architecture Patterns

### Pattern 1: Manifest-Backed Data Contract
**What:** Generate and persist dataset metadata (schema, row count, checksum, source revision) at download time, then validate against it.
**When to use:** All dataset refreshes and DVC reproduction runs.

### Pattern 2: Stage-Separated DVC Graph
**What:** Model data lifecycle as discrete stages (`download -> validate -> prepare`) with explicit deps/outs.
**When to use:** Any transformation that must be reproducible and auditable.

### Pattern 3: Script-First MLflow Logging
**What:** Route training/evaluation through a script entrypoint that logs params, metrics, and run tags from CLI/config.
**When to use:** Every baseline run that should appear in experiment history.

### Anti-Patterns to Avoid
- Validation logic embedded only in notebooks.
- Implicit data assumptions (no stored row-count/schema/checksum contract).
- Running model experiments without run metadata (dataset revision, git SHA, stage provenance).
- Mixing DVC and MLflow concerns in a single monolithic script.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---|---|---|---|
| Pipeline orchestration | Custom DAG engine | DVC stages in `dvc.yaml` | Built-in reproducibility and artifact lineage |
| Experiment registry | Homegrown JSON log files | MLflow tracking store | Standard UI/query model and run metadata |
| Data integrity | Ad-hoc prints/manual inspection | Deterministic validator script + tests | Reliable non-zero exits and CI compatibility |

## Common Pitfalls

### Pitfall 1: Dataset source changes silently
**What goes wrong:** Same command produces different data over time.
**How to avoid:** Pin dataset revision/commit and persist it in manifest + MLflow tags.

### Pitfall 2: DVC stage appears reproducible but contract is weak
**What goes wrong:** `dvc repro` succeeds on malformed data.
**How to avoid:** Make validation stage blocking with strict schema/row/checksum assertions.

### Pitfall 3: MLflow captures metrics without provenance
**What goes wrong:** Runs are not comparable or auditable.
**How to avoid:** Log run tags for dataset checksum, DVC stage hash context, and code version.

## Open Questions

1. Should schema validation enforce exact column order or only required-column presence + dtype compatibility?
2. Should expected row count/checksum be pinned to one canonical dataset snapshot in-repo, or refreshed through an explicit maintenance command?
3. Do we want local-only MLflow tracking for v1, or also pre-wire remote tracking URI/env support in this phase?

## Planning Impact

- Keep plan count to three with 2-3 tasks each for manageable execution context.
- Put deterministic download + manifest first, then run DVC and MLflow plans in parallel wave 2.
- Require explicit verification commands (`python ...`, `dvc repro`, `pytest`) in every plan for execution automation.
