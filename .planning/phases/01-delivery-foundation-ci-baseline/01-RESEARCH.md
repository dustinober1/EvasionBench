# Phase 1: Delivery Foundation & CI Baseline - Research

**Researched:** 2026-02-08
**Domain:** Python script-first project scaffolding and GitHub Actions CI baseline
**Confidence:** HIGH

## User Constraints

No `*-CONTEXT.md` decisions were provided for this phase directory at planning time.

## Summary

Phase 1 is primarily about operational discipline: repository structure, script-first entrypoints, and CI that runs on every push and pull request. The current codebase already has a basic CI workflow (`.github/workflows/ci.yml`) and core execution surfaces (`Makefile`, `scripts/download_data.py`, `pytest` smoke test), but script-first ownership boundaries are still implicit and notebook-first guidance remains in top-level docs.

The roadmap only maps requirement `QUAL-03` to this phase, but Phase 1 success criteria require three deliverables: reliable CI outcomes, executable script/task conventions, and explicit repository boundaries aligned with notebook deprecation direction. Planning should therefore include both automation enforcement and documentation boundary-setting.

**Primary recommendation:** Implement Phase 1 as three plans: (1) script-first execution contract, (2) CI hardening for consistent pass/fail gates, (3) migration-facing docs that codify boundaries and remove notebook-first default guidance.

## Standard Stack

### Core
| Library/Tool | Purpose | Why Standard |
|---|---|---|
| GitHub Actions | CI on push/PR | Existing workflow already in place and aligned with requirement `QUAL-03` |
| pytest | Fast test gate | Already used in CI and local smoke tests |
| black | Formatting/lint gate | Already configured via `pyproject.toml` and CI |
| Makefile + scripts/ | Script-first task surface | Gives stable command UX independent of notebook tooling |

### Supporting
| Tool | Purpose | When to Use |
|---|---|---|
| pre-commit | Local guardrails mirroring CI | Before committing and for style consistency |
| docs/ markdown runbooks | Team-visible conventions and boundaries | To enforce notebook migration direction and ownership |

## Architecture Patterns

### Pattern 1: Single Command Surface for Core Dev Tasks
**What:** Expose deterministic commands (`make` + scripts) for install, lint, test, CI-local checks.
**When to use:** Every phase moving forward so execution remains script-first.

### Pattern 2: CI as Enforcement, Docs as Contract
**What:** CI enforces quality checks; docs define expected workflow and repository boundaries.
**When to use:** Foundation phases where process quality is part of deliverable scope.

### Anti-Patterns to Avoid
- **Notebook-first onboarding:** makes reproducibility and CI drift likely.
- **Hidden task entrypoints:** if commands are spread ad-hoc, automation and contributor behavior diverge.
- **Overloading Phase 1 with data/model work:** this phase is infrastructure/process baseline only.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---|---|---|---|
| CI orchestration | Custom shell-only CI runner | GitHub Actions workflow | Native push/PR integration and visible status checks |
| Task UX | Custom CLI framework | Makefile + focused scripts | Lower maintenance and already compatible with repo |
| Style enforcement | Custom formatter wrappers | `black --check` + pre-commit | Existing project standard and predictable outputs |

## Common Pitfalls

### Pitfall 1: CI and local commands diverge
**What goes wrong:** CI passes/fails for different reasons than local checks.
**How to avoid:** Add one local command that mirrors CI checks exactly.

### Pitfall 2: Script-first stated but notebook-first documented
**What goes wrong:** Contributors continue starting from notebooks.
**How to avoid:** Update README/contributing docs to make scripts the default, notebooks legacy/supporting.

### Pitfall 3: Boundaries are implicit
**What goes wrong:** New logic lands in notebooks instead of `src/` and `scripts/`.
**How to avoid:** Document ownership boundaries and acceptance rules in docs.

## Open Questions

1. Should CI add dependency pin/lock validation now or defer to a later reproducibility phase?
2. Should notebook removal happen in this phase or only policy/documentation plus forward guardrails?
3. Is there a preferred command alias for full local CI parity (e.g., `make ci-check` vs `make verify`)?

## Planning Impact

- Keep plans small (2-3 tasks each) and map directly to the three phase success criteria.
- Use wave sequencing so docs can depend on finalized task conventions and CI commands.
- Include explicit verification commands in every task to support `gsd-execute-phase` automation.
