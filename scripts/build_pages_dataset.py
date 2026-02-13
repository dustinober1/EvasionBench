"""Build a curated, size-bounded dataset for GitHub Pages publication."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting import (
    build_report_context,
    load_report_manifest,
    normalize_path,
    resolve_artifact_path,
    utc_now_iso,
    write_json,
)

ALLOWED_SUFFIXES = {
    ".csv",
    ".html",
    ".json",
    ".md",
    ".pdf",
    ".png",
    ".svg",
    ".txt",
}
BLOCKED_SUFFIXES = {
    ".bin",
    ".parquet",
    ".pkl",
    ".pt",
    ".safetensors",
}
FEATURED_FIGURE_SUFFIXES = (
    "core_stats/class_distribution.png",
    "core_stats/question_length_by_label.png",
    "linguistic_quality/evasive_marker_rate.png",
    "semantic_similarity/semantic_similarity_by_label.png",
    "topic_modeling/topic_prevalence_by_label.png",
    "question_behavior/question_behavior_by_label.png",
    "model_comparison/macro_f1_by_model.png",
    "model_comparison/per_class_f1_delta_heatmap.png",
    "logreg/shap_summary.png",
)


class PagesDatasetBuildError(RuntimeError):
    """Raised when publication dataset prerequisites are invalid."""


@dataclass(frozen=True)
class CandidateAsset:
    source_rel: str
    section: str
    stage: str
    entry_id: str
    kind: str


@dataclass(frozen=True)
class BuildInputs:
    manifest: Path
    report_html: Path
    report_pdf: Path
    report_markdown: Path
    report_traceability: Path
    report_provenance: Path
    report_run_summary: Path
    phase3_index: Path
    phase4_index: Path
    model_comparison_summary: Path
    diagnostics_summary: Path
    xai_summary: Path


REQUIRED_INPUT_FIELDS = (
    "manifest",
    "report_html",
    "report_pdf",
    "report_markdown",
    "report_traceability",
    "report_provenance",
)
OPTIONAL_INPUT_FIELDS = (
    "report_run_summary",
    "phase3_index",
    "phase4_index",
    "model_comparison_summary",
    "diagnostics_summary",
    "xai_summary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="artifacts/reports/phase7/provenance_manifest.json",
        help="Phase-7 report manifest path.",
    )
    parser.add_argument(
        "--report-html",
        default="artifacts/reports/phase7/report.html",
        help="Rendered report HTML path.",
    )
    parser.add_argument(
        "--report-pdf",
        default="artifacts/reports/phase7/report.pdf",
        help="Rendered report PDF path.",
    )
    parser.add_argument(
        "--report-markdown",
        default="artifacts/reports/phase7/report.md",
        help="Canonical report markdown path.",
    )
    parser.add_argument(
        "--report-traceability",
        default="artifacts/reports/phase7/report_traceability.json",
        help="Report traceability map path.",
    )
    parser.add_argument(
        "--report-provenance",
        default="artifacts/reports/phase7/provenance_manifest.json",
        help="Report provenance manifest path.",
    )
    parser.add_argument(
        "--report-run-summary",
        default="artifacts/reports/phase7/run_summary.json",
        help="Phase-7 pipeline run summary path.",
    )
    parser.add_argument(
        "--phase3-index",
        default="artifacts/analysis/phase3/artifact_index.json",
        help="Phase-3 artifact index path.",
    )
    parser.add_argument(
        "--phase4-index",
        default="artifacts/analysis/phase4/artifact_index.json",
        help="Phase-4 artifact index path.",
    )
    parser.add_argument(
        "--model-comparison-summary",
        default="artifacts/models/phase5/model_comparison/summary.json",
        help="Phase-5 model-comparison summary path.",
    )
    parser.add_argument(
        "--diagnostics-summary",
        default="artifacts/diagnostics/phase6/label_diagnostics_summary.json",
        help="Phase-6 diagnostics summary path.",
    )
    parser.add_argument(
        "--xai-summary",
        default="artifacts/explainability/phase6/xai_summary.json",
        help="Phase-6 classical explainability summary path.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/publish",
        help="Output root for publication assets and site data.",
    )
    parser.add_argument(
        "--project-root",
        default=str(ROOT),
        help="Project root used to resolve relative paths.",
    )
    parser.add_argument(
        "--max-total-mb",
        type=float,
        default=250.0,
        help="Maximum total published artifact size in MB.",
    )
    parser.add_argument(
        "--max-single-file-mb",
        type=float,
        default=25.0,
        help="Maximum published artifact size per file in MB.",
    )
    return parser.parse_args()


def _resolve(path_like: str, *, project_root: Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def _git_sha(project_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _normalize_absolute_path(value: str, *, project_root: Path) -> str:
    path_value = Path(value)
    if not path_value.is_absolute():
        return normalize_path(path_value)

    try:
        return normalize_path(path_value, base=project_root)
    except Exception:
        return path_value.name


def _sanitize_paths(value: Any, *, project_root: Path) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_paths(val, project_root=project_root)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [_sanitize_paths(item, project_root=project_root) for item in value]

    if isinstance(value, str):
        return _normalize_absolute_path(value, project_root=project_root)

    return value


def _iter_string_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        for item in value.values():
            values.extend(_iter_string_values(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(_iter_string_values(item))
    return values


def _to_source_rel(path: Path, *, project_root: Path) -> str:
    try:
        return normalize_path(path, base=project_root)
    except Exception as exc:
        raise PagesDatasetBuildError(
            f"Artifact path is outside project root: {path}"
        ) from exc


def _source_from_raw(raw_value: str, *, project_root: Path) -> Path | None:
    raw = Path(raw_value)
    if raw.is_absolute():
        if raw.exists():
            return raw
        return None

    candidate = project_root / raw
    if candidate.exists():
        return candidate
    return None


def _artifact_is_publishable(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in BLOCKED_SUFFIXES:
        return False
    if suffix in ALLOWED_SUFFIXES:
        return True
    return False


def _validate_required_inputs(inputs: BuildInputs) -> list[Path]:
    missing: list[str] = []
    for field in REQUIRED_INPUT_FIELDS:
        target = getattr(inputs, field)
        if not target.exists():
            missing.append(normalize_path(target, base=ROOT))

    if missing:
        raise PagesDatasetBuildError(
            "Missing required publication inputs:\n- " + "\n- ".join(missing)
        )

    missing_optional: list[Path] = []
    for field in OPTIONAL_INPUT_FIELDS:
        target = getattr(inputs, field)
        if not target.exists():
            missing_optional.append(target)

    return missing_optional


def _register_candidate(
    candidates: dict[str, CandidateAsset],
    source: Path,
    *,
    section: str,
    stage: str,
    entry_id: str,
    kind: str,
    project_root: Path,
) -> None:
    if not source.exists() or not source.is_file():
        return
    if not _artifact_is_publishable(source):
        return

    source_rel = _to_source_rel(source, project_root=project_root)
    if source_rel not in candidates:
        candidates[source_rel] = CandidateAsset(
            source_rel=source_rel,
            section=section,
            stage=stage,
            entry_id=entry_id,
            kind=kind,
        )


def _copy_assets(
    *,
    candidates: list[CandidateAsset],
    output_root: Path,
    project_root: Path,
    max_total_mb: float,
    max_single_file_mb: float,
) -> tuple[list[dict[str, Any]], dict[str, str], int]:
    assets_root = output_root / "assets"
    assets_root.mkdir(parents=True, exist_ok=True)

    max_total_bytes = int(max_total_mb * 1024 * 1024)
    max_single_file_bytes = int(max_single_file_mb * 1024 * 1024)

    total_bytes = 0
    rows: list[dict[str, Any]] = []
    published_lookup: dict[str, str] = {}

    for candidate in candidates:
        source = project_root / candidate.source_rel
        destination = assets_root / candidate.source_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

        size_bytes = destination.stat().st_size
        if size_bytes > max_single_file_bytes:
            raise PagesDatasetBuildError(
                "Published file exceeds per-file budget: "
                f"{normalize_path(destination, base=output_root)} "
                f"({size_bytes} bytes > {max_single_file_bytes})"
            )

        total_bytes += size_bytes
        if total_bytes > max_total_bytes:
            raise PagesDatasetBuildError(
                "Published dataset exceeds total budget: "
                f"{total_bytes} bytes > {max_total_bytes}"
            )

        published_path = normalize_path(destination.relative_to(output_root))
        published_lookup[candidate.source_rel] = published_path

        rows.append(
            {
                "entry_id": candidate.entry_id,
                "section": candidate.section,
                "stage": candidate.stage,
                "kind": candidate.kind,
                "source_path": candidate.source_rel,
                "published_path": published_path,
                "size_bytes": size_bytes,
            }
        )

    rows.sort(key=lambda row: (row["section"], row["stage"], row["source_path"]))
    return rows, published_lookup, total_bytes


def _attach_published_paths(
    *,
    entries: list[dict[str, Any]],
    published_lookup: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        source_rel = normalize_path(entry["artifact_path"])
        published = published_lookup.get(source_rel)
        if not published:
            continue

        row = dict(entry)
        row["published_path"] = published
        rows.append(row)

    return rows


def _featured_figures(
    *,
    analysis_entries: list[dict[str, Any]],
    explainability_entries: list[dict[str, Any]],
    published_lookup: Mapping[str, str],
) -> list[dict[str, Any]]:
    by_suffix: dict[str, dict[str, Any]] = {}

    for entry in [*analysis_entries, *explainability_entries]:
        artifact_path = normalize_path(entry["artifact_path"])
        published = published_lookup.get(artifact_path)
        if not published:
            continue
        for suffix in FEATURED_FIGURE_SUFFIXES:
            if artifact_path.endswith(suffix):
                by_suffix[suffix] = {
                    "id": entry["id"],
                    "stage": entry["stage"],
                    "artifact_path": artifact_path,
                    "published_path": published,
                    "caption": suffix.replace("_", " ").replace("/", " - "),
                }

    return [
        by_suffix[suffix] for suffix in FEATURED_FIGURE_SUFFIXES if suffix in by_suffix
    ]


def build_pages_dataset(args: argparse.Namespace) -> Path:
    project_root = Path(args.project_root).resolve()
    output_root = _resolve(args.output_root, project_root=project_root)
    data_root = output_root / "data"
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    inputs = BuildInputs(
        manifest=_resolve(args.manifest, project_root=project_root),
        report_html=_resolve(args.report_html, project_root=project_root),
        report_pdf=_resolve(args.report_pdf, project_root=project_root),
        report_markdown=_resolve(args.report_markdown, project_root=project_root),
        report_traceability=_resolve(
            args.report_traceability, project_root=project_root
        ),
        report_provenance=_resolve(args.report_provenance, project_root=project_root),
        report_run_summary=_resolve(args.report_run_summary, project_root=project_root),
        phase3_index=_resolve(args.phase3_index, project_root=project_root),
        phase4_index=_resolve(args.phase4_index, project_root=project_root),
        model_comparison_summary=_resolve(
            args.model_comparison_summary, project_root=project_root
        ),
        diagnostics_summary=_resolve(
            args.diagnostics_summary, project_root=project_root
        ),
        xai_summary=_resolve(args.xai_summary, project_root=project_root),
    )

    missing_optional_inputs = _validate_required_inputs(inputs)
    for missing in missing_optional_inputs:
        print(
            "warning: optional publication input not found: "
            + normalize_path(missing, base=project_root)
        )

    manifest = load_report_manifest(inputs.manifest)
    report_context = build_report_context(manifest, project_root=project_root)

    phase3_index = _sanitize_paths(
        _load_json_if_exists(inputs.phase3_index) or {"phase": "phase3", "entries": []},
        project_root=project_root,
    )
    phase4_index = _sanitize_paths(
        _load_json_if_exists(inputs.phase4_index) or {"phase": "phase4", "entries": []},
        project_root=project_root,
    )

    loaded_model_summary = _load_json_if_exists(inputs.model_comparison_summary)
    if loaded_model_summary is None:
        fallback_models: dict[str, Any] = {}
        if report_context.get(
            "best_classical_family"
        ) != "unknown" and report_context.get("best_classical_metrics"):
            fallback_models[str(report_context["best_classical_family"])] = dict(
                report_context["best_classical_metrics"]
            )
        model_comparison_summary = {
            "best_model_family": report_context.get("best_classical_family", "unknown"),
            "models": fallback_models,
            "artifacts": {},
        }
    else:
        model_comparison_summary = loaded_model_summary
    model_comparison_summary = _sanitize_paths(
        model_comparison_summary, project_root=project_root
    )

    diagnostics_summary = _sanitize_paths(
        _load_json_if_exists(inputs.diagnostics_summary)
        or dict(report_context.get("diagnostics_summary", {})),
        project_root=project_root,
    )
    xai_summary = _sanitize_paths(
        _load_json_if_exists(inputs.xai_summary)
        or dict(report_context.get("xai_summary", {})),
        project_root=project_root,
    )

    report_run_summary: dict[str, Any] = {}
    if inputs.report_run_summary.exists():
        report_run_summary = _sanitize_paths(
            _load_json(inputs.report_run_summary), project_root=project_root
        )

    candidates: dict[str, CandidateAsset] = {}

    for section, entries in (
        ("analyses", report_context["analyses_entries"]),
        ("models", report_context["model_entries"]),
        ("explainability", report_context["explainability_entries"]),
        ("diagnostics", report_context["diagnostics_entries"]),
    ):
        for entry in entries:
            source = resolve_artifact_path(
                entry["artifact_path"], project_root=project_root
            )
            _register_candidate(
                candidates,
                source,
                section=section,
                stage=str(entry["stage"]),
                entry_id=str(entry["id"]),
                kind=str(entry.get("kind", "artifact")),
                project_root=project_root,
            )

    static_report_assets = [
        (
            inputs.report_html,
            "reports",
            "report_render",
            "phase7:report_html",
            "artifact",
        ),
        (
            inputs.report_pdf,
            "reports",
            "report_render",
            "phase7:report_pdf",
            "artifact",
        ),
        (
            inputs.report_markdown,
            "reports",
            "report_markdown",
            "phase7:report_markdown",
            "artifact",
        ),
        (
            inputs.report_traceability,
            "reports",
            "report_render",
            "phase7:report_traceability",
            "table",
        ),
        (
            inputs.report_provenance,
            "reports",
            "report_manifest",
            "phase7:provenance_manifest",
            "table",
        ),
        (
            inputs.phase3_index,
            "analyses",
            "analyze_phase3",
            "phase3:artifact_index",
            "table",
        ),
        (
            inputs.phase4_index,
            "analyses",
            "analyze_phase4",
            "phase4:artifact_index",
            "table",
        ),
        (
            inputs.model_comparison_summary,
            "models",
            "phase5_model_comparison",
            "phase5:model_comparison_summary",
            "table",
        ),
        (
            inputs.diagnostics_summary,
            "diagnostics",
            "phase6_label_diagnostics",
            "phase6:diagnostics_summary",
            "table",
        ),
        (
            inputs.xai_summary,
            "explainability",
            "phase6_xai_classical",
            "phase6:xai_summary",
            "table",
        ),
    ]

    for source, section, stage, entry_id, kind in static_report_assets:
        _register_candidate(
            candidates,
            source,
            section=section,
            stage=stage,
            entry_id=entry_id,
            kind=kind,
            project_root=project_root,
        )

    for raw in _iter_string_values(model_comparison_summary):
        source = _source_from_raw(raw, project_root=project_root)
        if source is None:
            continue
        _register_candidate(
            candidates,
            source,
            section="models",
            stage="phase5_model_comparison",
            entry_id=f"phase5:model_comparison:{normalize_path(source, base=project_root)}",
            kind="artifact",
            project_root=project_root,
        )

    candidate_rows = sorted(candidates.values(), key=lambda row: row.source_rel)
    copied_assets, published_lookup, total_bytes = _copy_assets(
        candidates=candidate_rows,
        output_root=output_root,
        project_root=project_root,
        max_total_mb=args.max_total_mb,
        max_single_file_mb=args.max_single_file_mb,
    )

    analyses_with_links = _attach_published_paths(
        entries=report_context["analyses_entries"],
        published_lookup=published_lookup,
    )
    models_with_links = _attach_published_paths(
        entries=report_context["model_entries"],
        published_lookup=published_lookup,
    )
    explainability_with_links = _attach_published_paths(
        entries=report_context["explainability_entries"],
        published_lookup=published_lookup,
    )
    diagnostics_with_links = _attach_published_paths(
        entries=report_context["diagnostics_entries"],
        published_lookup=published_lookup,
    )

    downloads = {
        "report_html": published_lookup.get(
            normalize_path(inputs.report_html, base=project_root)
        ),
        "report_pdf": published_lookup.get(
            normalize_path(inputs.report_pdf, base=project_root)
        ),
        "report_markdown": published_lookup.get(
            normalize_path(inputs.report_markdown, base=project_root)
        ),
        "traceability": published_lookup.get(
            normalize_path(inputs.report_traceability, base=project_root)
        ),
        "provenance_manifest": published_lookup.get(
            normalize_path(inputs.report_provenance, base=project_root)
        ),
    }

    best_model_family = str(
        model_comparison_summary.get("best_model_family", "unknown")
    )
    best_metrics = dict(
        model_comparison_summary.get("models", {}).get(best_model_family, {})
    )
    transformer_metrics = dict(report_context.get("transformer_metrics", {}))

    key_findings = [
        {
            "label": "Best classical model",
            "value": best_model_family,
        },
        {
            "label": "Best classical macro F1",
            "value": str(best_metrics.get("f1_macro", "n/a")),
        },
        {
            "label": "Best classical accuracy",
            "value": str(best_metrics.get("accuracy", "n/a")),
        },
        {
            "label": "Transformer macro F1",
            "value": str(transformer_metrics.get("f1_macro", "n/a")),
        },
        {
            "label": "Label quality score",
            "value": str(diagnostics_summary.get("quality_score", "n/a")),
        },
    ]

    metadata = {
        "generated_at": utc_now_iso(),
        "generated_by": "scripts/build_pages_dataset.py",
        "git_sha": _git_sha(project_root),
        "project_root": normalize_path(project_root, base=project_root),
    }

    site_data = {
        "metadata": metadata,
        "summary": {
            "manifest_version": report_context["manifest_version"],
            "manifest_generated_at": report_context["manifest_generated_at"],
            "dataset_artifacts": len(report_context["dataset_entries"]),
            "analysis_artifacts": len(report_context["analyses_entries"]),
            "model_artifacts": len(report_context["model_entries"]),
            "explainability_artifacts": len(report_context["explainability_entries"]),
            "diagnostics_artifacts": len(report_context["diagnostics_entries"]),
            "published_assets": len(copied_assets),
            "published_size_mb": round(total_bytes / (1024 * 1024), 3),
        },
        "key_findings": key_findings,
        "downloads": downloads,
        "analysis_indexes": {
            "phase3": _sanitize_paths(phase3_index, project_root=project_root),
            "phase4": _sanitize_paths(phase4_index, project_root=project_root),
        },
        "model_comparison": model_comparison_summary,
        "best_model_family": best_model_family,
        "best_model_metrics": best_metrics,
        "transformer_metrics": transformer_metrics,
        "xai_summary": xai_summary,
        "diagnostics_summary": diagnostics_summary,
        "pipeline_status": report_run_summary,
        "artifacts": {
            "analyses": analyses_with_links,
            "models": models_with_links,
            "explainability": explainability_with_links,
            "diagnostics": diagnostics_with_links,
        },
        "featured_figures": _featured_figures(
            analysis_entries=analyses_with_links,
            explainability_entries=explainability_with_links,
            published_lookup=published_lookup,
        ),
        "traceability": _sanitize_paths(
            report_context["traceability"], project_root=project_root
        ),
    }

    publication_manifest = {
        "metadata": metadata,
        "budgets": {
            "max_total_mb": args.max_total_mb,
            "max_single_file_mb": args.max_single_file_mb,
            "actual_total_mb": round(total_bytes / (1024 * 1024), 3),
        },
        "asset_count": len(copied_assets),
        "assets": copied_assets,
    }

    site_data_path = write_json(data_root / "site_data.json", site_data)
    manifest_path = write_json(
        data_root / "publication_manifest.json", publication_manifest
    )

    print(f"wrote site data: {normalize_path(site_data_path, base=project_root)}")
    print(f"wrote publish manifest: {normalize_path(manifest_path, base=project_root)}")
    print(
        f"published assets: {len(copied_assets)} files, "
        f"{round(total_bytes / (1024 * 1024), 3)} MB"
    )

    return site_data_path


def main() -> int:
    args = parse_args()
    try:
        build_pages_dataset(args)
    except PagesDatasetBuildError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
