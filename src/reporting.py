"""Shared report manifest contracts and provenance helpers for phase 7."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

MANIFEST_VERSION = "phase7-report-manifest-v1"
REQUIRED_MANIFEST_SECTIONS = (
    "dataset",
    "analyses",
    "models",
    "explainability",
    "diagnostics",
)
REQUIRED_PROVENANCE_KEYS = ("stage", "script", "artifact_path", "generated_at")


class ManifestValidationError(ValueError):
    """Raised when the phase-7 report manifest contract is invalid."""


class ReportContextError(ValueError):
    """Raised when report generation context cannot be derived from manifest."""


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format with ``Z`` suffix."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def utc_from_timestamp(ts: float) -> str:
    """Convert epoch timestamp to UTC ISO-8601 format."""
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def normalize_path(path: str | Path, *, base: str | Path | None = None) -> str:
    """Normalize a path string and make it relative to ``base`` when possible."""
    raw = Path(path)
    if base is not None:
        try:
            raw = raw.resolve().relative_to(Path(base).resolve())
        except ValueError:
            pass
    return raw.as_posix()


def classify_artifact_kind(path: str | Path) -> str:
    """Infer report artifact kind from file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
        return "figure"
    if suffix in {".csv", ".tsv", ".json", ".parquet"}:
        return "table"
    return "artifact"


def build_entry_id(stage: str, artifact_path: str | Path) -> str:
    """Build a deterministic entry id for cross-format traceability."""
    slug = str(artifact_path).replace("/", "_").replace(".", "_")
    return f"{stage}:{slug}"


def make_provenance_entry(
    *,
    stage: str,
    script: str,
    artifact_path: str,
    generated_at: str,
    kind: str | None = None,
    entry_id: str | None = None,
    title: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a validated provenance entry for a manifest section."""
    resolved_kind = kind or classify_artifact_kind(artifact_path)
    payload: dict[str, Any] = {
        "id": entry_id or build_entry_id(stage, artifact_path),
        "stage": stage,
        "script": script,
        "artifact_path": artifact_path,
        "generated_at": generated_at,
        "kind": resolved_kind,
    }
    if title:
        payload["title"] = title
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def _sorted_entries(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in entries]
    rows.sort(
        key=lambda row: (
            str(row.get("stage", "")),
            str(row.get("script", "")),
            str(row.get("artifact_path", "")),
            str(row.get("id", "")),
        )
    )
    return rows


def validate_provenance_entry(entry: Mapping[str, Any]) -> None:
    """Validate a single provenance entry."""
    missing = [key for key in REQUIRED_PROVENANCE_KEYS if key not in entry]
    if missing:
        raise ManifestValidationError(
            "Manifest entry missing required provenance keys: " + ", ".join(missing)
        )

    for key in REQUIRED_PROVENANCE_KEYS:
        if not str(entry.get(key, "")).strip():
            raise ManifestValidationError(
                f"Manifest entry field '{key}' must be non-empty"
            )

    if "id" not in entry or not str(entry["id"]).strip():
        raise ManifestValidationError("Manifest entry field 'id' must be non-empty")


def validate_manifest_sections(sections: Mapping[str, Any]) -> None:
    """Validate required manifest sections and entry contracts."""
    missing_sections = [
        section for section in REQUIRED_MANIFEST_SECTIONS if section not in sections
    ]
    if missing_sections:
        raise ManifestValidationError(
            "Manifest sections missing required keys: " + ", ".join(missing_sections)
        )

    for section in REQUIRED_MANIFEST_SECTIONS:
        entries = sections[section]
        if not isinstance(entries, list):
            raise ManifestValidationError(
                f"Manifest section '{section}' must be a list"
            )
        for idx, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ManifestValidationError(
                    f"Manifest section '{section}' entry {idx} must be an object"
                )
            validate_provenance_entry(entry)


def validate_report_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate top-level manifest schema and required sections."""
    for key in ("manifest_version", "generated_at", "generated_by", "sections"):
        if key not in manifest:
            raise ManifestValidationError(
                f"Manifest missing required top-level key: {key}"
            )

    if not str(manifest.get("manifest_version", "")).strip():
        raise ManifestValidationError(
            "Manifest field 'manifest_version' must be non-empty"
        )
    if not str(manifest.get("generated_at", "")).strip():
        raise ManifestValidationError("Manifest field 'generated_at' must be non-empty")
    if not str(manifest.get("generated_by", "")).strip():
        raise ManifestValidationError("Manifest field 'generated_by' must be non-empty")

    sections = manifest["sections"]
    if not isinstance(sections, Mapping):
        raise ManifestValidationError("Manifest field 'sections' must be an object")

    validate_manifest_sections(sections)


def build_report_manifest(
    *,
    section_entries: Mapping[str, Iterable[Mapping[str, Any]]],
    generated_by: str = "scripts/build_report_manifest.py",
    generated_at: str | None = None,
    manifest_version: str = MANIFEST_VERSION,
) -> dict[str, Any]:
    """Build a deterministic, validated report manifest payload."""
    sections: dict[str, list[dict[str, Any]]] = {}
    for section in REQUIRED_MANIFEST_SECTIONS:
        sections[section] = _sorted_entries(section_entries.get(section, []))

    payload: dict[str, Any] = {
        "manifest_version": manifest_version,
        "generated_at": generated_at or utc_now_iso(),
        "generated_by": generated_by,
        "sections": sections,
    }

    validate_report_manifest(payload)
    return payload


def load_json(path: str | Path) -> dict[str, Any]:
    """Read and decode JSON from ``path``."""
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON payload with stable formatting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def load_report_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a phase-7 report manifest file."""
    manifest = load_json(path)
    validate_report_manifest(manifest)
    return manifest


def get_section_entries(
    manifest: Mapping[str, Any], section: str
) -> list[dict[str, Any]]:
    """Return entries for a section from a validated manifest."""
    validate_report_manifest(manifest)
    return [dict(item) for item in manifest["sections"][section]]


def build_traceability_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Build a map of figure/table ids to provenance metadata."""
    validate_report_manifest(manifest)

    traceability: dict[str, dict[str, Any]] = {}
    for entries in manifest["sections"].values():
        for entry in entries:
            if entry.get("kind") not in {"figure", "table"}:
                continue
            entry_id = str(entry["id"])
            traceability[entry_id] = {
                "stage": entry["stage"],
                "script": entry["script"],
                "artifact_path": entry["artifact_path"],
                "generated_at": entry["generated_at"],
                "kind": entry["kind"],
            }

    return dict(sorted(traceability.items(), key=lambda row: row[0]))


def resolve_artifact_path(
    artifact_path: str | Path, *, project_root: str | Path
) -> Path:
    """Resolve manifest artifact path relative to ``project_root``."""
    candidate = Path(artifact_path)
    if candidate.is_absolute():
        return candidate
    return Path(project_root) / candidate


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _select_entry(
    entries: Iterable[Mapping[str, Any]],
    *,
    suffix: str,
    stage: str | None = None,
) -> dict[str, Any] | None:
    for entry in entries:
        if stage and entry.get("stage") != stage:
            continue
        if str(entry.get("artifact_path", "")).endswith(suffix):
            return dict(entry)
    return None


def build_report_context(
    manifest: Mapping[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    """Build normalized report context from validated manifest content."""
    validate_report_manifest(manifest)

    sections = manifest["sections"]
    missing_segments = [
        segment for segment in REQUIRED_MANIFEST_SECTIONS if len(sections[segment]) == 0
    ]
    if missing_segments:
        raise ReportContextError(
            "Manifest has empty required sections: " + ", ".join(missing_segments)
        )

    dataset_entries = get_section_entries(manifest, "dataset")
    analyses_entries = get_section_entries(manifest, "analyses")
    model_entries = get_section_entries(manifest, "models")
    explainability_entries = get_section_entries(manifest, "explainability")
    diagnostics_entries = get_section_entries(manifest, "diagnostics")

    phase5_summary_entry = _select_entry(
        model_entries, suffix="artifacts/models/phase5/run_summary.json"
    )
    phase5_summary: dict[str, Any] = {}
    if phase5_summary_entry is not None:
        loaded = _load_json_if_exists(
            resolve_artifact_path(
                phase5_summary_entry["artifact_path"], project_root=project_root
            )
        )
        if loaded:
            phase5_summary = loaded

    best_classical_family = "unknown"
    best_classical_metrics: dict[str, Any] = {}
    if phase5_summary:
        sorted_families = sorted(
            phase5_summary.items(),
            key=lambda row: float(row[1].get("f1_macro", 0.0)),
            reverse=True,
        )
        best_classical_family, best_classical_metrics = sorted_families[0]

    transformer_metrics_entry = _select_entry(
        model_entries,
        suffix="artifacts/models/phase6/transformer/metrics.json",
        stage="phase6_transformer_baselines",
    )
    transformer_metrics = {}
    if transformer_metrics_entry is not None:
        loaded = _load_json_if_exists(
            resolve_artifact_path(
                transformer_metrics_entry["artifact_path"], project_root=project_root
            )
        )
        if loaded:
            transformer_metrics = loaded

    xai_summary_entry = _select_entry(
        explainability_entries,
        suffix="artifacts/explainability/phase6/xai_summary.json",
        stage="phase6_xai_classical",
    )
    xai_summary = {}
    if xai_summary_entry is not None:
        loaded = _load_json_if_exists(
            resolve_artifact_path(
                xai_summary_entry["artifact_path"], project_root=project_root
            )
        )
        if loaded:
            xai_summary = loaded

    diagnostics_summary_entry = _select_entry(
        diagnostics_entries,
        suffix="artifacts/diagnostics/phase6/label_diagnostics_summary.json",
        stage="phase6_label_diagnostics",
    )
    diagnostics_summary = {}
    if diagnostics_summary_entry is not None:
        loaded = _load_json_if_exists(
            resolve_artifact_path(
                diagnostics_summary_entry["artifact_path"], project_root=project_root
            )
        )
        if loaded:
            diagnostics_summary = loaded

    return {
        "manifest_version": manifest["manifest_version"],
        "manifest_generated_at": manifest["generated_at"],
        "dataset_entries": dataset_entries,
        "analyses_entries": analyses_entries,
        "analysis_figures": [
            row for row in analyses_entries if row.get("kind") == "figure"
        ],
        "analysis_tables": [row for row in analyses_entries if row.get("kind") == "table"],
        "model_entries": model_entries,
        "best_classical_family": best_classical_family,
        "best_classical_metrics": best_classical_metrics,
        "transformer_metrics": transformer_metrics,
        "explainability_entries": explainability_entries,
        "xai_summary": xai_summary,
        "diagnostics_entries": diagnostics_entries,
        "diagnostics_summary": diagnostics_summary,
        "traceability": build_traceability_map(manifest),
    }
