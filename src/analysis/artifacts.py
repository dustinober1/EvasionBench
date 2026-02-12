"""Shared artifact contract helpers for phase analyses."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PHASE3_FAMILIES = ("core_stats", "lexical", "linguistic_quality")
PHASE4_FAMILIES = ("semantic_similarity", "topic_modeling", "question_behavior")
INDEX_FILENAME = "artifact_index.json"


def ensure_phase3_layout(output_root: str | Path) -> dict[str, Path]:
    """Create deterministic phase-3 output directories and return their paths."""
    root = Path(output_root)
    paths = {
        "root": root,
        "core_stats": root / "core_stats",
        "lexical": root / "lexical",
        "linguistic_quality": root / "linguistic_quality",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_phase4_layout(output_root: str | Path) -> dict[str, Path]:
    """Create deterministic phase-4 output directories and return their paths."""
    root = Path(output_root)
    paths = {
        "root": root,
        "semantic_similarity": root / "semantic_similarity",
        "topic_modeling": root / "topic_modeling",
        "question_behavior": root / "question_behavior",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def phase3_family_dir(output_root: str | Path, family: str) -> Path:
    if family not in PHASE3_FAMILIES:
        raise ValueError(f"Unsupported family '{family}'")
    return ensure_phase3_layout(output_root)[family]


def phase4_family_dir(output_root: str | Path, family: str) -> Path:
    if family not in PHASE4_FAMILIES:
        raise ValueError(f"Unsupported family '{family}'")
    return ensure_phase4_layout(output_root)[family]


def _normalize_files(root: Path, files: Iterable[str | Path]) -> list[str]:
    normalized: list[str] = []
    for file_path in files:
        p = Path(file_path)
        try:
            normalized.append(str(p.relative_to(root)))
        except ValueError:
            normalized.append(str(p))
    return sorted(set(normalized))


def _validate_phase4_metadata(stage: str, metadata: dict[str, Any]) -> None:
    required_keys = {"hypothesis_summary", "analysis_version"}
    missing = sorted(required_keys - set(metadata.keys()))
    if missing:
        raise ValueError(
            f"Phase-4 stage '{stage}' metadata missing required keys: {', '.join(missing)}"
        )
    if not str(metadata["hypothesis_summary"]).strip():
        raise ValueError(
            f"Phase-4 stage '{stage}' metadata key 'hypothesis_summary' must be non-empty"
        )


def write_artifact_index(
    output_root: str | Path,
    *,
    stage: str,
    generated_files: Iterable[str | Path],
    source_data: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Upsert an entry in ``artifact_index.json`` for downstream discovery."""
    layout = ensure_phase3_layout(output_root)
    root = layout["root"]
    index_path = root / INDEX_FILENAME

    payload: dict[str, Any]
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        payload = {"phase": "phase3", "entries": []}

    entries: list[dict[str, Any]] = payload.get("entries", [])
    entries = [entry for entry in entries if entry.get("stage") != stage]

    entry = {
        "stage": stage,
        "source_data": str(Path(source_data)),
        "generated_files": _normalize_files(root, generated_files),
        "metadata": metadata or {},
    }
    entries.append(entry)
    payload["entries"] = sorted(entries, key=lambda item: str(item["stage"]))

    index_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index_path


def write_phase4_artifact_index(
    output_root: str | Path,
    *,
    stage: str,
    generated_files: Iterable[str | Path],
    source_data: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Upsert a phase-4 entry in ``artifact_index.json`` with contract validation."""
    layout = ensure_phase4_layout(output_root)
    root = layout["root"]
    index_path = root / INDEX_FILENAME

    if stage not in PHASE4_FAMILIES:
        raise ValueError(f"Unsupported phase-4 stage '{stage}'")

    entry_metadata = metadata or {}
    _validate_phase4_metadata(stage, entry_metadata)

    payload: dict[str, Any]
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        payload = {"phase": "phase4", "entries": []}

    if payload.get("phase") != "phase4":
        raise ValueError("artifact_index.json phase mismatch: expected 'phase4'")

    entries: list[dict[str, Any]] = payload.get("entries", [])
    entries = [entry for entry in entries if entry.get("stage") != stage]

    entry = {
        "stage": stage,
        "source_data": str(Path(source_data)),
        "generated_files": _normalize_files(root, generated_files),
        "metadata": entry_metadata,
    }
    entries.append(entry)
    payload["entries"] = sorted(entries, key=lambda item: str(item["stage"]))
    payload["families"] = list(PHASE4_FAMILIES)

    index_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index_path
