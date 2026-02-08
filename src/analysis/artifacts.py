"""Shared artifact contract helpers for phase-3 analyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


PHASE3_FAMILIES = ("core_stats", "lexical", "linguistic_quality")
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


def phase3_family_dir(output_root: str | Path, family: str) -> Path:
    if family not in PHASE3_FAMILIES:
        raise ValueError(f"Unsupported family '{family}'")
    return ensure_phase3_layout(output_root)[family]


def _normalize_files(root: Path, files: Iterable[str | Path]) -> list[str]:
    normalized: list[str] = []
    for file_path in files:
        p = Path(file_path)
        try:
            normalized.append(str(p.relative_to(root)))
        except ValueError:
            normalized.append(str(p))
    return sorted(set(normalized))


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

    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index_path
