"""Build a deterministic report provenance manifest for phase 7."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting import (
    ManifestValidationError,
    build_report_manifest,
    make_provenance_entry,
    normalize_path,
    utc_from_timestamp,
    write_json,
)


class ManifestBuildError(RuntimeError):
    """Raised when required report prerequisites are missing."""


@dataclass(frozen=True)
class SourceSpec:
    section: str
    stage: str
    script: str
    required_paths: tuple[Path, ...]
    include_roots: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        default="data/processed/evasionbench_prepared.parquet",
        help="Prepared dataset artifact used by analysis/model stages.",
    )
    parser.add_argument(
        "--phase3-root",
        default="artifacts/analysis/phase3",
        help="Phase-3 artifacts root.",
    )
    parser.add_argument(
        "--phase4-root",
        default="artifacts/analysis/phase4",
        help="Phase-4 artifacts root.",
    )
    parser.add_argument(
        "--phase5-root",
        default="artifacts/models/phase5",
        help="Phase-5 artifacts root.",
    )
    parser.add_argument(
        "--phase6-root",
        default="artifacts/models/phase6/transformer",
        help="Phase-6 transformer artifacts root.",
    )
    parser.add_argument(
        "--xai-root",
        default="artifacts/explainability/phase6",
        help="Phase-6 explainability artifacts root.",
    )
    parser.add_argument(
        "--diagnostics-root",
        default="artifacts/diagnostics/phase6",
        help="Phase-6 diagnostics artifacts root.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/reports/phase7/provenance_manifest.json",
        help="Manifest output JSON path.",
    )
    return parser.parse_args()


def _resolve(path_like: str) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _default_sources(args: argparse.Namespace) -> list[SourceSpec]:
    phase3_root = _resolve(args.phase3_root)
    phase4_root = _resolve(args.phase4_root)
    phase5_root = _resolve(args.phase5_root)
    phase6_root = _resolve(args.phase6_root)
    xai_root = _resolve(args.xai_root)
    diagnostics_root = _resolve(args.diagnostics_root)

    return [
        SourceSpec(
            section="dataset",
            stage="prepare_data",
            script="scripts/prepare_data.py",
            required_paths=(_resolve(args.dataset_path),),
            include_roots=(_resolve(args.dataset_path),),
        ),
        SourceSpec(
            section="analyses",
            stage="analyze_phase3",
            script="scripts/run_phase3_analyses.py",
            required_paths=(phase3_root / "artifact_index.json",),
            include_roots=(phase3_root,),
        ),
        SourceSpec(
            section="analyses",
            stage="analyze_phase4",
            script="scripts/run_phase4_analyses.py",
            required_paths=(phase4_root / "artifact_index.json",),
            include_roots=(phase4_root,),
        ),
        SourceSpec(
            section="models",
            stage="phase5_classical_baselines",
            script="scripts/run_classical_baselines.py",
            required_paths=(
                phase5_root / "run_summary.json",
                phase5_root / "logreg" / "metrics.json",
                phase5_root / "tree" / "metrics.json",
                phase5_root / "boosting" / "metrics.json",
            ),
            include_roots=(phase5_root,),
        ),
        SourceSpec(
            section="models",
            stage="phase6_transformer_baselines",
            script="scripts/run_transformer_baselines.py",
            required_paths=(
                phase6_root / "metrics.json",
                phase6_root / "run_metadata.json",
                phase6_root / "model" / "config.json",
            ),
            include_roots=(phase6_root,),
        ),
        SourceSpec(
            section="explainability",
            stage="phase6_xai_classical",
            script="scripts/run_explainability_analysis.py",
            required_paths=(xai_root / "xai_summary.json",),
            include_roots=(xai_root,),
        ),
        SourceSpec(
            section="explainability",
            stage="phase6_xai_transformer",
            script="scripts/run_transformer_explainability.py",
            required_paths=(xai_root / "transformer" / "transformer_xai_summary.json",),
            include_roots=(xai_root / "transformer",),
        ),
        SourceSpec(
            section="diagnostics",
            stage="phase6_label_diagnostics",
            script="scripts/run_label_diagnostics.py",
            required_paths=(
                diagnostics_root / "label_diagnostics_summary.json",
                diagnostics_root / "label_diagnostics_report.md",
            ),
            include_roots=(diagnostics_root,),
        ),
    ]


def _list_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []

    files = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.name != ".gitignore"
    ]
    return sorted(files)


def _validate_required(specs: list[SourceSpec]) -> None:
    missing_lines: list[str] = []
    for spec in specs:
        for required in spec.required_paths:
            if not required.exists():
                missing_lines.append(
                    f"- [{spec.stage}] missing {normalize_path(required, base=ROOT)} (run {spec.script})"
                )

    if missing_lines:
        message = "Missing required artifacts for report manifest:\n" + "\n".join(
            missing_lines
        )
        raise ManifestBuildError(message)


def generate_manifest(args: argparse.Namespace) -> dict:
    specs = _default_sources(args)
    _validate_required(specs)

    section_entries: dict[str, list[dict]] = {}

    for spec in specs:
        for root in spec.include_roots:
            for artifact in _list_files(root):
                normalized = normalize_path(artifact, base=ROOT)
                entry = make_provenance_entry(
                    stage=spec.stage,
                    script=spec.script,
                    artifact_path=normalized,
                    generated_at=utc_from_timestamp(artifact.stat().st_mtime),
                )
                section_entries.setdefault(spec.section, []).append(entry)

    return build_report_manifest(section_entries=section_entries)


def main() -> int:
    args = parse_args()

    try:
        manifest = generate_manifest(args)
    except (ManifestBuildError, ManifestValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_path = write_json(_resolve(args.output), manifest)
    print(f"wrote manifest: {normalize_path(output_path, base=ROOT)}")
    print(
        "sections: "
        + ", ".join(
            f"{section}={len(entries)}"
            for section, entries in manifest["sections"].items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
