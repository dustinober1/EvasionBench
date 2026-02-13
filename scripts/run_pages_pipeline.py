"""Run the GitHub Pages publication pipeline end-to-end."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting import normalize_path


@dataclass(frozen=True)
class PipelineStage:
    key: str
    command: tuple[str, ...]
    required_outputs: tuple[Path, ...]


class PagesPipelineError(RuntimeError):
    """Raised when the pages pipeline cannot complete."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish-root",
        default="artifacts/publish",
        help="Publication root for curated assets and metadata.",
    )
    parser.add_argument(
        "--site-root",
        default="artifacts/publish/site",
        help="Rendered static site root.",
    )
    parser.add_argument(
        "--input",
        default="data/processed/evasionbench_prepared.parquet",
        help="Prepared dataset used by phase-9 analysis stages.",
    )
    parser.add_argument(
        "--max-total-mb",
        type=float,
        default=250.0,
        help="Maximum total publish dataset size in MB.",
    )
    parser.add_argument(
        "--max-single-file-mb",
        type=float,
        default=25.0,
        help="Maximum single published file size in MB.",
    )
    parser.add_argument(
        "--project-root",
        default=str(ROOT),
        help="Project root used to resolve relative paths.",
    )
    return parser.parse_args()


def _resolve(path_like: str, *, project_root: Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def _run_stage(stage: PipelineStage, *, project_root: Path) -> None:
    print(f"[RUN ] {stage.key}")
    proc = subprocess.run(
        list(stage.command),
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        message = (
            f"Stage '{stage.key}' failed with exit code {proc.returncode}\n"
            f"Command: {' '.join(stage.command)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
        raise PagesPipelineError(message)

    for output in stage.required_outputs:
        if not output.exists():
            raise PagesPipelineError(
                f"Stage '{stage.key}' did not produce required output: "
                f"{normalize_path(output, base=project_root)}"
            )


def run_pipeline(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    publish_root = _resolve(args.publish_root, project_root=project_root)
    site_root = _resolve(args.site_root, project_root=project_root)
    input_path = _resolve(args.input, project_root=project_root)

    stages = [
        PipelineStage(
            key="phase9_error_analysis",
            command=(
                sys.executable,
                "scripts/analyze_error_profiles.py",
                "--input",
                str(input_path),
                "--output-root",
                str(project_root / "artifacts/analysis/phase9/error_analysis"),
                "--project-root",
                str(project_root),
            ),
            required_outputs=(
                project_root
                / "artifacts/analysis/phase9/error_analysis/error_summary.json",
                project_root
                / "artifacts/analysis/phase9/error_analysis/misclassification_routes.csv",
                project_root / "artifacts/analysis/phase9/error_analysis/hard_cases.md",
                project_root
                / "artifacts/analysis/phase9/error_analysis/class_failure_heatmap.png",
                project_root
                / "artifacts/analysis/phase9/error_analysis/artifact_index.json",
            ),
        ),
        PipelineStage(
            key="phase9_exploration",
            command=(
                sys.executable,
                "scripts/analyze_phase9_exploration.py",
                "--input",
                str(input_path),
                "--output-root",
                str(project_root / "artifacts/analysis/phase9/exploration"),
                "--project-root",
                str(project_root),
            ),
            required_outputs=(
                project_root
                / "artifacts/analysis/phase9/exploration/temporal_summary.json",
                project_root
                / "artifacts/analysis/phase9/exploration/segment_summary.json",
                project_root
                / "artifacts/analysis/phase9/exploration/question_intent_error_map.csv",
                project_root
                / "artifacts/analysis/phase9/exploration/artifact_index.json",
            ),
        ),
        PipelineStage(
            key="build_pages_dataset",
            command=(
                sys.executable,
                "scripts/build_pages_dataset.py",
                "--output-root",
                str(publish_root),
                "--max-total-mb",
                str(args.max_total_mb),
                "--max-single-file-mb",
                str(args.max_single_file_mb),
                "--project-root",
                str(project_root),
            ),
            required_outputs=(
                publish_root / "data" / "kpi_summary.json",
                publish_root / "data" / "site_quality_report.json",
                publish_root / "data" / "site_data.json",
                publish_root / "data" / "publication_manifest.json",
            ),
        ),
        PipelineStage(
            key="render_github_pages",
            command=(
                sys.executable,
                "scripts/render_github_pages.py",
                "--publish-root",
                str(publish_root),
                "--site-data",
                str(publish_root / "data" / "site_data.json"),
                "--publication-manifest",
                str(publish_root / "data" / "publication_manifest.json"),
                "--output-root",
                str(site_root),
                "--project-root",
                str(project_root),
            ),
            required_outputs=(
                site_root / "index.html",
                site_root / "methodology.html",
                site_root / "findings.html",
                site_root / "modeling.html",
                site_root / "explainability.html",
                site_root / "reproducibility.html",
                site_root / "error-analysis.html",
                site_root / "static" / "site.css",
                site_root / ".nojekyll",
            ),
        ),
    ]

    for stage in stages:
        _run_stage(stage, project_root=project_root)

    site_quality_path = publish_root / "data" / "site_quality_report.json"
    site_quality: dict[str, object] = {}
    if site_quality_path.exists():
        loaded = json.loads(site_quality_path.read_text(encoding="utf-8"))
        site_quality = {
            "missing_fields": loaded.get("missing_fields", []),
            "placeholder_cards": loaded.get("placeholder_cards", []),
            "kpi_source": loaded.get("fallback_usage", {}).get("kpi_source"),
        }

    summary = {
        "pipeline": "github_pages_publication",
        "status": "passed",
        "publish_root": normalize_path(publish_root, base=project_root),
        "site_root": normalize_path(site_root, base=project_root),
        "stages": [stage.key for stage in stages],
        "site_quality": site_quality,
    }
    summary_path = publish_root / "data" / "pages_pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("[PASS] pages pipeline")
    print(f"site root: {normalize_path(site_root, base=project_root)}")


def main() -> int:
    args = parse_args()
    try:
        run_pipeline(args)
    except PagesPipelineError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
