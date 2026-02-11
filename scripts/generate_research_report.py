"""Generate the canonical markdown research report from a phase-7 manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting import (
    ReportContextError,
    build_report_context,
    load_report_manifest,
    normalize_path,
    utc_now_iso,
)


def _git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="artifacts/reports/phase7/provenance_manifest.json",
        help="Path to phase-7 provenance manifest JSON.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/reports/phase7/report.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--template",
        default="templates/research_report.md.j2",
        help="Jinja template path used to render report markdown.",
    )
    parser.add_argument(
        "--project-root",
        default=str(ROOT),
        help="Project root used to resolve manifest artifact paths.",
    )
    return parser.parse_args()


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def _pipeline_run_id(project_root: Path) -> str:
    summary_path = project_root / "artifacts/reports/phase7/run_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        started = str(payload.get("started_at", "")).replace("-", "").replace(":", "")
        started = started.replace("T", "-").replace("Z", "")
        if started:
            return f"phase7-{started}-{_git_sha()}"
    return f"phase7-manual-{_git_sha()}"


def main() -> int:
    args = parse_args()
    manifest_path = _resolve(args.manifest)
    output_path = _resolve(args.output)
    template_path = _resolve(args.template)
    project_root = _resolve(args.project_root)

    try:
        manifest = load_report_manifest(manifest_path)
        report_context = build_report_context(manifest, project_root=project_root)
    except (ValueError, ReportContextError) as exc:
        print(f"Failed to build report context: {exc}", file=sys.stderr)
        return 1

    if not template_path.exists():
        print(f"Template not found: {template_path}", file=sys.stderr)
        return 1

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path.name)

    metadata = {
        "generated_at": utc_now_iso(),
        "git_sha": _git_sha(),
        "pipeline_run_id": _pipeline_run_id(project_root),
    }

    rendered = template.render(metadata=metadata, report=report_context)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered.strip() + "\n", encoding="utf-8")

    print(f"wrote report: {normalize_path(output_path, base=ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
