"""Run the one-command phase-7 research reporting pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


@dataclass(frozen=True)
class PipelineStage:
    key: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    hint: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/evasionbench_prepared.parquet",
        help="Prepared input dataset path.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/reports/phase7",
        help="Phase-7 reporting output root.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stages when expected outputs already exist.",
    )
    parser.add_argument(
        "--from-stage",
        default=None,
        help="Start execution from this stage key.",
    )
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    target = Path(path)
    if target.is_absolute():
        return target
    return ROOT / target


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_default_stages(input_path: str | Path, output_root: str | Path) -> list[PipelineStage]:
    input_abs = _resolve(input_path)
    report_root = _resolve(output_root)

    return [
        PipelineStage(
            key="phase3_analysis",
            command=(
                sys.executable,
                "scripts/run_phase3_analyses.py",
                "--input",
                str(input_abs),
                "--output-root",
                "artifacts/analysis/phase3",
                "--families",
                "all",
            ),
            outputs=(ROOT / "artifacts/analysis/phase3/artifact_index.json",),
            hint="Run missing NLP prerequisites and retry phase3 analysis.",
        ),
        PipelineStage(
            key="phase4_analysis",
            command=(
                sys.executable,
                "scripts/run_phase4_analyses.py",
                "--input",
                str(input_abs),
                "--output-root",
                "artifacts/analysis/phase4",
                "--families",
                "all",
            ),
            outputs=(ROOT / "artifacts/analysis/phase4/artifact_index.json",),
            hint="Review topic/semantic analysis dependencies and rerun phase4.",
        ),
        PipelineStage(
            key="phase5_models",
            command=(
                sys.executable,
                "scripts/run_classical_baselines.py",
                "--input",
                str(input_abs),
                "--output-root",
                "artifacts/models/phase5",
                "--families",
                "all",
                "--compare",
            ),
            outputs=(ROOT / "artifacts/models/phase5/run_summary.json",),
            hint="Check classical baseline training errors and rerun phase5 models.",
        ),
        PipelineStage(
            key="phase6_transformer",
            command=(
                sys.executable,
                "scripts/run_transformer_baselines.py",
                "--input",
                str(input_abs),
                "--output-root",
                "artifacts/models/phase6/transformer",
            ),
            outputs=(ROOT / "artifacts/models/phase6/transformer/metrics.json",),
            hint="Validate transformer dependencies/hardware and rerun phase6 transformer.",
        ),
        PipelineStage(
            key="phase6_xai_classical",
            command=(
                sys.executable,
                "scripts/run_explainability_analysis.py",
                "--data",
                str(input_abs),
                "--models-root",
                "artifacts/models/phase5",
                "--output-root",
                "artifacts/explainability/phase6",
                "--families",
                "all",
            ),
            outputs=(ROOT / "artifacts/explainability/phase6/xai_summary.json",),
            hint="Ensure phase5 model artifacts exist before running classical XAI.",
        ),
        PipelineStage(
            key="phase6_xai_transformer",
            command=(
                sys.executable,
                "scripts/run_transformer_explainability.py",
                "--model-path",
                "artifacts/models/phase6/transformer/model",
                "--data-path",
                str(input_abs),
                "--output-root",
                "artifacts/explainability/phase6/transformer",
            ),
            outputs=(ROOT / "artifacts/explainability/phase6/transformer/transformer_xai_summary.json",),
            hint="Ensure phase6 transformer model is available before transformer explainability.",
        ),
        PipelineStage(
            key="phase6_diagnostics",
            command=(
                sys.executable,
                "scripts/run_label_diagnostics.py",
                "--input",
                str(input_abs),
                "--output-root",
                "artifacts/diagnostics/phase6",
            ),
            outputs=(ROOT / "artifacts/diagnostics/phase6/label_diagnostics_summary.json",),
            hint="Inspect label-diagnostics dependencies and rerun diagnostics stage.",
        ),
        PipelineStage(
            key="report_manifest",
            command=(
                sys.executable,
                "scripts/build_report_manifest.py",
                "--output",
                str(report_root / "provenance_manifest.json"),
            ),
            outputs=(report_root / "provenance_manifest.json",),
            hint="Resolve missing artifact prerequisites reported by manifest builder.",
        ),
    ]


def _outputs_ready(outputs: Sequence[Path]) -> bool:
    return all(path.exists() for path in outputs)


def _stage_log_path(output_root: Path, ordinal: int, key: str) -> Path:
    return output_root / "logs" / f"{ordinal:02d}_{key}.log"


def _run_stage(stage: PipelineStage, log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"$ {' '.join(stage.command)}\n\n")
        stream.flush()
        proc = subprocess.run(
            list(stage.command),
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return proc.returncode, stage.hint


def run_pipeline(
    stages: Sequence[PipelineStage],
    *,
    output_root: Path,
    skip_existing: bool,
    from_stage: str | None,
) -> tuple[int, dict]:
    stage_list = list(stages)
    keys = [stage.key for stage in stage_list]

    if from_stage:
        if from_stage not in keys:
            raise ValueError(f"Unknown --from-stage value '{from_stage}'. Available: {', '.join(keys)}")
        stage_list = stage_list[keys.index(from_stage) :]

    execution_start = _utc_now()
    statuses: list[dict] = []
    failure: dict | None = None

    for index, stage in enumerate(stage_list, start=1):
        log_path = _stage_log_path(output_root, index, stage.key)
        output_paths = [_display_path(path) for path in stage.outputs]

        if skip_existing and _outputs_ready(stage.outputs):
            statuses.append(
                {
                    "stage": stage.key,
                    "status": "skipped_existing",
                    "log": _display_path(log_path),
                    "outputs": output_paths,
                }
            )
            print(f"[SKIP] {stage.key} (outputs already exist)")
            continue

        print(f"[RUN ] {stage.key}")
        code, hint = _run_stage(stage, log_path)
        if code == 0:
            statuses.append(
                {
                    "stage": stage.key,
                    "status": "passed",
                    "log": _display_path(log_path),
                    "outputs": output_paths,
                }
            )
            print(f"[PASS] {stage.key}")
            continue

        failure = {
            "stage": stage.key,
            "status": "failed",
            "exit_code": code,
            "log": _display_path(log_path),
            "hint": hint,
            "outputs": output_paths,
        }
        statuses.append(failure)
        print(f"[FAIL] {stage.key} (exit={code})")
        print(f"       log: {failure['log']}")
        print(f"       hint: {hint}")
        break

    summary = {
        "pipeline": "phase7_research_reporting",
        "started_at": execution_start,
        "completed_at": _utc_now(),
        "git_sha": _git_sha(),
        "skip_existing": skip_existing,
        "from_stage": from_stage,
        "stages": statuses,
    }
    if failure is not None:
        summary["status"] = "failed"
        summary["failure"] = failure
    else:
        summary["status"] = "passed"

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return (1 if failure else 0), summary


def main() -> int:
    args = parse_args()
    output_root = _resolve(args.output_root)

    try:
        stages = build_default_stages(args.input, args.output_root)
        exit_code, _ = run_pipeline(
            stages,
            output_root=output_root,
            skip_existing=args.skip_existing,
            from_stage=args.from_stage,
        )
        return exit_code
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
