from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.run_research_pipeline import PipelineStage, run_pipeline


def _touch_command(path: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            f"p=Path(r'{path.as_posix()}'); "
            "p.parent.mkdir(parents=True, exist_ok=True); "
            "p.write_text('ok\\n', encoding='utf-8')"
        ),
    )


def _success_stage(key: str, output: Path) -> PipelineStage:
    return PipelineStage(
        key=key,
        command=_touch_command(output),
        outputs=(output,),
        hint=f"rerun {key}",
    )


def _failing_stage(key: str, output: Path) -> PipelineStage:
    return PipelineStage(
        key=key,
        command=(sys.executable, "-c", "import sys; sys.exit(9)"),
        outputs=(output,),
        hint=f"inspect {key}",
    )


def test_pipeline_orders_stages_and_writes_logs(tmp_path: Path) -> None:
    output_root = tmp_path / "reports"
    stage_one_out = tmp_path / "a" / "one.txt"
    stage_two_out = tmp_path / "b" / "two.txt"

    stages = [
        _success_stage("stage_one", stage_one_out),
        _success_stage("stage_two", stage_two_out),
    ]

    exit_code, summary = run_pipeline(
        stages,
        output_root=output_root,
        skip_existing=False,
        from_stage=None,
    )

    assert exit_code == 0
    assert summary["status"] == "passed"
    assert [row["stage"] for row in summary["stages"]] == ["stage_one", "stage_two"]
    assert [row["status"] for row in summary["stages"]] == ["passed", "passed"]

    for idx, key in enumerate(["stage_one", "stage_two"], start=1):
        assert (output_root / "logs" / f"{idx:02d}_{key}.log").exists()

    run_summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["status"] == "passed"


def test_pipeline_fails_fast_and_returns_non_zero(tmp_path: Path) -> None:
    output_root = tmp_path / "reports"
    first_out = tmp_path / "a" / "one.txt"
    second_out = tmp_path / "b" / "two.txt"
    third_out = tmp_path / "c" / "three.txt"

    stages = [
        _success_stage("stage_one", first_out),
        _failing_stage("stage_two", second_out),
        _success_stage("stage_three", third_out),
    ]

    exit_code, summary = run_pipeline(
        stages,
        output_root=output_root,
        skip_existing=False,
        from_stage=None,
    )

    assert exit_code == 1
    assert summary["status"] == "failed"
    assert [row["stage"] for row in summary["stages"]] == ["stage_one", "stage_two"]
    assert summary["stages"][1]["status"] == "failed"
    assert not third_out.exists()


def test_pipeline_supports_skip_existing_and_from_stage(tmp_path: Path) -> None:
    output_root = tmp_path / "reports"
    first_out = tmp_path / "a" / "one.txt"
    second_out = tmp_path / "b" / "two.txt"
    first_out.parent.mkdir(parents=True, exist_ok=True)
    first_out.write_text("already\n", encoding="utf-8")

    stages = [
        _success_stage("stage_one", first_out),
        _success_stage("stage_two", second_out),
    ]

    exit_code, summary = run_pipeline(
        stages,
        output_root=output_root,
        skip_existing=True,
        from_stage="stage_two",
    )

    assert exit_code == 0
    assert [row["stage"] for row in summary["stages"]] == ["stage_two"]
    assert summary["stages"][0]["status"] == "passed"
    assert second_out.exists()
