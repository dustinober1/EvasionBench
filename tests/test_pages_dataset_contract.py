from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.reporting import build_report_manifest, make_provenance_entry, write_json

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, content: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _fixture_project(project_root: Path) -> Path:
    _write(project_root / "data/processed/evasionbench_prepared.parquet", "data\n")

    _write(
        project_root / "artifacts/analysis/phase3/core_stats/class_distribution.png",
        "png\n",
    )
    _write(
        project_root
        / "artifacts/analysis/phase4/semantic_similarity/semantic_similarity_by_label.csv",
        "label,value\n",
    )

    _write(
        project_root / "artifacts/models/phase5/run_summary.json",
        json.dumps(
            {
                "logreg": {"accuracy": 0.63, "f1_macro": 0.53},
                "tree": {"accuracy": 0.58, "f1_macro": 0.43},
            }
        )
        + "\n",
    )
    _write(
        project_root / "artifacts/models/phase6/transformer/metrics.json",
        '{"accuracy": 0.5, "f1_macro": 0.33}\n',
    )
    _write(project_root / "artifacts/models/phase5/logreg/model.pkl", "binary\n")

    chart_abs = (
        project_root / "artifacts/models/phase5/model_comparison/macro_f1_by_model.png"
    )
    _write(chart_abs, "png\n")
    _write(
        project_root
        / "artifacts/models/phase5/model_comparison/per_class_f1_delta_heatmap.png",
        "png\n",
    )
    _write(
        project_root / "artifacts/models/phase5/model_comparison/model_ranking.csv",
        "model,f1\n",
    )
    _write(
        project_root
        / "artifacts/models/phase5/model_comparison/per_class_f1_comparison.csv",
        "label,f1\n",
    )
    _write(
        project_root
        / "artifacts/models/phase5/model_comparison/per_class_f1_delta.csv",
        "label,delta\n",
    )
    _write(
        project_root / "artifacts/models/phase5/model_comparison/summary.json",
        json.dumps(
            {
                "best_model_family": "logreg",
                "models": {
                    "logreg": {"accuracy": 0.63, "f1_macro": 0.53},
                    "tree": {"accuracy": 0.58, "f1_macro": 0.43},
                },
                "artifacts": {
                    "macro_f1_chart": str(chart_abs),
                },
            }
        )
        + "\n",
    )
    _write(
        project_root / "artifacts/models/phase8/selected_model.json",
        json.dumps(
            {
                "best_model_family": "logreg",
                "artifact_root": "artifacts/models/phase8/logreg",
                "metrics": {
                    "accuracy": 0.66,
                    "f1_macro": 0.57,
                    "precision_macro": 0.58,
                    "recall_macro": 0.56,
                },
                "evaluation_protocol": "stratified_5fold_cv_plus_holdout",
                "selection_metric": "f1_macro",
            }
        )
        + "\n",
    )

    _write(
        project_root / "artifacts/explainability/phase6/xai_summary.json",
        '{"logreg": {"n_samples": 5}}\n',
    )
    _write(
        project_root / "artifacts/explainability/phase6/logreg/shap_summary.png",
        "png\n",
    )
    _write(
        project_root / "artifacts/diagnostics/phase6/label_diagnostics_summary.json",
        '{"quality_score": 99.1, "near_duplicate_issues": 3, "outlier_issues": 1, "label_issues": 0}\n',
    )
    _write(
        project_root / "artifacts/diagnostics/phase6/label_diagnostics_report.md",
        "# report\n",
    )

    _write(
        project_root / "artifacts/analysis/phase3/artifact_index.json",
        json.dumps(
            {
                "phase": "phase3",
                "entries": [
                    {
                        "stage": "core_stats",
                        "source_data": "data/processed/evasionbench_prepared.parquet",
                        "generated_files": ["core_stats/class_distribution.png"],
                    }
                ],
            }
        )
        + "\n",
    )
    _write(
        project_root / "artifacts/analysis/phase4/artifact_index.json",
        json.dumps(
            {
                "phase": "phase4",
                "entries": [
                    {
                        "stage": "semantic_similarity",
                        "source_data": "data/processed/evasionbench_prepared.parquet",
                        "generated_files": [
                            "semantic_similarity/semantic_similarity_by_label.csv"
                        ],
                    }
                ],
            }
        )
        + "\n",
    )

    _write(project_root / "artifacts/reports/phase7/report.md", "# Report\n")
    _write(project_root / "artifacts/reports/phase7/report.html", "<html></html>\n")
    _write(project_root / "artifacts/reports/phase7/report.pdf", "%PDF\n")
    _write(
        project_root / "artifacts/reports/phase7/report_traceability.json",
        '{"items": {}}\n',
    )
    _write(
        project_root / "artifacts/reports/phase7/run_summary.json",
        '{"status": "passed", "stages": []}\n',
    )

    manifest = build_report_manifest(
        section_entries={
            "dataset": [
                make_provenance_entry(
                    stage="prepare_data",
                    script="scripts/prepare_data.py",
                    artifact_path="data/processed/evasionbench_prepared.parquet",
                    generated_at="2026-02-11T00:00:00Z",
                )
            ],
            "analyses": [
                make_provenance_entry(
                    stage="analyze_phase3",
                    script="scripts/run_phase3_analyses.py",
                    artifact_path="artifacts/analysis/phase3/core_stats/class_distribution.png",
                    generated_at="2026-02-11T00:00:00Z",
                ),
                make_provenance_entry(
                    stage="analyze_phase4",
                    script="scripts/run_phase4_analyses.py",
                    artifact_path="artifacts/analysis/phase4/semantic_similarity/semantic_similarity_by_label.csv",
                    generated_at="2026-02-11T00:00:00Z",
                ),
            ],
            "models": [
                make_provenance_entry(
                    stage="phase5_classical_baselines",
                    script="scripts/run_classical_baselines.py",
                    artifact_path="artifacts/models/phase5/run_summary.json",
                    generated_at="2026-02-11T00:00:00Z",
                ),
                make_provenance_entry(
                    stage="phase5_classical_baselines",
                    script="scripts/run_classical_baselines.py",
                    artifact_path="artifacts/models/phase5/logreg/model.pkl",
                    generated_at="2026-02-11T00:00:00Z",
                ),
                make_provenance_entry(
                    stage="phase6_transformer_baselines",
                    script="scripts/run_transformer_baselines.py",
                    artifact_path="artifacts/models/phase6/transformer/metrics.json",
                    generated_at="2026-02-11T00:00:00Z",
                ),
            ],
            "explainability": [
                make_provenance_entry(
                    stage="phase6_xai_classical",
                    script="scripts/run_explainability_analysis.py",
                    artifact_path="artifacts/explainability/phase6/xai_summary.json",
                    generated_at="2026-02-11T00:00:00Z",
                ),
                make_provenance_entry(
                    stage="phase6_xai_classical",
                    script="scripts/run_explainability_analysis.py",
                    artifact_path="artifacts/explainability/phase6/logreg/shap_summary.png",
                    generated_at="2026-02-11T00:00:00Z",
                ),
            ],
            "diagnostics": [
                make_provenance_entry(
                    stage="phase6_label_diagnostics",
                    script="scripts/run_label_diagnostics.py",
                    artifact_path="artifacts/diagnostics/phase6/label_diagnostics_summary.json",
                    generated_at="2026-02-11T00:00:00Z",
                ),
                make_provenance_entry(
                    stage="phase6_label_diagnostics",
                    script="scripts/run_label_diagnostics.py",
                    artifact_path="artifacts/diagnostics/phase6/label_diagnostics_report.md",
                    generated_at="2026-02-11T00:00:00Z",
                ),
            ],
        },
        generated_at="2026-02-11T00:00:00Z",
    )

    manifest_path = project_root / "artifacts/reports/phase7/provenance_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def test_pages_dataset_outputs_and_sanitizes_paths(tmp_path: Path) -> None:
    manifest_path = _fixture_project(tmp_path)
    publish_root = tmp_path / "artifacts/publish"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/build_pages_dataset.py",
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(publish_root),
            "--project-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

    site_data_path = publish_root / "data/site_data.json"
    publish_manifest_path = publish_root / "data/publication_manifest.json"
    kpi_summary_path = publish_root / "data/kpi_summary.json"
    site_quality_path = publish_root / "data/site_quality_report.json"

    assert site_data_path.exists()
    assert publish_manifest_path.exists()
    assert kpi_summary_path.exists()
    assert site_quality_path.exists()

    site_data = json.loads(site_data_path.read_text(encoding="utf-8"))
    publish_manifest = json.loads(publish_manifest_path.read_text(encoding="utf-8"))

    dump = json.dumps(site_data)
    assert str(tmp_path) not in dump

    source_paths = [asset["source_path"] for asset in publish_manifest["assets"]]
    assert "artifacts/models/phase5/logreg/model.pkl" not in source_paths

    assert site_data["summary"]["published_assets"] >= 10
    assert site_data["downloads"]["report_html"]
    assert site_data["kpi_summary"]["source"] == "phase8_selected_model"


def test_pages_dataset_allows_missing_optional_inputs(tmp_path: Path) -> None:
    manifest_path = _fixture_project(tmp_path)
    publish_root = tmp_path / "artifacts/publish"

    (tmp_path / "artifacts/analysis/phase3/artifact_index.json").unlink()
    (tmp_path / "artifacts/analysis/phase4/artifact_index.json").unlink()
    (tmp_path / "artifacts/models/phase5/model_comparison/summary.json").unlink()
    (tmp_path / "artifacts/diagnostics/phase6/label_diagnostics_summary.json").unlink()
    (tmp_path / "artifacts/explainability/phase6/xai_summary.json").unlink()

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/build_pages_dataset.py",
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(publish_root),
            "--project-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

    site_data = json.loads(
        (publish_root / "data/site_data.json").read_text(encoding="utf-8")
    )
    assert site_data["analysis_indexes"]["phase3"]["entries"] == []
    assert site_data["analysis_indexes"]["phase4"]["entries"] == []
    assert "best_model_family" in site_data
