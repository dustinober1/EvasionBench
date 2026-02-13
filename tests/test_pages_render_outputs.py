from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from test_pages_dataset_contract import ROOT, _fixture_project


def test_pages_renderer_outputs_required_pages(tmp_path: Path) -> None:
    manifest_path = _fixture_project(tmp_path)
    publish_root = tmp_path / "artifacts/publish"
    site_root = publish_root / "site"

    build_proc = subprocess.run(
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
    assert build_proc.returncode == 0, build_proc.stderr

    render_proc = subprocess.run(
        [
            sys.executable,
            "scripts/render_github_pages.py",
            "--publish-root",
            str(publish_root),
            "--site-data",
            str(publish_root / "data/site_data.json"),
            "--publication-manifest",
            str(publish_root / "data/publication_manifest.json"),
            "--output-root",
            str(site_root),
            "--project-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert render_proc.returncode == 0, render_proc.stderr

    expected_files = [
        site_root / "index.html",
        site_root / "methodology.html",
        site_root / "findings.html",
        site_root / "modeling.html",
        site_root / "explainability.html",
        site_root / "reproducibility.html",
        site_root / "static/site.css",
        site_root / ".nojekyll",
    ]
    for path in expected_files:
        assert path.exists(), str(path)

    index_html = (site_root / "index.html").read_text(encoding="utf-8")
    methodology_html = (site_root / "methodology.html").read_text(encoding="utf-8")
    modeling_html = (site_root / "modeling.html").read_text(encoding="utf-8")

    assert "EvasionBench Findings Portal" in index_html
    assert 'href="methodology.html"' in index_html
    assert "Methodological Design" in methodology_html
    assert "Classical Benchmark Table" in modeling_html
