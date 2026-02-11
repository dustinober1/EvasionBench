# Reporting Pipeline (Phase 7)

Phase 7 produces report outputs from one canonical markdown source.

## Canonical Commands

- Build provenance manifest:
  - `python3 scripts/build_report_manifest.py --output artifacts/reports/phase7/provenance_manifest.json`
- Generate markdown report:
  - `python3 scripts/generate_research_report.py --manifest artifacts/reports/phase7/provenance_manifest.json --output artifacts/reports/phase7/report.md`
- Render HTML/PDF + traceability:
  - `python3 scripts/render_research_report.py --input artifacts/reports/phase7/report.md --manifest artifacts/reports/phase7/provenance_manifest.json --output-root artifacts/reports/phase7`
- Run full orchestrated pipeline:
  - `python3 scripts/run_research_pipeline.py --output-root artifacts/reports/phase7 --skip-existing`

## Output Contract

Expected files under `artifacts/reports/phase7/`:

- `provenance_manifest.json`
- `run_summary.json`
- `report.md`
- `report.html`
- `report.pdf`
- `report_traceability.json`
- `logs/*.log`

## Traceability

`report_traceability.json` maps each figure/table id to source provenance fields:

- `stage`
- `script`
- `artifact_path`
- `generated_at`

## Dependency Notes

Renderer behavior:

- Uses `markdown` for markdown-to-HTML when available.
- Uses `weasyprint` for HTML-to-PDF when available.
- If either dependency is missing, the renderer prints a warning with install guidance and uses a built-in fallback so outputs are still produced.

Install/reinstall dependencies with:

- `pip install -r requirements.txt`

## Troubleshooting

- Manifest build fails with missing artifact paths:
  - Run the stage script listed in the error message, then rerun manifest build.
- Pipeline stage fails:
  - Inspect the corresponding file in `artifacts/reports/phase7/logs/`.
- PDF output looks degraded:
  - Install `weasyprint` and rerun the render stage.
