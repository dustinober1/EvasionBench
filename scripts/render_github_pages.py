"""Render a professional, academic-style static website for GitHub Pages."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting import normalize_path

NAV_ITEMS = (
    ("index.html", "Home", "home"),
    ("methodology.html", "Methodology", "methodology"),
    ("findings.html", "Findings", "findings"),
    ("modeling.html", "Modeling", "modeling"),
    ("explainability.html", "Explainability", "explainability"),
    ("error-analysis.html", "Error Analysis", "error_analysis"),
    ("reproducibility.html", "Reproducibility", "reproducibility"),
)

SITE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --ink: #1c2b39;
  --muted: #566474;
  --paper: #f8f6f2;
  --card: #fffdf9;
  --line: #d6cec2;
  --accent: #8f3f1a;
  --accent-soft: #f4e8df;
  --navy: #1f3a5f;
  --teal: #136f63;
  --shadow: rgba(20, 30, 42, 0.08);
}

* {
  box-sizing: border-box;
}

html,
body {
  margin: 0;
  padding: 0;
}

body {
  color: var(--ink);
  background:
    radial-gradient(1200px 500px at 10% -10%, #e7eef6 0%, transparent 55%),
    radial-gradient(900px 300px at 95% -20%, #f5eee7 0%, transparent 55%),
    var(--paper);
  font-family: "Source Serif 4", Georgia, serif;
  line-height: 1.6;
}

a {
  color: var(--navy);
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

.shell {
  max-width: 1160px;
  margin: 0 auto;
  padding: 1.25rem;
}

.masthead {
  border: 1px solid var(--line);
  background: linear-gradient(170deg, #fffefc 0%, #fbf8f1 100%);
  box-shadow: 0 10px 24px var(--shadow);
  border-radius: 14px;
  padding: 1.3rem 1.4rem;
  margin-bottom: 1rem;
}

.masthead-top {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: baseline;
  flex-wrap: wrap;
}

.masthead h1 {
  margin: 0;
  font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
  font-size: 1.45rem;
  letter-spacing: 0.02em;
  color: var(--navy);
}

.masthead p {
  margin: 0.35rem 0 0;
  color: var(--muted);
  font-size: 0.98rem;
}

.meta {
  font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
  font-size: 0.8rem;
  color: var(--muted);
}

.nav {
  margin-top: 1rem;
  border-top: 1px solid var(--line);
  padding-top: 0.75rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.nav a {
  padding: 0.42rem 0.7rem;
  border-radius: 8px;
  font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
  font-size: 0.86rem;
  color: var(--ink);
  border: 1px solid transparent;
}

.nav a.active {
  background: var(--accent-soft);
  border-color: #e0c5b6;
  color: var(--accent);
  font-weight: 600;
}

.panel {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--card);
  box-shadow: 0 6px 20px var(--shadow);
  margin-bottom: 1rem;
  padding: 1rem 1.1rem;
}

h2,
h3 {
  font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
  color: var(--navy);
  line-height: 1.3;
  margin: 0 0 0.7rem;
}

h2 {
  font-size: 1.15rem;
}

h3 {
  font-size: 1rem;
}

.kicker {
  margin: 0 0 0.6rem;
  color: var(--muted);
  font-size: 0.95rem;
}

.grid {
  display: grid;
  gap: 0.9rem;
}

.grid.cards {
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
}

.card {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #fff;
  padding: 0.75rem 0.85rem;
}

.card .label {
  margin: 0;
  color: var(--muted);
  font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.card .value {
  margin: 0.25rem 0 0;
  color: var(--ink);
  font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
  font-size: 1.08rem;
  font-weight: 600;
}

.table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.93rem;
}

thead th {
  font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 0.75rem;
  color: var(--muted);
  background: #f6f2ec;
}

th,
td {
  border: 1px solid #ded7cb;
  padding: 0.46rem 0.52rem;
  text-align: left;
  vertical-align: top;
}

td code {
  font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
  font-size: 0.78rem;
  color: #4a3f36;
  background: #f8f3ed;
  padding: 0.08rem 0.24rem;
  border-radius: 4px;
}

.figure-grid {
  display: grid;
  gap: 0.9rem;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
}

figure {
  margin: 0;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
  padding: 0.55rem;
}

figure img {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 6px;
  border: 1px solid #e7e2d7;
  background: #fff;
}

figcaption {
  margin-top: 0.45rem;
  font-size: 0.86rem;
  color: var(--muted);
}

.caption-title {
  color: var(--ink);
  font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
  font-weight: 600;
  display: block;
}

.callout {
  border-left: 4px solid var(--teal);
  background: #eef8f5;
  padding: 0.66rem 0.75rem;
  border-radius: 8px;
}

.pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.pill {
  font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
  font-size: 0.8rem;
  padding: 0.24rem 0.48rem;
  background: #f2f5f8;
  border: 1px solid #d3dce4;
  border-radius: 999px;
  color: var(--ink);
}

.workflow {
  display: grid;
  gap: 0.6rem;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
}

.workflow-step {
  border: 1px solid #d9dfe8;
  border-radius: 10px;
  background: #f8fbff;
  padding: 0.55rem 0.6rem;
}

.workflow-step .step-title {
  margin: 0;
  font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
  font-size: 0.84rem;
  color: var(--navy);
}

.workflow-step .step-detail {
  margin: 0.22rem 0 0;
  color: var(--muted);
  font-size: 0.84rem;
}

.muted-list {
  margin: 0;
  padding-left: 1.1rem;
  color: var(--muted);
}

.muted-list li {
  margin: 0.24rem 0;
}

.note {
  margin-top: 0.32rem;
  color: var(--muted);
  font-size: 0.84rem;
}

.footer {
  color: var(--muted);
  font-size: 0.84rem;
  margin: 1.2rem 0 0.4rem;
  text-align: center;
}

@media (max-width: 700px) {
  .shell {
    padding: 0.8rem;
  }

  .masthead {
    padding: 1rem;
  }

  .panel {
    padding: 0.8rem;
  }
}
""".strip()


class PagesRenderError(RuntimeError):
    """Raised when pages rendering prerequisites are invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish-root",
        default="artifacts/publish",
        help="Publication root created by scripts/build_pages_dataset.py.",
    )
    parser.add_argument(
        "--site-data",
        default="artifacts/publish/data/site_data.json",
        help="Site data JSON generated by scripts/build_pages_dataset.py.",
    )
    parser.add_argument(
        "--publication-manifest",
        default="artifacts/publish/data/publication_manifest.json",
        help="Publication manifest JSON generated by scripts/build_pages_dataset.py.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/publish/site",
        help="Output directory for static website files.",
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _asset_lookup(publication_manifest: Mapping[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for asset in publication_manifest.get("assets", []):
        source_path = str(asset.get("source_path", ""))
        published_path = str(asset.get("published_path", ""))
        if source_path and published_path:
            lookup[source_path] = published_path
    return lookup


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _format_metric(value: Any) -> str:
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return f"{value:.3f}"
    return _esc(value)


def _render_nav(active_key: str) -> str:
    rows: list[str] = []
    for href, label, key in NAV_ITEMS:
        active = " active" if key == active_key else ""
        rows.append(
            f'<a class="nav-link{active}" href="{_esc(href)}">{_esc(label)}</a>'
        )
    return "\n".join(rows)


def _render_card_grid(items: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for item in items:
        cards.append(
            "\n".join(
                [
                    '<article class="card">',
                    f'<p class="label">{_esc(item.get("label", "Metric"))}</p>',
                    f'<p class="value">{_format_metric(item.get("value", "n/a"))}</p>',
                    "</article>",
                ]
            )
        )
    return '<div class="grid cards">' + "\n".join(cards) + "</div>"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    head_html = "".join(f"<th>{_esc(header)}</th>" for header in headers)
    row_html = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        row_html.append(f"<tr>{cells}</tr>")

    return (
        '<div class="table-wrap"><table>'
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody>"
        "</table></div>"
    )


def _page_shell(
    *,
    page_title: str,
    subtitle: str,
    active_nav: str,
    body_html: str,
    metadata: Mapping[str, Any],
) -> str:
    git_sha = _esc(metadata.get("git_sha", "unknown"))
    generated_at = _esc(metadata.get("generated_at", "unknown"))
    title = _esc(page_title)
    subtitle_text = _esc(subtitle)
    nav = _render_nav(active_nav)

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8" />\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        f"  <title>{title}</title>\n"
        '  <meta name="description" content="EvasionBench research findings portal" />\n'
        '  <link rel="stylesheet" href="static/site.css" />\n'
        "</head>\n"
        "<body>\n"
        '  <div class="shell">\n'
        '    <header class="masthead">\n'
        '      <div class="masthead-top">\n'
        "        <div>\n"
        "          <h1>EvasionBench Findings Portal</h1>\n"
        f"          <p>{subtitle_text}</p>\n"
        "        </div>\n"
        '        <div class="meta">\n'
        f"          <div>Git SHA: {git_sha}</div>\n"
        f"          <div>Published: {generated_at}</div>\n"
        "        </div>\n"
        "      </div>\n"
        '      <nav class="nav">\n'
        f"        {nav}\n"
        "      </nav>\n"
        "    </header>\n"
        f"    {body_html}\n"
        '    <p class="footer">Academic presentation generated from script-traceable artifacts.</p>\n'
        "  </div>\n"
        "</body>\n"
        "</html>\n"
    )


def _download_link(label: str, path: str | None) -> str:
    if not path:
        return f"<li>{_esc(label)}: unavailable</li>"
    return (
        f'<li><a href="{_esc(path)}" target="_blank" rel="noopener">'
        f"{_esc(label)}</a></li>"
    )


def _asset_for_suffix(asset_lookup: Mapping[str, str], suffix: str) -> str | None:
    for source_path, published_path in asset_lookup.items():
        if source_path.endswith(suffix):
            return published_path
    return None


def _traceability_ids_for_path(
    traceability: Mapping[str, Any], artifact_path: str
) -> list[str]:
    matches: list[str] = []
    for trace_id, payload in traceability.items():
        if str(payload.get("artifact_path", "")) == artifact_path:
            matches.append(str(trace_id))
    return sorted(matches)


def _figure_interpretation(suffix: str) -> str:
    if suffix.endswith("class_distribution.png"):
        return (
            "Label distribution is materially imbalanced, which motivates macro-F1 "
            "and class-level diagnostics over raw accuracy alone."
        )
    if suffix.endswith("question_length_by_label.png"):
        return (
            "Question length varies by label, indicating analysts phrase prompts "
            "differently before evasive versus direct responses."
        )
    if suffix.endswith("evasive_marker_rate.png"):
        return (
            "Linguistic markers linked to evasiveness are more frequent in evasive "
            "classes, supporting validity of text-derived behavioral features."
        )
    if suffix.endswith("semantic_similarity_by_label.png"):
        return (
            "Lower question-answer semantic alignment in evasive classes indicates "
            "deflection rather than direct response to analyst intent."
        )
    if suffix.endswith("topic_prevalence_by_label.png"):
        return (
            "Topic prevalence shifts by class, suggesting some thematic contexts "
            "are associated with higher rates of evasive responses."
        )
    if suffix.endswith("question_behavior_by_label.png"):
        return (
            "Behavior metrics vary by question type, revealing intent-dependent "
            "risk of evasive outcomes."
        )
    if suffix.endswith("macro_f1_by_model.png"):
        return (
            "Classical model families show a measurable spread in macro-F1, with the "
            "selected family balancing class coverage under holdout constraints."
        )
    if suffix.endswith("per_class_f1_delta_heatmap.png"):
        return (
            "Per-class deltas concentrate the remaining error budget into specific "
            "labels, motivating targeted error-analysis rather than generic tuning."
        )
    return "Interpretation generated from the linked analysis artifact."


def _workflow_figure() -> str:
    steps = [
        ("Phase 3", "Core, lexical, and linguistic statistics"),
        ("Phase 4", "Semantic, topic, and question-intent behavior"),
        ("Phase 5-6", "Modeling, explainability, diagnostics"),
        ("Phase 7", "Manifest + traceability report generation"),
        ("Phase 9", "Error profiling and conditional exploration"),
    ]
    blocks: list[str] = []
    for title, detail in steps:
        blocks.append(
            "\n".join(
                [
                    '<div class="workflow-step">',
                    f'<p class="step-title">{_esc(title)}</p>',
                    f'<p class="step-detail">{_esc(detail)}</p>',
                    "</div>",
                ]
            )
        )
    return '<div class="workflow">' + "\n".join(blocks) + "</div>"


def _render_index(
    *,
    site_data: Mapping[str, Any],
    asset_lookup: Mapping[str, str],
) -> str:
    summary = dict(site_data.get("summary", {}))
    key_findings = list(site_data.get("key_findings", []))
    downloads = dict(site_data.get("downloads", {}))
    kpi_summary = dict(site_data.get("kpi_summary", {}))
    transformer = dict(site_data.get("transformer_metrics", {}))

    primary_results_rows = [
        [
            "Selected model family",
            f"<code>{_esc(kpi_summary.get('model_family', 'n/a'))}</code>",
        ],
        ["Holdout accuracy", _format_metric(kpi_summary.get("accuracy", "n/a"))],
        ["Holdout macro F1", _format_metric(kpi_summary.get("f1_macro", "n/a"))],
        [
            "Holdout precision (macro)",
            _format_metric(kpi_summary.get("precision_macro", "n/a")),
        ],
        [
            "Holdout recall (macro)",
            _format_metric(kpi_summary.get("recall_macro", "n/a")),
        ],
        [
            "Transformer macro F1 (phase 6)",
            _format_metric(transformer.get("f1_macro", "n/a")),
        ],
    ]

    scope_rows = [
        [
            "Manifest version",
            f"<code>{_esc(summary.get('manifest_version', 'n/a'))}</code>",
        ],
        [
            "Published assets",
            _esc(
                f"{summary.get('published_assets', 'n/a')} "
                f"({summary.get('published_size_mb', 'n/a')} MB)"
            ),
        ],
        ["Phase-9 error artifacts", _esc(summary.get("phase9_error_artifacts", "0"))],
        [
            "Phase-9 exploration artifacts",
            _esc(summary.get("phase9_exploration_artifacts", "0")),
        ],
    ]

    abstract_lines = [
        "EvasionBench studies evasive behavior in earnings-call responses with a fully script-traceable workflow.",
        "The publication site is generated from manifest-linked artifacts spanning analysis, modeling, and diagnostics.",
        "Phase-9 extends prior phases with class-level error profiling and conditional exploratory slices.",
        "Primary KPIs are sourced through a deterministic priority chain and validated at publish time.",
        "This reduces narrative drift, removes placeholder metrics, and strengthens reviewer trust in reported results.",
        "All major figures and tables are linked to provenance or traceability artifacts for independent verification.",
    ]

    contributions = [
        "Deterministic KPI contract with fail-fast quality gates and site quality reporting.",
        "Error-analysis artifacts highlighting top confusion routes, hardest classes, and representative failures.",
        "Conditional exploratory outputs that document both generated slices and explicit prerequisite-based skips.",
    ]

    chart_path = _asset_for_suffix(
        asset_lookup, "model_comparison/macro_f1_by_model.png"
    )
    primary_figure = ""
    if chart_path:
        primary_figure = (
            '<div class="figure-grid">'
            "<figure>"
            f'<img src="{_esc(chart_path)}" alt="Macro F1 by model family" />'
            '<figcaption><span class="caption-title">Primary result figure.</span>'
            " Macro-F1 comparison across model families shows the selected baseline in context."
            "</figcaption></figure></div>"
        )

    body = [
        '<section class="panel">',
        "<h2>Abstract</h2>",
        '<p class="kicker">'
        + "<br/>".join(_esc(line) for line in abstract_lines)
        + "</p>",
        "</section>",
        '<section class="panel">',
        "<h2>Key Contributions</h2>",
        "<ul>",
        "\n".join(f"<li>{_esc(line)}</li>" for line in contributions),
        "</ul>",
        _render_card_grid(key_findings),
        "</section>",
        '<section class="panel">',
        "<h2>Primary Results Table</h2>",
        _render_table(["Metric", "Value"], primary_results_rows),
        primary_figure,
        "</section>",
        '<section class="panel">',
        "<h2>Practical Implications and Limitations</h2>",
        '<div class="grid" style="grid-template-columns: 1.2fr 1fr;">',
        '<div class="callout">'
        "Class-level performance indicates deployment should emphasize review workflows "
        "around classes with high confusion concentration, especially intermediate vs fully evasive."
        "</div>",
        '<div class="card"><p class="label">Limitations Snapshot</p>'
        '<ul class="muted-list">'
        "<li>Class imbalance persists and can mask minority failures if only accuracy is reported.</li>"
        "<li>Temporal and segment exploratory slices depend on source schema availability.</li>"
        "<li>Error evidence quality is bounded by available row-level prediction artifacts.</li>"
        "</ul></div>",
        "</div>",
        "<h3>Dataset and Report Access</h3>",
        '<div class="grid" style="grid-template-columns: 1.2fr 1fr;">',
        _render_table(["Item", "Value"], scope_rows),
        '<div class="card"><p class="label">Downloads</p><ul>',
        _download_link("Report (HTML)", downloads.get("report_html")),
        _download_link("Report (PDF)", downloads.get("report_pdf")),
        _download_link("Report (Markdown)", downloads.get("report_markdown")),
        _download_link("Traceability JSON", downloads.get("traceability")),
        _download_link("Provenance Manifest", downloads.get("provenance_manifest")),
        "</ul></div>",
        "</div>",
        "</section>",
    ]
    return "\n".join(line for line in body if line)


def _render_methodology(site_data: Mapping[str, Any]) -> str:
    phase3_entries = (
        site_data.get("analysis_indexes", {}).get("phase3", {}).get("entries", [])
    )
    phase4_entries = (
        site_data.get("analysis_indexes", {}).get("phase4", {}).get("entries", [])
    )

    phase_rows = [
        [
            "Phase 3",
            _esc(len(phase3_entries)),
            "Core stats, lexical, linguistic quality",
        ],
        [
            "Phase 4",
            _esc(len(phase4_entries)),
            "Semantic similarity, topic modeling, question behavior",
        ],
        ["Phase 5", "3", "Classical baselines and model comparison"],
        ["Phase 6", "2", "Transformer baseline, explainability, diagnostics"],
        ["Phase 7", "1", "Manifest-driven report generation and traceability"],
        [
            "Phase 9",
            _esc(site_data.get("summary", {}).get("phase9_error_artifacts", 0)),
            "Error profiling and exploration slices",
        ],
    ]

    stage_rows: list[list[str]] = []
    for entry in [*phase3_entries, *phase4_entries]:
        stage_rows.append(
            [
                f"<code>{_esc(entry.get('stage', 'unknown'))}</code>",
                _esc(len(entry.get("generated_files", []))),
                f"<code>{_esc(entry.get('source_data', 'n/a'))}</code>",
            ]
        )

    kpi_summary = dict(site_data.get("kpi_summary", {}))
    evaluation_protocol = _esc(kpi_summary.get("evaluation_protocol", "n/a"))
    assumptions = [
        "Prepared data labels are treated as reference truth for supervised evaluation.",
        "Question + answer text concatenation with [SEP] remains the canonical feature interface.",
        "Publish-time KPI values are read from canonical JSON, not hard-coded in templates.",
    ]
    validity_notes = [
        "Class imbalance and label ambiguity can suppress minority recall even at stable overall accuracy.",
        "Exploratory temporal/segment analyses are conditional and explicitly skipped when fields are absent.",
        "Error-route conclusions are specific to the selected model and evaluation split recorded in artifacts.",
    ]

    return "\n".join(
        [
            '<section class="panel">',
            "<h2>Study Design</h2>",
            '<p class="kicker">The publication workflow is script-first and contract-driven; '
            "all metrics and visuals are generated from artifacted outputs with provenance links.</p>",
            _workflow_figure(),
            '<p class="note">Evaluation protocol: <code>'
            + evaluation_protocol
            + "</code></p>",
            "</section>",
            '<section class="panel">',
            "<h2>Phase Topology</h2>",
            _render_table(["Phase", "Tracked units", "Scope"], phase_rows),
            "</section>",
            '<section class="panel">',
            "<h2>Artifact Production Stages</h2>",
            _render_table(["Stage", "Generated files", "Source"], stage_rows),
            "</section>",
            '<section class="panel">',
            "<h2>Assumptions and Validity Notes</h2>",
            "<h3>Assumptions</h3>",
            '<ul class="muted-list">',
            "\n".join(f"<li>{_esc(line)}</li>" for line in assumptions),
            "</ul>",
            "<h3>Validity Notes</h3>",
            '<ul class="muted-list">',
            "\n".join(f"<li>{_esc(line)}</li>" for line in validity_notes),
            "</ul>",
            "</section>",
        ]
    )


def _render_findings(site_data: Mapping[str, Any]) -> str:
    featured = list(site_data.get("featured_figures", []))
    analysis_entries = list(site_data.get("artifacts", {}).get("analyses", []))
    traceability = dict(site_data.get("traceability", {}))

    figure_blocks: list[str] = []
    for index, item in enumerate(featured, start=1):
        artifact_path = str(item.get("artifact_path", ""))
        trace_ids = _traceability_ids_for_path(traceability, artifact_path)
        trace_text = (
            ", ".join(f"<code>{_esc(trace_id)}</code>" for trace_id in trace_ids)
            if trace_ids
            else "<code>n/a</code>"
        )
        interpretation = _figure_interpretation(artifact_path)
        figure_blocks.append(
            "\n".join(
                [
                    "<figure>",
                    f'<img src="{_esc(item.get("published_path", ""))}" alt="{_esc(item.get("id", "figure"))}" />',
                    "<figcaption>",
                    f'<span class="caption-title">Figure {index}.</span> '
                    f"{_esc(item.get('caption', 'Published analysis figure'))} "
                    f"<code>{_esc(item.get('stage', 'unknown'))}</code>",
                    f"<br/>Interpretation: {_esc(interpretation)}",
                    f'<br/>Artifact link: <a href="{_esc(item.get("published_path", ""))}" target="_blank" rel="noopener">open artifact</a>',
                    f"<br/>Traceability IDs: {trace_text}",
                    "</figcaption>",
                    "</figure>",
                ]
            )
        )

    analysis_table_rows: list[list[str]] = []
    for entry in analysis_entries[:24]:
        label = f"<code>{_esc(entry.get('id', 'artifact'))}</code>"
        link = (
            f'<a href="{_esc(entry.get("published_path", ""))}" '
            'target="_blank" rel="noopener">open</a>'
        )
        analysis_table_rows.append(
            [
                label,
                f"<code>{_esc(entry.get('stage', 'unknown'))}</code>",
                _esc(entry.get("kind", "artifact")),
                link,
            ]
        )

    return "\n".join(
        [
            '<section class="panel">',
            "<h2>Phase-3 and Phase-4 Findings</h2>",
            '<p class="kicker">Key figures below summarize lexical, linguistic, semantic, '
            "and question-behavior trends across direct, intermediate, and fully evasive "
            "responses. Each visual includes a short interpretation and traceability pointer.</p>",
            '<div class="figure-grid">',
            "\n".join(figure_blocks),
            "</div>",
            "</section>",
            '<section class="panel">',
            "<h2>Published Analysis Artifacts</h2>",
            _render_table(
                ["Artifact ID", "Stage", "Kind", "Access"], analysis_table_rows
            ),
            "</section>",
        ]
    )


def _render_modeling(
    *,
    site_data: Mapping[str, Any],
    asset_lookup: Mapping[str, str],
) -> str:
    model_comparison = dict(site_data.get("model_comparison", {}))
    models = dict(model_comparison.get("models", {}))
    best_family = str(site_data.get("best_model_family", "unknown"))
    best_metrics = dict(site_data.get("best_model_metrics", {}))
    transformer_metrics = dict(site_data.get("transformer_metrics", {}))

    model_rows: list[list[str]] = []
    for family in sorted(models.keys()):
        metrics = dict(models.get(family, {}))
        marker = "*" if family == best_family else ""
        model_rows.append(
            [
                f"<code>{_esc(family)}</code>{marker}",
                _format_metric(metrics.get("accuracy", "n/a")),
                _format_metric(metrics.get("f1_macro", "n/a")),
                _format_metric(metrics.get("precision_macro", "n/a")),
                _format_metric(metrics.get("recall_macro", "n/a")),
            ]
        )

    summary_cards = [
        {"label": "Selected Classical Family", "value": best_family},
        {
            "label": "Selected Classical Macro F1",
            "value": _format_metric(best_metrics.get("f1_macro", "n/a")),
        },
        {
            "label": "Transformer Macro F1",
            "value": _format_metric(transformer_metrics.get("f1_macro", "n/a")),
        },
    ]

    chart_paths = [
        _asset_for_suffix(asset_lookup, "model_comparison/macro_f1_by_model.png"),
        _asset_for_suffix(
            asset_lookup, "model_comparison/per_class_f1_delta_heatmap.png"
        ),
    ]

    figure_rows: list[str] = []
    for chart in [path for path in chart_paths if path]:
        figure_rows.append(
            "\n".join(
                [
                    "<figure>",
                    f'<img src="{_esc(chart)}" alt="Model comparison chart" />',
                    '<figcaption><span class="caption-title">Model comparison visual.</span>'
                    " Published from phase-5 model comparison outputs.</figcaption>",
                    "</figure>",
                ]
            )
        )

    return "\n".join(
        [
            '<section class="panel">',
            "<h2>Model Selection Results</h2>",
            _render_card_grid(summary_cards),
            '<p class="kicker">* denotes the selected classical model family in the '
            "ranking table.</p>",
            "</section>",
            '<section class="panel">',
            "<h2>Classical Benchmark Table</h2>",
            _render_table(
                ["Model", "Accuracy", "Macro F1", "Precision", "Recall"], model_rows
            ),
            "</section>",
            '<section class="panel">',
            "<h2>Comparison Figures</h2>",
            '<div class="figure-grid">',
            "\n".join(figure_rows),
            "</div>",
            "</section>",
        ]
    )


def _render_explainability(
    *,
    site_data: Mapping[str, Any],
    asset_lookup: Mapping[str, str],
) -> str:
    xai_summary = dict(site_data.get("xai_summary", {}))
    diagnostics_summary = dict(site_data.get("diagnostics_summary", {}))
    diagnostics_entries = list(site_data.get("artifacts", {}).get("diagnostics", []))

    xai_rows: list[list[str]] = []
    for family in sorted(xai_summary.keys()):
        details = xai_summary.get(family, {})
        xai_rows.append(
            [
                f"<code>{_esc(family)}</code>",
                _esc(json.dumps(details, sort_keys=True)),
            ]
        )

    diagnostics_rows = [
        [
            "Quality score",
            _format_metric(diagnostics_summary.get("quality_score", "n/a")),
        ],
        [
            "Near-duplicate issues",
            _esc(diagnostics_summary.get("near_duplicate_issues", "n/a")),
        ],
        ["Outlier issues", _esc(diagnostics_summary.get("outlier_issues", "n/a"))],
        ["Label issues", _esc(diagnostics_summary.get("label_issues", "n/a"))],
    ]

    shap_paths: list[str] = []
    for source_path, published in asset_lookup.items():
        if source_path.endswith("shap_summary.png"):
            shap_paths.append(published)

    shap_paths.sort()
    shap_figures = [
        "\n".join(
            [
                "<figure>",
                f'<img src="{_esc(path)}" alt="SHAP summary" />',
                '<figcaption><span class="caption-title">SHAP summary.</span>'
                " Family-level attribution plot from phase-6 explainability outputs.</figcaption>",
                "</figure>",
            ]
        )
        for path in shap_paths
    ]

    diagnostics_links: list[str] = []
    for entry in diagnostics_entries:
        diagnostics_links.append(
            f'<li><a href="{_esc(entry.get("published_path", ""))}" '
            f'target="_blank" rel="noopener">{_esc(entry.get("artifact_path", "artifact"))}</a></li>'
        )

    return "\n".join(
        [
            '<section class="panel">',
            "<h2>Explainability and Diagnostics</h2>",
            '<p class="kicker">Model behavior interpretation is based on SHAP for classical '
            "models and transformer explainability outputs, supplemented with label-quality "
            "diagnostics.</p>",
            "</section>",
            '<section class="panel">',
            "<h2>XAI Summary by Family</h2>",
            _render_table(["Family", "Summary"], xai_rows),
            "</section>",
            '<section class="panel">',
            "<h2>SHAP Figures</h2>",
            '<div class="figure-grid">',
            "\n".join(shap_figures),
            "</div>",
            "</section>",
            '<section class="panel">',
            "<h2>Label Quality Snapshot</h2>",
            _render_table(["Metric", "Value"], diagnostics_rows),
            "<h3>Diagnostic Artifacts</h3>",
            "<ul>",
            "\n".join(diagnostics_links),
            "</ul>",
            "</section>",
        ]
    )


def _render_error_analysis(site_data: Mapping[str, Any]) -> str:
    phase9 = dict(site_data.get("phase9", {}))
    error_payload = dict(phase9.get("error_analysis", {}))
    exploration_payload = dict(phase9.get("exploration", {}))

    if error_payload.get("status") != "available":
        return "\n".join(
            [
                '<section class="panel">',
                "<h2>Error Analysis</h2>",
                '<p class="kicker">Phase-9 error-analysis artifacts are not available in the current publish bundle.</p>',
                "</section>",
            ]
        )

    files = dict(error_payload.get("files", {}))
    error_summary = dict(error_payload.get("error_summary", {}))
    top_routes = list(error_summary.get("top_misclassification_routes", []))[:3]
    route_rows: list[list[str]] = []
    for route in top_routes:
        route_rows.append(
            [
                f"<code>{_esc(route.get('true_label', 'n/a'))}</code>",
                f"<code>{_esc(route.get('predicted_label', 'n/a'))}</code>",
                _esc(route.get("count", "0")),
                _format_metric(route.get("true_label_route_share", 0.0)),
            ]
        )
    if not route_rows:
        route_rows.append(["n/a", "n/a", "0", "0.000"])

    implications: list[str] = []
    for route in top_routes:
        implications.append(
            "Prioritize interventions for "
            f"{route.get('true_label', 'n/a')} -> {route.get('predicted_label', 'n/a')} "
            "because this route dominates observed misclassifications."
        )
    if not implications:
        implications.append(
            "No dominant routes were detected in the current bundle; review raw route CSV for low-count patterns."
        )

    heatmap = files.get("class_failure_heatmap.png")
    route_csv = files.get("misclassification_routes.csv")
    hard_cases = files.get("hard_cases.md")
    error_summary_link = files.get("error_summary.json")
    index_link = files.get("artifact_index.json")

    exploration_files = dict(exploration_payload.get("files", {}))
    temporal = dict(exploration_payload.get("temporal_summary", {}))
    segment = dict(exploration_payload.get("segment_summary", {}))
    question_intent_path = exploration_files.get("question_intent_error_map.csv")

    exploration_rows = [
        ["Temporal slice", _esc(temporal.get("status", "missing"))],
        ["Segment slice", _esc(segment.get("status", "missing"))],
        [
            "Question-intent error map",
            (
                f'<a href="{_esc(question_intent_path)}" target="_blank" rel="noopener">open CSV</a>'
                if question_intent_path
                else "unavailable"
            ),
        ],
    ]

    heatmap_block = ""
    if heatmap:
        heatmap_block = (
            '<div class="figure-grid"><figure>'
            f'<img src="{_esc(heatmap)}" alt="Class failure heatmap" />'
            '<figcaption><span class="caption-title">Class-level confusion heatmap.</span>'
            " Concentrated off-diagonal mass indicates targeted failure routes for the next iteration."
            "</figcaption></figure></div>"
        )

    return "\n".join(
        [
            '<section class="panel">',
            "<h2>Error Analysis</h2>",
            '<p class="kicker">This section summarizes class-level failure concentration for the selected model and surfaces representative hard cases.</p>',
            _render_table(
                ["True label", "Predicted label", "Count", "Route share"], route_rows
            ),
            "</section>",
            '<section class="panel">',
            "<h2>Class Failure Heatmap</h2>",
            heatmap_block,
            "<h3>Implications for Next Iteration</h3>",
            '<ul class="muted-list">',
            "\n".join(f"<li>{_esc(line)}</li>" for line in implications),
            "</ul>",
            "<h3>Artifacts</h3>",
            "<ul>",
            _download_link("Error summary JSON", error_summary_link),
            _download_link("Misclassification routes CSV", route_csv),
            _download_link("Hard cases markdown", hard_cases),
            _download_link("Error-analysis index", index_link),
            "</ul>",
            "</section>",
            '<section class="panel">',
            "<h2>Exploratory Slice Status</h2>",
            _render_table(["Slice", "Status / Access"], exploration_rows),
            '<p class="note">Temporal and segment slices are generated only when prerequisite columns exist; skipped status is recorded explicitly.</p>',
            "</section>",
        ]
    )


def _render_reproducibility(site_data: Mapping[str, Any]) -> str:
    metadata = dict(site_data.get("metadata", {}))
    downloads = dict(site_data.get("downloads", {}))
    traceability = dict(site_data.get("traceability", {}))
    pipeline_status = dict(site_data.get("pipeline_status", {}))
    run_status = dict(site_data.get("run_status_summary", {}))

    reproducibility_rows = [
        ["Generated at", f"<code>{_esc(metadata.get('generated_at', 'n/a'))}</code>"],
        ["Generated by", f"<code>{_esc(metadata.get('generated_by', 'n/a'))}</code>"],
        ["Git SHA", f"<code>{_esc(metadata.get('git_sha', 'n/a'))}</code>"],
        [
            "Latest successful run",
            f"<code>{_esc(run_status.get('latest_successful_run_timestamp', 'n/a'))}</code>",
        ],
        [
            "Latest attempted status",
            f"<code>{_esc(run_status.get('latest_attempted_status', 'n/a'))}</code>",
        ],
        [
            "Latest attempted timestamp",
            f"<code>{_esc(run_status.get('latest_attempted_timestamp', 'n/a'))}</code>",
        ],
    ]

    pipeline_rows: list[list[str]] = []
    for stage in pipeline_status.get("stages", []):
        log_path = str(stage.get("log", "n/a"))
        pipeline_rows.append(
            [
                f"<code>{_esc(stage.get('stage', 'unknown'))}</code>",
                _esc(stage.get("status", "unknown")),
                f"<code>{_esc(log_path)}</code>",
            ]
        )

    if not pipeline_rows:
        pipeline_rows.append(["n/a", "No run summary available", "n/a"])

    trace_rows: list[list[str]] = []
    for trace_id, trace in sorted(traceability.items())[:160]:
        trace_rows.append(
            [
                f"<code>{_esc(trace_id)}</code>",
                f"<code>{_esc(trace.get('stage', 'unknown'))}</code>",
                f"<code>{_esc(trace.get('script', 'unknown'))}</code>",
                f"<code>{_esc(trace.get('artifact_path', 'unknown'))}</code>",
            ]
        )

    download_list = [
        _download_link("Report (HTML)", downloads.get("report_html")),
        _download_link("Report (PDF)", downloads.get("report_pdf")),
        _download_link("Traceability JSON", downloads.get("traceability")),
        _download_link("Provenance Manifest", downloads.get("provenance_manifest")),
        _download_link("Run Summary", downloads.get("run_summary")),
    ]
    failure_log_link = run_status.get("latest_failure_log_published_path")
    if not failure_log_link:
        failure_log_link = run_status.get("latest_failure_log_path")

    run_log_links = []
    for item in run_status.get("published_log_artifacts", []):
        run_log_links.append(
            f'<li><a href="{_esc(item)}" target="_blank" rel="noopener">{_esc(item)}</a></li>'
        )
    if not run_log_links:
        run_log_links.append("<li>No published run logs recorded.</li>")

    return "\n".join(
        [
            '<section class="panel">',
            "<h2>Reproducibility Metadata</h2>",
            _render_table(["Field", "Value"], reproducibility_rows),
            '<div class="card">',
            '<p class="label">Reproducibility Downloads</p>',
            "<ul>",
            "\n".join(download_list),
            "</ul>",
            "</div>",
            "</section>",
            '<section class="panel">',
            "<h2>Run Status Hardening</h2>",
            _render_table(["Field", "Value"], reproducibility_rows),
            '<div class="callout">Latest successful run and latest attempted run status are surfaced from phase-7 run metadata and linked logs.</div>',
            "<h3>Failure Log</h3>",
            (
                f'<p><a href="{_esc(failure_log_link)}" target="_blank" rel="noopener">{_esc(failure_log_link)}</a></p>'
                if failure_log_link
                else "<p>No failure log recorded.</p>"
            ),
            "<h3>Published Run Logs</h3>",
            "<ul>",
            "\n".join(run_log_links),
            "</ul>",
            "</section>",
            '<section class="panel">',
            "<h2>Pipeline Status</h2>",
            _render_table(["Stage", "Status", "Log"], pipeline_rows),
            "</section>",
            '<section class="panel">',
            "<h2>Traceability Matrix (Sample)</h2>",
            '<p class="kicker">Showing up to 160 rows. Full lineage is available in the '
            "downloaded traceability JSON artifact.</p>",
            _render_table(
                ["Traceability ID", "Stage", "Script", "Artifact"], trace_rows
            ),
            "</section>",
        ]
    )


def render_site(args: argparse.Namespace) -> Path:
    project_root = Path(args.project_root).resolve()
    publish_root = _resolve(args.publish_root, project_root=project_root)
    site_data_path = _resolve(args.site_data, project_root=project_root)
    publication_manifest_path = _resolve(
        args.publication_manifest, project_root=project_root
    )
    output_root = _resolve(args.output_root, project_root=project_root)

    if not site_data_path.exists():
        raise PagesRenderError(f"Site data not found: {site_data_path}")
    if not publication_manifest_path.exists():
        raise PagesRenderError(
            f"Publication manifest not found: {publication_manifest_path}"
        )

    site_data = _load_json(site_data_path)
    publication_manifest = _load_json(publication_manifest_path)
    asset_lookup = _asset_lookup(publication_manifest)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "static").mkdir(parents=True, exist_ok=True)

    css_path = output_root / "static" / "site.css"
    css_path.write_text(SITE_CSS + "\n", encoding="utf-8")

    page_definitions = [
        (
            "index.html",
            "Home",
            "home",
            _render_index(site_data=site_data, asset_lookup=asset_lookup),
        ),
        (
            "methodology.html",
            "Methodology",
            "methodology",
            _render_methodology(site_data),
        ),
        (
            "findings.html",
            "Findings",
            "findings",
            _render_findings(site_data),
        ),
        (
            "modeling.html",
            "Modeling",
            "modeling",
            _render_modeling(site_data=site_data, asset_lookup=asset_lookup),
        ),
        (
            "explainability.html",
            "Explainability",
            "explainability",
            _render_explainability(site_data=site_data, asset_lookup=asset_lookup),
        ),
        (
            "error-analysis.html",
            "Error Analysis",
            "error_analysis",
            _render_error_analysis(site_data),
        ),
        (
            "reproducibility.html",
            "Reproducibility",
            "reproducibility",
            _render_reproducibility(site_data),
        ),
    ]

    for file_name, title, nav_key, body in page_definitions:
        html_doc = _page_shell(
            page_title=f"EvasionBench | {title}",
            subtitle="Earnings-call evasion detection: report, findings, and reproducibility",
            active_nav=nav_key,
            body_html=body,
            metadata=site_data.get("metadata", {}),
        )
        (output_root / file_name).write_text(html_doc, encoding="utf-8")

    # GitHub Pages serves static sites more reliably without Jekyll processing.
    (output_root / ".nojekyll").write_text("\n", encoding="utf-8")

    # Ensure published assets are available under site output.
    assets_source = publish_root / "assets"
    assets_target = output_root / "assets"
    if assets_source.exists():
        if assets_target.exists():
            shutil.rmtree(assets_target)
        shutil.copytree(assets_source, assets_target)

    print(f"rendered site: {normalize_path(output_root, base=project_root)}")
    print("pages: " + ", ".join(file_name for file_name, _, _, _ in page_definitions))

    return output_root


def main() -> int:
    args = parse_args()
    try:
        render_site(args)
    except PagesRenderError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
