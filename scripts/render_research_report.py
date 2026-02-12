"""Render phase-7 markdown research report into HTML/PDF plus traceability map."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting import (
    build_traceability_map,
    load_report_manifest,
    normalize_path,
    utc_now_iso,
)

BASE_CSS = """
body {
  font-family: "Georgia", "Times New Roman", serif;
  color: #20222a;
  line-height: 1.55;
  margin: 2.2rem auto;
  max-width: 900px;
  padding: 0 1.5rem;
}
h1, h2, h3 {
  color: #1f2d3d;
}
code {
  background: #f3f4f8;
  border-radius: 4px;
  padding: 0.1rem 0.3rem;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1rem 0;
}
th, td {
  border: 1px solid #d5d8e0;
  padding: 0.45rem;
  text-align: left;
  font-size: 0.95rem;
}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="artifacts/reports/phase7/report.md",
        help="Input markdown report path.",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/reports/phase7/provenance_manifest.json",
        help="Manifest path used for traceability mapping.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/reports/phase7",
        help="Output directory for HTML/PDF/traceability artifacts.",
    )
    parser.add_argument(
        "--project-root",
        default=str(ROOT),
        help="Project root for resolving relative paths.",
    )
    return parser.parse_args()


def _resolve(path_like: str, *, project_root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root / path


def _markdown_to_html(markdown_text: str) -> tuple[str, list[str]]:
    warnings: list[str] = []

    try:
        import markdown as markdown_lib

        html = markdown_lib.markdown(
            markdown_text,
            extensions=["tables", "fenced_code"],
        )
        return html, warnings
    except ImportError:
        warnings.append(
            "Dependency 'markdown' not installed. Falling back to minimal renderer; run `pip install markdown`."
        )

    lines = markdown_text.splitlines()
    chunks: list[str] = []
    in_list = False

    for raw in lines:
        line = raw.rstrip()
        if not line:
            if in_list:
                chunks.append("</ul>")
                in_list = False
            continue

        if line.startswith("### "):
            if in_list:
                chunks.append("</ul>")
                in_list = False
            chunks.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            if in_list:
                chunks.append("</ul>")
                in_list = False
            chunks.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            if in_list:
                chunks.append("</ul>")
                in_list = False
            chunks.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("- "):
            if not in_list:
                chunks.append("<ul>")
                in_list = True
            chunks.append(f"<li>{line[2:]}</li>")
        else:
            if in_list:
                chunks.append("</ul>")
                in_list = False
            chunks.append(f"<p>{line}</p>")

    if in_list:
        chunks.append("</ul>")

    return "\n".join(chunks), warnings


def _strip_markdown(markdown_text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", markdown_text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text


def _escape_pdf(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_minimal_pdf(text: str, path: Path) -> None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        lines = ["(empty report)"]

    content_lines = ["BT", "/F1 10 Tf", "50 800 Td", "12 TL"]
    for line in lines[:120]:
        content_lines.append(f"({_escape_pdf(line)}) Tj")
        content_lines.append("T*")
    content_lines.append("ET")
    stream_data = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects: list[bytes] = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    objects.append(
        b"4 0 obj\n<< /Length "
        + str(len(stream_data)).encode("ascii")
        + b" >>\nstream\n"
        + stream_data
        + b"\nendstream\nendobj\n"
    )
    objects.append(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(output))
        output.extend(obj)

    xref_start = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    output.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )

    path.write_bytes(bytes(output))


def _render_pdf(html_document: str, markdown_text: str, output_path: Path) -> list[str]:
    warnings: list[str] = []
    try:
        from weasyprint import HTML

        HTML(string=html_document, base_url=str(ROOT)).write_pdf(str(output_path))
        return warnings
    except ImportError:
        warnings.append(
            "Dependency 'weasyprint' not installed. Using minimal PDF fallback; run `pip install weasyprint`."
        )
        _write_minimal_pdf(_strip_markdown(markdown_text), output_path)
        return warnings


def _append_pdf_metadata_marker(pdf_path: Path, markdown_text: str) -> None:
    marker = ""
    for line in markdown_text.splitlines():
        if line.startswith("Pipeline run id:"):
            marker = line.strip()
            break
    if not marker:
        return
    with pdf_path.open("ab") as stream:
        stream.write(f"\n% {marker}\n".encode("latin-1", errors="ignore"))


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    markdown_path = _resolve(args.input, project_root=project_root)
    manifest_path = _resolve(args.manifest, project_root=project_root)
    output_root = _resolve(args.output_root, project_root=project_root)

    if not markdown_path.exists():
        print(f"Input markdown not found: {markdown_path}", file=sys.stderr)
        return 1

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    markdown_text = markdown_path.read_text(encoding="utf-8")
    manifest = load_report_manifest(manifest_path)
    traceability = build_traceability_map(manifest)

    body_html, markdown_warnings = _markdown_to_html(markdown_text)
    html_document = (
        "<!doctype html>\n"
        "<html><head><meta charset='utf-8'><title>EvasionBench Research Report</title>"
        f"<style>{BASE_CSS}</style></head><body>{body_html}</body></html>"
    )

    output_root.mkdir(parents=True, exist_ok=True)
    html_path = output_root / "report.html"
    pdf_path = output_root / "report.pdf"
    traceability_path = output_root / "report_traceability.json"

    html_path.write_text(html_document, encoding="utf-8")
    pdf_warnings = _render_pdf(html_document, markdown_text, pdf_path)
    _append_pdf_metadata_marker(pdf_path, markdown_text)

    traceability_payload = {
        "generated_at": utc_now_iso(),
        "source_markdown": normalize_path(markdown_path, base=project_root),
        "manifest": normalize_path(manifest_path, base=project_root),
        "items": {
            trace_id: {
                **entry,
                "referenced_in_report": trace_id in markdown_text,
            }
            for trace_id, entry in traceability.items()
        },
    }
    traceability_path.write_text(
        json.dumps(traceability_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for warning in [*markdown_warnings, *pdf_warnings]:
        print(f"warning: {warning}", file=sys.stderr)

    print(f"wrote html: {normalize_path(html_path, base=project_root)}")
    print(f"wrote pdf: {normalize_path(pdf_path, base=project_root)}")
    print(f"wrote traceability: {normalize_path(traceability_path, base=project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
