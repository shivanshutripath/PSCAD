#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def _parse_markdown(md_text: str):
    lines = md_text.splitlines()
    blocks: list[tuple[str, str]] = []
    in_code = False
    code_buffer: list[str] = []
    para_buffer: list[str] = []

    def flush_para():
        nonlocal para_buffer
        if para_buffer:
            text = " ".join(x.strip() for x in para_buffer if x.strip())
            if text:
                blocks.append(("p", text))
        para_buffer = []

    for line in lines:
        stripped = line.rstrip("\n")

        if stripped.startswith("```"):
            if in_code:
                blocks.append(("code", "\n".join(code_buffer)))
                code_buffer = []
                in_code = False
            else:
                flush_para()
                in_code = True
            continue

        if in_code:
            code_buffer.append(stripped)
            continue

        if stripped.startswith("# "):
            flush_para()
            blocks.append(("h1", stripped[2:].strip()))
            continue
        if stripped.startswith("## "):
            flush_para()
            blocks.append(("h2", stripped[3:].strip()))
            continue
        if stripped.startswith("### "):
            flush_para()
            blocks.append(("h3", stripped[4:].strip()))
            continue

        if stripped.startswith("- "):
            flush_para()
            blocks.append(("li", stripped[2:].strip()))
            continue

        if stripped[:2].isdigit() and "). " in stripped:
            flush_para()
            blocks.append(("li", stripped.strip()))
            continue

        if not stripped.strip():
            flush_para()
            continue

        para_buffer.append(stripped)

    flush_para()
    if code_buffer:
        blocks.append(("code", "\n".join(code_buffer)))
    return blocks


def build_pdf(input_md: Path, output_pdf: Path) -> None:
    md_text = input_md.read_text(encoding="utf-8")
    blocks = _parse_markdown(md_text)

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        spaceAfter=10,
        textColor=colors.HexColor("#12263A"),
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        spaceBefore=8,
        spaceAfter=6,
        textColor=colors.HexColor("#12263A"),
    )
    h3 = ParagraphStyle(
        "H3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        spaceBefore=6,
        spaceAfter=4,
    )
    p = ParagraphStyle(
        "P",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.2,
        leading=14,
        spaceAfter=5,
    )
    li = ParagraphStyle(
        "LI",
        parent=p,
        leftIndent=12,
        bulletIndent=2,
    )
    code = ParagraphStyle(
        "CODE",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.8,
        leading=11.2,
        backColor=colors.HexColor("#F4F7FB"),
        borderPadding=6,
        borderColor=colors.HexColor("#D9E2EC"),
        borderWidth=0.5,
        borderRadius=2,
    )

    story = []
    for kind, text in blocks:
        safe = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        if kind == "h1":
            story.append(Paragraph(safe, h1))
        elif kind == "h2":
            story.append(Paragraph(safe, h2))
        elif kind == "h3":
            story.append(Paragraph(safe, h3))
        elif kind == "li":
            story.append(Paragraph(safe, li, bulletText="•"))
        elif kind == "code":
            code_html = safe.replace("\n", "<br/>")
            story.append(Paragraph(code_html, code))
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(safe, p))

    doc.build(story)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PDF from markdown report")
    parser.add_argument(
        "--input",
        default="veragrid_acopf_problem_formulation_report.md",
        help="Input markdown file",
    )
    parser.add_argument(
        "--output",
        default="veragrid_acopf_problem_formulation_report.pdf",
        help="Output PDF file",
    )
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    build_pdf(in_path, out_path)
    print(f"[DONE] PDF generated: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
