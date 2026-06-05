"""
Export the root README thesis draft to a formatted DOCX.

This script is intentionally small and deterministic. It handles the markdown
patterns used in the thesis README: headings, paragraphs, bullet/numbered
lists, simple markdown tables, code blocks, and horizontal rules.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
OUTPUT = ROOT / "thesis_portfolio_opt" / "deliverables" / "Thesis_Draft_Formatted.docx"


def add_page_number(paragraph) -> None:
    """Add a Word PAGE field to a paragraph."""
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char_1 = OxmlElement("w:fldChar")
    fld_char_1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char_2 = OxmlElement("w:fldChar")
    fld_char_2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char_1)
    run._r.append(instr_text)
    run._r.append(fld_char_2)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(2.5)

    add_page_number(section.footer.paragraphs[0])

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.first_line_indent = Cm(1.25)
    normal.paragraph_format.space_after = Pt(0)

    for style_name, size in [
        ("Title", 16),
        ("Heading 1", 16),
        ("Heading 2", 14),
        ("Heading 3", 12),
        ("Heading 4", 12),
    ]:
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(6)


def strip_inline_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = text.replace("---", "-")
    return text


def add_markdown_paragraph(doc: Document, text: str, style: str | None = None) -> None:
    paragraph = doc.add_paragraph(style=style)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.line_spacing = 1.5
    if style in {"List Bullet", "List Number"}:
        paragraph.paragraph_format.first_line_indent = None
    paragraph.add_run(strip_inline_markdown(text))


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip().strip("|")
        cells = [strip_inline_markdown(cell.strip()) for cell in raw.split("|")]
        # Skip markdown separator rows like |---|---|
        if not all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            rows.append(cells)
        i += 1
    return rows, i


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    n_cols = max(len(row) for row in rows)
    table = doc.add_table(rows=len(rows), cols=n_cols)
    table.style = "Table Grid"
    for r_idx, row in enumerate(rows):
        for c_idx in range(n_cols):
            text = row[c_idx] if c_idx < len(row) else ""
            cell = table.cell(r_idx, c_idx)
            cell.text = text
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if r_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
                for run in paragraph.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(10)
                    run.font.bold = r_idx == 0
    doc.add_paragraph()


def export() -> None:
    doc = Document()
    configure_document(doc)

    lines = README.read_text(encoding="utf-8").splitlines()
    in_code = False
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code = not in_code
            i += 1
            continue

        if in_code:
            if stripped:
                add_markdown_paragraph(doc, stripped, style="No Spacing")
            i += 1
            continue

        if not stripped or stripped == "---":
            i += 1
            continue

        if stripped.startswith("|"):
            rows, i = parse_table(lines, i)
            add_table(doc, rows)
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            text = strip_inline_markdown(heading.group(2))
            if level == 1:
                paragraph = doc.add_paragraph(text, style="Title")
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                style = f"Heading {min(level - 1, 4)}"
                doc.add_paragraph(text, style=style)
            i += 1
            continue

        if stripped.startswith("- "):
            add_markdown_paragraph(doc, stripped[2:], style="List Bullet")
            i += 1
            continue

        numbered = re.match(r"^\d+\.\s+(.*)$", stripped)
        if numbered:
            add_markdown_paragraph(doc, numbered.group(1), style="List Number")
            i += 1
            continue

        add_markdown_paragraph(doc, stripped)
        i += 1

    # Restart page numbering after front matter placeholder if needed by template.
    doc.add_section(WD_SECTION_START.CONTINUOUS)
    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    export()
