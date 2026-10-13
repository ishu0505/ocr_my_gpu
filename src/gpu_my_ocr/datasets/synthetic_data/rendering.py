"""Shared selectable-text PDF layouts, independent of category-specific content."""

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

from .content import FOOTER, NOTICE

PAGE_SIZE = letter
LEFT = 40
WIDTH = PAGE_SIZE[0] - 2 * LEFT


def draw_text(canvas: Canvas, text: str, x: float, y: float, size: int = 10) -> None:
    """Draw one ASCII line, rejecting text that would overflow the page boundary."""
    canvas.setFont("Helvetica", size)
    if canvas.stringWidth(text, "Helvetica", size) > PAGE_SIZE[0] - LEFT - x:
        raise ValueError(f"Text exceeds page width: {text}")
    canvas.drawString(x, y, text)


def draw_fields(canvas: Canvas, document: dict[str, Any], y: float) -> float:
    """Draw a boxed form or compact ledger; return the next available baseline."""
    height = 23 if document["layout"] == "form" else 19
    for name, value in document["fields"].items():
        if document["layout"] == "form":
            canvas.setFillColor(colors.HexColor("#f0f3f5"))
            canvas.rect(LEFT, y - 6, WIDTH, height - 2, fill=1, stroke=0)
            canvas.setFillColor(colors.black)
        draw_text(canvas, f"{name.replace('_', ' ').title()}: {value}", LEFT + 6, y)
        y -= height
    return y - 12


def draw_table(canvas: Canvas, table: dict[str, Any], y: float) -> float:
    """Draw a readable ruled table with ordered cells; return the next baseline."""
    draw_text(canvas, table["title"], LEFT, y, 12)
    y -= 28
    count = len(table["columns"])
    widths = [WIDTH * 0.46] + [WIDTH * 0.54 / (count - 1)] * (count - 1)
    for row_index, row in enumerate([table["columns"], *table["rows"]]):
        if row_index == 0:
            canvas.setFillColor(colors.HexColor("#e0e8ed"))
            canvas.rect(LEFT, y - 8, WIDTH, 24, fill=1, stroke=0)
            canvas.setFillColor(colors.black)
        x = LEFT
        for cell, width in zip(row, widths, strict=True):
            if canvas.stringWidth(cell, "Helvetica", 10) > width - 12:
                raise ValueError(f"Table cell too wide: {cell}")
            draw_text(canvas, cell, x + 6, y)
            x += width
        canvas.setStrokeColor(colors.HexColor("#b8c3ca"))
        canvas.line(LEFT, y - 8, LEFT + WIDTH, y - 8)
        y -= 24
    return y - 14


def render_digital(document: dict[str, Any], output: Path) -> None:
    """Render pre-existing structured truth to a deterministic one-page digital PDF."""
    canvas = Canvas(str(output), pagesize=PAGE_SIZE, invariant=1, pageCompression=1)
    canvas.setTitle(document["title"])
    canvas.setAuthor("gpu_my_ocr synthetic generator")
    draw_text(canvas, document["title"], LEFT, 744, 18)
    draw_text(canvas, NOTICE, LEFT, 721, 9)
    draw_text(canvas, f"Document ID: {document['document_id']}", LEFT, 700, 9)
    y = draw_fields(canvas, document, 668)
    for table in document["tables"]:
        y = draw_table(canvas, table, y)
    for line in document["narrative"]:
        draw_text(canvas, line, LEFT, y, 9)
        y -= 20
    if document["category"] == "auditor_report":
        path = canvas.beginPath()
        path.moveTo(LEFT + 5, y)
        for offset in range(0, 110, 10):
            path.lineTo(LEFT + offset, y + (12 if offset % 20 else 0))
        canvas.drawPath(path)
        y -= 20
    if y < 65:
        raise ValueError("Content exceeds one-page budget")
    draw_text(canvas, FOOTER, LEFT, 35, 9)
    canvas.showPage()
    canvas.save()
