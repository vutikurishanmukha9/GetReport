"""
report_styles.py
~~~~~~~~~~~~~~~~
Shared styles, helpers, page callbacks, and data classes for the PDF report.
Extracted from report_generator.py for maintainability.
"""
from __future__ import annotations

import base64
import html
import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Spacer,
    Image,
    Table,
    TableStyle,
    HRFlowable,
    Paragraph,
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from app.services.theme import Brand

# ─── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ─── Custom Exceptions ───────────────────────────────────────────────────────
class InvalidReportInputError(ValueError):
    """Raised when required report inputs are missing or malformed."""


# ─── Report Metadata (returned alongside the buffer) ────────────────────────
@dataclass
class ReportMetadata:
    """
    Tracks what went into the generated PDF so the caller knows exactly
    what the report contains.

    Attributes:
        filename:          The source file name used in the title.
        sections_included: Which sections made it into the report.
        sections_skipped:  Which sections were skipped and why.
        charts_included:   How many chart images were successfully embedded.
        charts_skipped:    How many chart images failed and were skipped.
        timing_ms:         How long PDF generation took (milliseconds).
        success:           True if the PDF was generated without a fatal error.
    """
    filename:          str                        = ""
    sections_included: list[str]                  = field(default_factory=list)
    sections_skipped:  list[dict[str, str]]       = field(default_factory=list)
    charts_included:   int                        = 0
    charts_skipped:    int                        = 0
    timing_ms:         float                      = 0.0
    success:           bool                       = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename":          self.filename,
            "sections_included": self.sections_included,
            "sections_skipped":  self.sections_skipped,
            "charts_included":   self.charts_included,
            "charts_skipped":    self.charts_skipped,
            "timing_ms":         round(self.timing_ms, 2),
            "success":           self.success,
        }


# ─── Style Factory ───────────────────────────────────────────────────────────
def get_custom_styles() -> dict[str, ParagraphStyle]:
    """
    Create the full branded style sheet for the report.
    Extends the base ReportLab styles with GetReport-specific overrides.

    Returns a dictionary keyed by style name for easy lookup.
    """
    base = getSampleStyleSheet()

    custom: dict[str, ParagraphStyle] = {}

    # Normal — base style
    custom["Normal"] = base["Normal"]

    # Title — large, white, centered (used on the dark title page)
    custom["ReportTitle"] = ParagraphStyle(
        "ReportTitle",
        parent=base["Title"],
        fontSize=28,
        textColor=Brand.ACCENT,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        leading=34,
        spaceAfter=10,
    )

    # Subtitle — smaller, light accent, centered
    custom["ReportSubtitle"] = ParagraphStyle(
        "ReportSubtitle",
        parent=base["Normal"],
        fontSize=12,
        textColor=Brand.TEXT_MUTED,
        alignment=TA_CENTER,
        fontName="Helvetica",
        leading=16,
        spaceAfter=6,
    )

    # Section heading — accent color, left-aligned, keepWithNext=True to prevent orphan titles!
    custom["SectionHeading"] = ParagraphStyle(
        "SectionHeading",
        parent=base["Heading2"],
        fontSize=15,
        textColor=Brand.ACCENT,
        fontName="Helvetica-Bold",
        spaceBefore=14,
        spaceAfter=6,
        alignment=TA_LEFT,
        leading=18,
        keepWithNext=True,
    )

    # Sub-heading — slightly smaller, dark text
    custom["SubHeading"] = ParagraphStyle(
        "SubHeading",
        parent=base["Heading3"],
        fontSize=11,
        textColor=Brand.TEXT_DARK,
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=4,
        leading=14,
        keepWithNext=True,
    )

    # Body text — justified, readable
    custom["Body"] = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontSize=9.5,
        textColor=Brand.TEXT_DARK,
        fontName="Helvetica",
        alignment=TA_LEFT,
        leading=14,
        spaceAfter=4,
    )

    # Insight text — used inside the insight box
    custom["InsightText"] = ParagraphStyle(
        "InsightText",
        parent=base["Normal"],
        fontSize=9.5,
        textColor=Brand.TEXT_DARK,
        fontName="Helvetica",
        alignment=TA_LEFT,
        leading=14,
        leftIndent=8,
        rightIndent=8,
    )

    # Insight — alias for general usage
    custom["Insight"] = custom["InsightText"]

    # Warning text — used inside quality flag boxes
    custom["WarningText"] = ParagraphStyle(
        "WarningText",
        parent=base["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#7C2D12"),
        fontName="Helvetica",
        alignment=TA_LEFT,
        leading=13,
        leftIndent=6,
    )

    # Table Caption — used above tables
    custom["TableCaption"] = ParagraphStyle(
        "TableCaption",
        parent=base["Normal"],
        fontSize=10,
        textColor=Brand.TEXT_DARK,
        alignment=TA_LEFT,
        fontName="Helvetica-Bold",
        leading=14,
        spaceAfter=4,
        keepWithNext=True,
    )

    # Footer text
    custom["Footer"] = ParagraphStyle(
        "Footer",
        parent=base["Normal"],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )

    # ─── Aliases used by report_renderer.py ──────────────────────────────
    custom["ModernTitle"] = ParagraphStyle(
        "ModernTitle",
        parent=base["Title"],
        fontSize=22,
        textColor=Brand.ACCENT,
        alignment=TA_LEFT,
        spaceAfter=8,
        fontName="Helvetica-Bold",
        leading=26,
        keepWithNext=True,
    )

    custom["ModernHeading"] = ParagraphStyle(
        "ModernHeading",
        parent=base["Heading2"],
        fontSize=15,
        textColor=Brand.ACCENT,
        fontName="Helvetica-Bold",
        spaceBefore=14,
        spaceAfter=6,
        alignment=TA_LEFT,
        leading=18,
        keepWithNext=True,
    )

    custom["ModernBody"] = ParagraphStyle(
        "ModernBody",
        parent=base["Normal"],
        fontSize=9.5,
        textColor=Brand.TEXT_DARK,
        fontName="Helvetica",
        alignment=TA_LEFT,
        leading=14,
        spaceAfter=4,
    )

    custom["MetaValue"] = ParagraphStyle(
        "MetaValue",
        parent=base["Normal"],
        fontSize=18,
        textColor=Brand.ACCENT,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        spaceAfter=2,
    )

    custom["MetaLabel"] = ParagraphStyle(
        "MetaLabel",
        parent=base["Normal"],
        fontSize=8.5,
        textColor=colors.HexColor("#555555"),
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )

    custom["InsightBox"] = ParagraphStyle(
        "InsightBox",
        parent=base["Normal"],
        fontSize=9.5,
        textColor=Brand.TEXT_DARK,
        fontName="Helvetica",
        alignment=TA_LEFT,
        leading=14,
        leftIndent=8,
        rightIndent=8,
        backColor=Brand.INSIGHT_BG,
        borderColor=Brand.INSIGHT_BORDER,
        borderWidth=1,
        borderPadding=8,
        borderRadius=6,
    )

    return custom


# ─── Page Header / Footer Callbacks ──────────────────────────────────────────
def _header_callback(canvas, doc) -> None:
    """Draw a thin burgundy accent bar at the top of every page (except page 1 = title)."""
    if doc.page == 1:
        return  # title page has no running header
    canvas.saveState()
    canvas.setFillColor(Brand.ACCENT)
    canvas.rect(0, letter[1] - 0.35 * inch, letter[0], 0.35 * inch, fill=1, stroke=0)
    canvas.setFillColor(Brand.TEXT_LIGHT)
    canvas.setFont("Helvetica-Bold", 8.5)
    canvas.drawString(0.6 * inch, letter[1] - 0.22 * inch, "GetReport — Executive Data Analysis Briefing")
    canvas.restoreState()


def _footer_callback(canvas, doc) -> None:
    """Draw footer page number and confidentiality note at the bottom of every page."""
    canvas.saveState()
    canvas.setStrokeColor(Brand.DIVIDER)
    canvas.setLineWidth(0.5)
    canvas.line(0.6 * inch, 0.45 * inch, letter[0] - 0.6 * inch, 0.45 * inch)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.6 * inch, 0.3 * inch, "Confidential — Generated by GetReport AI")
    canvas.drawRightString(letter[0] - 0.6 * inch, 0.3 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _page_callback(canvas, doc) -> None:
    """Apply both header and footer to the page."""
    _header_callback(canvas, doc)
    _footer_callback(canvas, doc)


# ─── Input Validation ────────────────────────────────────────────────────────
def _validate_inputs(
    analysis_results: dict[str, Any],
    charts: dict[str, Any],
    filename: str,
) -> None:
    """
    Validate all inputs before any PDF work begins.

    Raises:
        InvalidReportInputError: If any required input is missing or wrong type.
    """
    if not isinstance(filename, str) or filename.strip() == "":
        raise InvalidReportInputError("filename must be a non-empty string.")

    if not isinstance(analysis_results, dict):
        raise InvalidReportInputError(
            f"analysis_results must be a dict, got {type(analysis_results).__name__}."
        )

    if not isinstance(charts, dict):
        raise InvalidReportInputError(
            f"charts must be a dict, got {type(charts).__name__}."
        )

    logger.info("Input validation passed for report: '%s'.", filename)


# ─── Safe Base64 Image Decoder ───────────────────────────────────────────────
def _decode_image(
    b64_string: Any,
    width: float,
    height: float,
    label: str,
    meta: ReportMetadata,
) -> Image | None:
    """
    Safely decode a base64 string or chart dictionary into a ReportLab Image.

    Returns:
        A ReportLab Image object, or None if decoding failed.
    """
    try:
        if isinstance(b64_string, dict):
            b64_string = b64_string.get("image", "")

        if not b64_string or not isinstance(b64_string, str):
            meta.charts_skipped += 1
            return None

        # Clean base64 URI prefixes if present
        if b64_string.startswith("data:image/") and "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]

        # If image is raw SVG XML, ReportLab cannot rasterize directly
        if b64_string.strip().startswith("<svg") or b64_string.strip().startswith("<?xml"):
            meta.charts_skipped += 1
            logger.info("SVG vector image '%s' skipped for ReportLab rasterizer.", label)
            return None

        img_data = base64.b64decode(b64_string)
        img_io   = BytesIO(img_data)
        img      = Image(img_io, width=width, height=height)
        meta.charts_included += 1
        logger.info("Image decoded successfully — '%s'.", label)
        return img
    except Exception as e:
        meta.charts_skipped += 1
        logger.warning("Failed to decode image '%s': %s — skipping.", label, str(e))
        return None


# ─── Styled Table Builder ────────────────────────────────────────────────────
def _build_styled_table(
    data: list[list[str]],
    col_widths: list[float] | None = None,
) -> Table:
    """
    Build a ReportLab Table with the branded GetReport style.

    Args:
        data:       2D list where data[0] is the header row.
        col_widths: Optional list of column widths in points.

    Returns:
        A styled ReportLab Table.
    """
    wrapped_data = _wrap_table_cells(data)
    t = Table(wrapped_data, colWidths=col_widths, splitByRow=True, repeatRows=1)

    style_commands = [
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), Brand.TABLE_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), Brand.TEXT_LIGHT),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        # Body rows
        ("BACKGROUND", (0, 1), (-1, -1), Brand.TABLE_ROW),
        ("TEXTCOLOR", (0, 1), (-1, -1), Brand.TEXT_DARK),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("TOPPADDING", (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        # Grid
        ("GRID", (0, 0), (-1, -1), 0.5, Brand.DIVIDER),
        # Left-align first column (metric names)
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
    ]

    # Alternating row backgrounds
    for row_idx in range(1, len(wrapped_data)):
        if row_idx % 2 == 0:
            style_commands.append(
                ("BACKGROUND", (0, row_idx), (-1, row_idx), Brand.TABLE_ROW_ALT)
            )

    t.setStyle(TableStyle(style_commands))
    return t


def _wrap_table_cells(data: list[list[Any]]) -> list[list[Any]]:
    """Convert scalar cells to Paragraphs so table text wraps and markup renders."""
    if not data:
        return data

    header_style = ParagraphStyle(
        "ReportTableHeader",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=Brand.TEXT_LIGHT,
        alignment=TA_CENTER,
        wordWrap="CJK",
    )
    body_style = ParagraphStyle(
        "ReportTableCell",
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=Brand.TEXT_DARK,
        alignment=TA_LEFT,
        wordWrap="CJK",
    )

    wrapped: list[list[Any]] = []
    for row_idx, row in enumerate(data):
        style = header_style if row_idx == 0 else body_style
        wrapped.append([
            cell if isinstance(cell, Flowable) else Paragraph(_safe_table_markup(cell), style)
            for cell in row
        ])
    return wrapped


def _safe_table_markup(value: Any) -> str:
    """Escape arbitrary values while allowing report-generated inline tags."""
    text = html.escape("" if value is None else str(value))
    allowed = {
        "&lt;b&gt;": "<b>",
        "&lt;/b&gt;": "</b>",
        "&lt;i&gt;": "<i>",
        "&lt;/i&gt;": "</i>",
        "&lt;br/&gt;": "<br/>",
        "&lt;br /&gt;": "<br/>",
        "&lt;font color=&#x27;": "<font color='",
        "&#x27;&gt;": "'>",
        "&lt;/font&gt;": "</font>",
    }
    for escaped, raw in allowed.items():
        text = text.replace(escaped, raw)
    return text


# ─── Section Divider ─────────────────────────────────────────────────────────
def _divider() -> list[Flowable]:
    """Return a spacer + horizontal rule + spacer for visual section breaks."""
    return [
        Spacer(1, 0.15 * inch),
        HRFlowable(width="100%", thickness=1, color=Brand.DIVIDER, spaceAfter=6),
        Spacer(1, 0.1 * inch),
    ]
