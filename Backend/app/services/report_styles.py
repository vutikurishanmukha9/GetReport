"""
report_styles.py
~~~~~~~~~~~~~~~~
Shared styles, helpers, page callbacks, and data classes for the PDF report.
Extracted from report_generator.py for maintainability.
"""
from __future__ import annotations

import base64
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
def _build_styles() -> dict[str, ParagraphStyle]:
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
        textColor=Brand.TEXT_LIGHT,
        alignment=TA_CENTER,
        spaceAfter=6,
    )

    # Subtitle — smaller, light accent, centered
    custom["ReportSubtitle"] = ParagraphStyle(
        "ReportSubtitle",
        parent=base["Normal"],
        fontSize=12,
        textColor=Brand.ACCENT_LIGHT,
        alignment=TA_CENTER,
        spaceAfter=4,
    )

    # Section heading — accent color, left-aligned
    custom["SectionHeading"] = ParagraphStyle(
        "SectionHeading",
        parent=base["Heading2"],
        fontSize=16,
        textColor=Brand.ACCENT,
        spaceBefore=12,
        spaceAfter=6,
        alignment=TA_LEFT,
    )

    # Sub-heading — slightly smaller, dark text
    custom["SubHeading"] = ParagraphStyle(
        "SubHeading",
        parent=base["Heading3"],
        fontSize=12,
        textColor=Brand.TEXT_DARK,
        spaceBefore=8,
        spaceAfter=4,
    )

    # Body text — justified, readable
    custom["Body"] = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontSize=10,
        textColor=Brand.TEXT_DARK,
        alignment=TA_JUSTIFY,
        leading=14,
    )

    # Insight text — used inside the insight box
    custom["InsightText"] = ParagraphStyle(
        "InsightText",
        parent=base["Normal"],
        fontSize=10,
        textColor=Brand.TEXT_DARK,
        alignment=TA_LEFT,
        leading=14,
        leftIndent=12,
        rightIndent=12,
    )

    # Insight — alias for general usage
    custom["Insight"] = custom["InsightText"]

    # Warning text — used inside quality flag boxes
    custom["WarningText"] = ParagraphStyle(
        "WarningText",
        parent=base["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#92400e"),
        alignment=TA_LEFT,
        leading=13,
        leftIndent=10,
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
    )

    # Footer text
    custom["Footer"] = ParagraphStyle(
        "Footer",
        parent=base["Normal"],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )

    return custom


# ─── Page Header / Footer Callbacks ──────────────────────────────────────────
def _header_callback(canvas, doc) -> None:
    """Draw a thin accent bar at the top of every page (except page 1 = title)."""
    if doc.page == 1:
        return  # title page has no header
    canvas.setFillColor(Brand.ACCENT)
    canvas.rect(0, letter[1] - 0.35 * inch, letter[0], 0.35 * inch, fill=1, stroke=0)
    canvas.setFillColor(Brand.TEXT_LIGHT)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(0.5 * inch, letter[1] - 0.22 * inch, "GetReport — Data Analysis Report")


def _footer_callback(canvas, doc) -> None:
    """Draw page number at the bottom center of every page."""
    canvas.setFillColor(colors.grey)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(letter[0] / 2, 0.3 * inch, f"Page {doc.page}")

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
    b64_string: str,
    width: float,
    height: float,
    label: str,
    meta: ReportMetadata,
) -> Image | None:
    """
    Safely decode a base64 string into a ReportLab Image.

    Returns:
        A ReportLab Image object, or None if decoding failed.
    """
    try:
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
    t = Table(data, colWidths=col_widths)

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
    for row_idx in range(1, len(data)):
        if row_idx % 2 == 0:
            style_commands.append(
                ("BACKGROUND", (0, row_idx), (-1, row_idx), Brand.TABLE_ROW_ALT)
            )

    t.setStyle(TableStyle(style_commands))
    return t


# ─── Section Divider ─────────────────────────────────────────────────────────
def _divider() -> list[Flowable]:
    """Return a spacer + horizontal rule + spacer for visual section breaks."""
    return [
        Spacer(1, 0.15 * inch),
        HRFlowable(width="100%", thickness=1, color=Brand.DIVIDER, spaceAfter=6),
        Spacer(1, 0.1 * inch),
    ]
