from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ─── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Brand Color Palette ─────────────────────────────────────────────────────
# Consistent across every table, header, and accent in the report
class Brand:
    DARK_BG       = colors.HexColor("#1e1b4b")   # title page background
    ACCENT        = colors.HexColor("#6366f1")   # headings, header bar
    ACCENT_LIGHT  = colors.HexColor("#a5b4fc")   # sub-accents
    TABLE_HEADER  = colors.HexColor("#4338ca")   # table header row
    TABLE_ROW_ALT = colors.HexColor("#eef2ff")   # alternating row fill
    TABLE_ROW     = colors.white                 # default row
    TEXT_DARK     = colors.HexColor("#1e1b4b")   # body text
    TEXT_LIGHT    = colors.white                 # text on dark backgrounds
    DIVIDER       = colors.HexColor("#c7d2fe")   # horizontal rule color
    INSIGHT_BG    = colors.HexColor("#f0f4ff")   # insight box background
    WARNING_BG    = colors.HexColor("#fef3c7")   # quality flag warning background
    WARNING_BORDER= colors.HexColor("#f59e0b")   # quality flag border


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

    Original logic preserved:
        - base64.b64decode() → BytesIO → Image with explicit width/height

    Enhanced:
        - Wrapped in try/except so a corrupted image skips gracefully
        - Increments meta.charts_included or meta.charts_skipped
        - Logs exactly which image failed and why

    Returns:
        A ReportLab Image object, or None if decoding failed.
    """
    try:
        img_data = base64.b64decode(b64_string)     # original logic
        img_io   = BytesIO(img_data)                # original logic
        img      = Image(img_io, width=width, height=height)  # original logic
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

    Original logic preserved:
        - Table(data) with TableStyle
        - Grey/colored header row with white bold text
        - Alternating row fills
        - Center-aligned, grid lines

    Enhanced:
        - Uses Brand color palette instead of generic grey/beige
        - Alternating rows (white / light accent) for readability
        - Optional column width control
        - Consistent padding

    Args:
        data:       2D list where data[0] is the header row.
        col_widths: Optional list of column widths in points.

    Returns:
        A styled ReportLab Table.
    """
    t = Table(data, colWidths=col_widths)

    style_commands = [
        # Header row — original logic preserved with brand colors
        ("BACKGROUND", (0, 0), (-1, 0), Brand.TABLE_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), Brand.TEXT_LIGHT),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),       # original logic
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        # Body rows
        ("BACKGROUND", (0, 1), (-1, -1), Brand.TABLE_ROW),
        ("TEXTCOLOR", (0, 1), (-1, -1), Brand.TEXT_DARK),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("TOPPADDING", (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        # Grid — original logic preserved
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


# ─── Section Builders ────────────────────────────────────────────────────────

def _build_title_page(filename: str, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    """
    Build the title page.

    Original logic preserved:
        - Title includes the filename
        - Subtitle says "Generated by GetReport AI"
        - Spacers for layout

    Enhanced:
        - Title page ends with a PageBreak so it's isolated
        - Uses branded styles (white text, accent subtitle)
    """
    story: list[Flowable] = []

    story.append(Spacer(1, 2.0 * inch))                                          # push title down
    story.append(Paragraph(f"Data Analysis Report", styles["ReportTitle"]))       # original: title
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(f"Source: {filename}", styles["ReportSubtitle"]))      # filename
    story.append(Spacer(1, 0.25 * inch))                                         # original spacer
    story.append(Paragraph("Generated by GetReport AI", styles["ReportSubtitle"]))  # original subtitle
    story.append(PageBreak())                                                     # title page is its own page

    logger.info("Title page built.")
    return story


def _build_metadata_section(
    analysis_results: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    meta: ReportMetadata,
) -> list[Flowable]:
    """
    Build a dataset overview section showing row/column counts and column types.
    This section does not exist in the original — it's a new addition that surfaces
    the metadata block from analysis.py.
    """
    if "metadata" not in analysis_results or not analysis_results["metadata"]:
        meta.sections_skipped.append({"section": "Dataset Overview", "reason": "No metadata available."})
        logger.debug("Metadata section skipped — no data.")
        return []

    story: list[Flowable] = []
    metadata = analysis_results["metadata"]

    story.append(Paragraph("Dataset Overview", styles["SectionHeading"]))
    story.append(Spacer(1, 0.1 * inch))

    # Build a clean 2-column summary table
    table_data = [["Property", "Value"]]
    display_keys = {
        "total_rows":            "Total Rows",
        "total_columns":         "Total Columns",
        "numeric_columns":       "Numeric Columns",
        "categorical_columns":   "Categorical Columns",
        "total_missing_values":  "Missing Values",
        "missing_value_pct":     "Missing Value %",
    }
    for key, label in display_keys.items():
        if key in metadata:
            val = metadata[key]
            formatted = f"{val}%" if "pct" in key else str(val)
            table_data.append([label, formatted])

    story.append(_build_styled_table(table_data, col_widths=[3.0 * inch, 2.5 * inch]))
    story.extend(_divider())

    meta.sections_included.append("Dataset Overview")
    logger.info("Metadata section built.")
    return story


def _build_cleaning_section(
    analysis_results: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    meta: ReportMetadata,
) -> list[Flowable]:
    """
    Build the data cleaning summary section.
    Surfaces the CleaningReport from ingestion.py so the user sees
    exactly what the pipeline did to their data.
    """
    if "cleaning_report" not in analysis_results or not analysis_results["cleaning_report"]:
        meta.sections_skipped.append({"section": "Cleaning Summary", "reason": "No cleaning report available."})
        logger.debug("Cleaning section skipped — no data.")
        return []

    story: list[Flowable] = []
    report = analysis_results["cleaning_report"]

    story.append(Paragraph("Data Cleaning Summary", styles["SectionHeading"]))
    story.append(Spacer(1, 0.1 * inch))

    table_data = [["Action Taken", "Count"]]
    actions = {
        "empty_rows_dropped":       "Empty Rows Removed",
        "empty_columns_dropped":    "Empty Columns Removed",
        "duplicate_rows_removed":   "Duplicate Rows Removed",
        "numeric_nans_filled":      "Numeric NaNs Filled (→ 0)",
        "categorical_nans_filled":  "Categorical NaNs Filled (→ Unknown)",
    }
    for key, label in actions.items():
        if key in report:
            table_data.append([label, str(report[key])])

    # Column renames sub-table
    if report.get("columns_renamed"):
        table_data.append(["", ""])
        table_data.append(["Columns Renamed", ""])
        table_data.append(["Original Name", "New Name"])
        for old, new in report["columns_renamed"].items():
            table_data.append([old, new])

    story.append(_build_styled_table(table_data, col_widths=[3.5 * inch, 2.0 * inch]))
    story.extend(_divider())

    meta.sections_included.append("Cleaning Summary")
    logger.info("Cleaning section built.")
    return story


def _build_executive_summary(
    analysis_results: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    meta: ReportMetadata,
) -> list[Flowable]:
    """
    Build the executive summary table showing descriptive stats for ALL numeric columns.

    Original logic preserved:
        - Only builds if "summary" key exists in analysis_results
        - Table with ["Metric", "Value"] headers
        - Floats formatted to .2f, others to str()
        - TableStyle with header bg, body bg, grid, center align

    Enhanced:
        - Iterates over ALL columns in summary (original only showed the first one)
        - Each column gets its own clearly labeled sub-table
        - Uses branded styling via _build_styled_table
    """
    if "summary" not in analysis_results or not analysis_results["summary"]:
        meta.sections_skipped.append({"section": "Executive Summary", "reason": "No summary stats available."})
        logger.debug("Executive summary skipped — no data.")
        return []

    story: list[Flowable] = []
    story.append(Paragraph("Executive Summary", styles["SectionHeading"]))   # original heading
    story.append(Spacer(1, 0.1 * inch))                                      # original spacer

    summary = analysis_results["summary"]

    # Enhanced: iterate ALL columns (original only took the first)
    for col_name, metrics in summary.items():
        story.append(Paragraph(f"Statistics: {col_name}", styles["SubHeading"]))

        # Original logic: build ["Metric", "Value"] table, format floats to .2f
        table_data = [["Metric", "Value"]]
        for k, v in metrics.items():
            val = f"{v:.2f}" if isinstance(v, (int, float)) else str(v)  # original formatting
            table_data.append([k.capitalize(), val])

        story.append(_build_styled_table(table_data, col_widths=[3.0 * inch, 2.5 * inch]))
        story.append(Spacer(1, 0.15 * inch))

    story.extend(_divider())

    meta.sections_included.append("Executive Summary")
    logger.info("Executive summary built — %d column(s).", len(summary))
    return story


def _build_insights_section(
    analysis_results: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    meta: ReportMetadata,
) -> list[Flowable]:
    """
    Build the AI Insights section using text from insights.py.
    Renders the insight text inside a lightly shaded box for visual emphasis.
    """
    if "insights" not in analysis_results or not analysis_results["insights"]:
        meta.sections_skipped.append({"section": "AI Insights", "reason": "No insights text available."})
        logger.debug("Insights section skipped — no data.")
        return []

    story: list[Flowable] = []
    insights_text = analysis_results["insights"]

    story.append(Paragraph("AI-Generated Insights", styles["SectionHeading"]))
    story.append(Spacer(1, 0.1 * inch))

    # Render inside a shaded box using a single-cell table
    insight_para = Paragraph(insights_text.replace("\n", "<br/>"), styles["InsightText"])
    insight_table = Table([[insight_para]], colWidths=[6.0 * inch])
    insight_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), Brand.INSIGHT_BG),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("BOX", (0, 0), (-1, -1), 1, Brand.ACCENT_LIGHT),
    ]))
    story.append(insight_table)
    story.extend(_divider())

    meta.sections_included.append("AI Insights")
    logger.info("Insights section built.")
    return story


def _build_correlations_section(
    analysis_results: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    meta: ReportMetadata,
) -> list[Flowable]:
    """
    Build the strong correlations section as a text table.
    Surfaces the strong_correlations list from analysis.py.
    """
    if "strong_correlations" not in analysis_results or not analysis_results["strong_correlations"]:
        meta.sections_skipped.append({"section": "Correlations", "reason": "No strong correlations found."})
        logger.debug("Correlations section skipped — no strong pairs.")
        return []

    story: list[Flowable] = []
    story.append(Paragraph("Strong Correlations", styles["SectionHeading"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "The following column pairs have a notably strong statistical relationship (|r| ≥ 0.7).",
        styles["Body"]
    ))
    story.append(Spacer(1, 0.1 * inch))

    table_data = [["Column A", "Column B", "r Value", "Direction", "Strength"]]
    for pair in analysis_results["strong_correlations"]:
        table_data.append([
            str(pair.get("column_a", "—")),
            str(pair.get("column_b", "—")),
            str(pair.get("r_value", "—")),
            str(pair.get("direction", "—")).capitalize(),
            str(pair.get("strength", "—")).capitalize(),
        ])

    col_widths = [1.3 * inch, 1.3 * inch, 1.0 * inch, 1.2 * inch, 1.2 * inch]
    story.append(_build_styled_table(table_data, col_widths=col_widths))
    story.extend(_divider())

    meta.sections_included.append("Strong Correlations")
    logger.info("Correlations section built — %d pair(s).", len(analysis_results["strong_correlations"]))
    return story


def _build_outliers_section(
    analysis_results: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    meta: ReportMetadata,
) -> list[Flowable]:
    """
    Build the outliers section.
    Surfaces outlier data from analysis.py with bounds and counts.
    """
    if "outliers" not in analysis_results or not analysis_results["outliers"]:
        meta.sections_skipped.append({"section": "Outliers", "reason": "No outliers detected."})
        logger.debug("Outliers section skipped — none detected.")
        return []

    story: list[Flowable] = []
    story.append(Paragraph("Outlier Detection", styles["SectionHeading"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "Values falling outside 1.5× the interquartile range (IQR) are flagged as outliers.",
        styles["Body"]
    ))
    story.append(Spacer(1, 0.1 * inch))

    table_data = [["Column", "Outlier Count", "% of Data", "Lower Bound", "Upper Bound"]]
    for col, info in analysis_results["outliers"].items():
        table_data.append([
            col,
            str(info.get("count", "—")),
            f"{info.get('percentage', 0):.1f}%",
            f"{info.get('lower_bound', '—')}",
            f"{info.get('upper_bound', '—')}",
        ])

    col_widths = [1.6 * inch, 1.1 * inch, 1.0 * inch, 1.2 * inch, 1.2 * inch]
    story.append(_build_styled_table(table_data, col_widths=col_widths))
    story.extend(_divider())

    meta.sections_included.append("Outlier Detection")
    logger.info("Outliers section built — %d column(s).", len(analysis_results["outliers"]))
    return story


def _build_categorical_section(
    analysis_results: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    meta: ReportMetadata,
) -> list[Flowable]:
    """
    Build the categorical distribution section.
    Shows top categories per column with counts and percentages.
    """
    if "categorical_distribution" not in analysis_results or not analysis_results["categorical_distribution"]:
        meta.sections_skipped.append({"section": "Categorical Distribution", "reason": "No categorical data."})
        logger.debug("Categorical section skipped — no data.")
        return []

    story: list[Flowable] = []
    story.append(Paragraph("Categorical Distributions", styles["SectionHeading"]))
    story.append(Spacer(1, 0.1 * inch))

    for col_name, col_data in analysis_results["categorical_distribution"].items():
        story.append(Paragraph(f"{col_name}", styles["SubHeading"]))

        categories = col_data.get("categories", {})
        if not categories:
            continue

        table_data = [["Category", "Count", "Percentage"]]
        for cat_name, cat_info in categories.items():
            table_data.append([
                str(cat_name),
                str(cat_info.get("count", "—")),
                f"{cat_info.get('percentage', 0):.1f}%",
            ])

        col_widths = [3.2 * inch, 1.3 * inch, 1.3 * inch]
        story.append(_build_styled_table(table_data, col_widths=col_widths))
        story.append(Spacer(1, 0.15 * inch))

    story.extend(_divider())

    meta.sections_included.append("Categorical Distribution")
    logger.info("Categorical section built.")
    return story


def _build_quality_flags_section(
    analysis_results: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    meta: ReportMetadata,
) -> list[Flowable]:
    """
    Build the data quality flags section.
    Renders each flagged column inside a warning-styled box.
    """
    if "column_quality_flags" not in analysis_results or not analysis_results["column_quality_flags"]:
        meta.sections_skipped.append({"section": "Quality Flags", "reason": "No quality flags."})
        logger.debug("Quality flags section skipped — none found.")
        return []

    story: list[Flowable] = []
    story.append(Paragraph("Data Quality Flags", styles["SectionHeading"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "The following columns have data-quality issues that may affect analysis accuracy.",
        styles["Body"]
    ))
    story.append(Spacer(1, 0.1 * inch))

    for col_name, flags in analysis_results["column_quality_flags"].items():
        flag_text = "<br/>".join(f"⚠ {flag}" for flag in flags)
        flag_para = Paragraph(f"<b>{col_name}</b><br/>{flag_text}", styles["WarningText"])

        warning_box = Table([[flag_para]], colWidths=[6.0 * inch])
        warning_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), Brand.WARNING_BG),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("BOX", (0, 0), (-1, -1), 1, Brand.WARNING_BORDER),
        ]))
        story.append(warning_box)
        story.append(Spacer(1, 0.1 * inch))

    story.extend(_divider())

    meta.sections_included.append("Data Quality Flags")
    logger.info("Quality flags section built — %d column(s) flagged.", len(analysis_results["column_quality_flags"]))
    return story


def _build_visualizations(
    charts: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    meta: ReportMetadata,
) -> list[Flowable]:
    """
    Build all chart/visualization sections.

    Original logic preserved:
        - correlation_heatmap: base64 → Image at 6×4.5 inch
        - distributions: loop, each has "column" + "image", rendered at 5×3 inch
        - Heading and label per chart

    Enhanced:
        - Each image decode is wrapped in _decode_image for graceful failure
        - Skips individual charts that fail without killing the whole section
        - Handles additional chart types: bar_charts, trend_charts, pie_charts
        - Logs which charts succeeded and which were skipped
    """
    story: list[Flowable] = []
    has_any_chart = False

    # ── Correlation Heatmap (original logic preserved) ──────────────────────
    if "correlation_heatmap" in charts and charts["correlation_heatmap"]:
        story.append(Paragraph("Correlation Analysis", styles["SectionHeading"]))   # original heading
        story.append(Spacer(1, 0.1 * inch))                                         # original spacer

        img = _decode_image(charts["correlation_heatmap"], 6 * inch, 4.5 * inch, "Correlation Heatmap", meta)
        if img:
            story.append(img)                                                        # original logic
            story.append(Spacer(1, 0.25 * inch))                                     # original spacer
            has_any_chart = True
        else:
            story.append(Paragraph("Correlation heatmap could not be rendered.", styles["Body"]))

    # ── Distribution Charts (original logic preserved) ──────────────────────
    if "distributions" in charts and charts["distributions"]:
        story.append(Paragraph("Feature Distributions", styles["SectionHeading"]))   # original heading

        for dist in charts["distributions"]:                                         # original loop
            col_label = dist.get("column", "Unknown")
            story.append(Paragraph(f"Distribution of {col_label}", styles["SubHeading"]))  # original label

            img = _decode_image(dist.get("image", ""), 5 * inch, 3 * inch, f"Distribution: {col_label}", meta)
            if img:
                story.append(img)                                                    # original logic
                story.append(Spacer(1, 0.1 * inch))                                  # original spacer
                has_any_chart = True
            else:
                story.append(Paragraph(f"Distribution chart for '{col_label}' could not be rendered.", styles["Body"]))
                story.append(Spacer(1, 0.1 * inch))

    # ── Bar Charts (enhanced — new chart type support) ──────────────────────
    if "bar_charts" in charts and charts["bar_charts"]:
        story.append(Paragraph("Category Comparisons", styles["SectionHeading"]))

        for bar in charts["bar_charts"]:
            col_label = bar.get("column", "Unknown")
            story.append(Paragraph(f"Bar Chart: {col_label}", styles["SubHeading"]))

            img = _decode_image(bar.get("image", ""), 5.5 * inch, 3.5 * inch, f"Bar Chart: {col_label}", meta)
            if img:
                story.append(img)
                story.append(Spacer(1, 0.15 * inch))
                has_any_chart = True
            else:
                story.append(Paragraph(f"Bar chart for '{col_label}' could not be rendered.", styles["Body"]))

    # ─── Bivariate Boxplots (new) ───────────────────────────────────────────
    if "boxplots" in charts and charts["boxplots"]:
        story.append(Paragraph("Category vs Numeric Analysis", styles["SectionHeading"]))
        
        for box in charts["boxplots"]:
            col_label = box.get("column", "Unknown")
            story.append(Paragraph(f"Spread: {col_label}", styles["SubHeading"])) # e.g. "Spread: Salary vs Gender"
            
            img = _decode_image(box.get("image", ""), 5.5 * inch, 4 * inch, f"Boxplot: {col_label}", meta)
            if img:
                story.append(img)
                story.append(Spacer(1, 0.15 * inch))
                has_any_chart = True
            else:
                story.append(Paragraph(f"Boxplot for '{col_label}' could not be rendered.", styles["Body"]))

    # ── Trend Charts (enhanced — new chart type support) ────────────────────
    if "trend_charts" in charts and charts["trend_charts"]:
        story.append(Paragraph("Trend Analysis", styles["SectionHeading"]))

        for trend in charts["trend_charts"]:
            col_label = trend.get("column", "Unknown")
            story.append(Paragraph(f"Trend: {col_label}", styles["SubHeading"]))

            img = _decode_image(trend.get("image", ""), 5.5 * inch, 3.0 * inch, f"Trend: {col_label}", meta)
            if img:
                story.append(img)
                story.append(Spacer(1, 0.15 * inch))
                has_any_chart = True
            else:
                story.append(Paragraph(f"Trend chart for '{col_label}' could not be rendered.", styles["Body"]))

    # ── Pie Charts (enhanced — new chart type support) ──────────────────────
    if "pie_charts" in charts and charts["pie_charts"]:
        story.append(Paragraph("Composition Analysis", styles["SectionHeading"]))

        for pie in charts["pie_charts"]:
            col_label = pie.get("column", "Unknown")
            story.append(Paragraph(f"Composition: {col_label}", styles["SubHeading"]))

            img = _decode_image(pie.get("image", ""), 4.0 * inch, 4.0 * inch, f"Pie Chart: {col_label}", meta)
            if img:
                story.append(img)
                story.append(Spacer(1, 0.15 * inch))
                has_any_chart = True
            else:
                story.append(Paragraph(f"Pie chart for '{col_label}' could not be rendered.", styles["Body"]))

    if has_any_chart:
        story.extend(_divider())
        meta.sections_included.append("Visualizations")
        logger.info("Visualizations section built — %d chart(s) included, %d skipped.",
                    meta.charts_included, meta.charts_skipped)
    else:
        meta.sections_skipped.append({"section": "Visualizations", "reason": "No charts could be rendered."})
        logger.warning("Visualizations section skipped — no charts rendered successfully.")

    return story


# ─── Main Entry Point ────────────────────────────────────────────────────────
def generate_pdf_report(
    analysis_results: dict[str, Any],
    charts: dict[str, Any],
    filename: str,
) -> tuple[BytesIO, ReportMetadata]:
    """
    Generate the full GetReport PDF and return it as a BytesIO buffer.

    Original logic preserved:
        - BytesIO buffer as output
        - SimpleDocTemplate with letter pagesize
        - story list assembled sequentially → doc.build(story)
        - Title page with filename and "Generated by GetReport AI"
        - Executive summary table (if summary exists)
        - Correlation heatmap and distribution charts (if charts exist)
        - buffer.seek(0) before returning
        - On exception: logs error and re-raises

    Enhanced:
        - Input validation before any work begins
        - Branded, consistent styling across all elements
        - Full section coverage (metadata, cleaning, summary, insights,
          correlations, outliers, categorical, quality flags, visualizations)
        - Page headers (accent bar + title) and footers (page numbers)
        - Title page is isolated with a PageBreak
        - Each section builder is independent — one failing doesn't kill the report
        - Graceful image handling (corrupted images are skipped, not crashed)
        - ReportMetadata returned alongside the buffer so the caller knows
          exactly what's in the PDF

    Args:
        analysis_results: The full dict from analyze_dataset() + cleaning_report + insights.
        charts:           Dict of base64-encoded chart images.
        filename:         The original uploaded file name.

    Returns:
        Tuple of (BytesIO buffer containing the PDF, ReportMetadata).

    Raises:
        InvalidReportInputError: If inputs fail validation.
        Exception:               Re-raised on any unexpected PDF build failure (original behavior).
    """
    start_time = time.perf_counter()
    meta       = ReportMetadata(filename=filename)
    logger.info("═══ PDF Report Generation Started — '%s' ═══", filename)

    # ── 1. Validate inputs ──────────────────────────────────────────────────
    _validate_inputs(analysis_results, charts, filename)

    # ── 2. Set up document (original logic preserved) ───────────────────────
    buffer = BytesIO()                                                           # original
    doc    = SimpleDocTemplate(                                                  # original
        buffer,
        pagesize=letter,                                                         # original
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = _build_styles()
    story: list[Flowable] = []

    # ── 3. Build sections in order ───────────────────────────────────────────
    # Each builder is independent — if one has no data it returns [] and logs why

    story.extend(_build_title_page(filename, styles))
    story.extend(_build_metadata_section(analysis_results, styles, meta))
    story.extend(_build_cleaning_section(analysis_results, styles, meta))
    story.extend(_build_executive_summary(analysis_results, styles, meta))
    
    # New: Statistical Deep Dive
    story.extend(_build_advanced_stats_section(analysis_results, styles, meta))
    story.extend(_build_multicollinearity_section(analysis_results, styles, meta))
    story.extend(_build_time_series_section(analysis_results, styles, meta))
    
    story.extend(_build_insights_section(analysis_results, styles, meta))
    story.extend(_build_correlations_section(analysis_results, styles, meta))
    story.extend(_build_outliers_section(analysis_results, styles, meta))
    story.extend(_build_categorical_section(analysis_results, styles, meta))
    story.extend(_build_quality_flags_section(analysis_results, styles, meta))
    story.extend(_build_visualizations(charts, styles, meta))

    # ── 4. Build PDF (original logic preserved) ─────────────────────────────

    # print("DEBUG: Calling doc.build")
    try:
        doc.build(
            story,
            onFirstPage=_page_callback,
            onLaterPages=_page_callback,
        )
    except Exception as e:
        logger.error("PDF build failed: %s", str(e))
        raise

    # ── 5. Finalize (original logic preserved) ──────────────────────────────
    buffer.seek(0)                                                               # original

    meta.timing_ms = (time.perf_counter() - start_time) * 1000
    meta.success   = True

    logger.info(
        "═══ PDF Report Complete — %.2f ms | Sections: %d included, %d skipped | Charts: %d included, %d skipped ═══",
        meta.timing_ms,
        len(meta.sections_included),
        len(meta.sections_skipped),
        meta.charts_included,
        meta.charts_skipped,
    )
    return buffer, meta 
 d e f   _ b u i l d _ a d v a n c e d _ s t a t s _ s e c t i o n (  
         a n a l y s i s _ r e s u l t s :   d i c t [ s t r ,   A n y ] ,  
         s t y l e s :   d i c t [ s t r ,   P a r a g r a p h S t y l e ] ,  
         m e t a :   R e p o r t M e t a d a t a ,  
 )   - >   l i s t [ F l o w a b l e ] :  
         " " "  
         S h o w s   S k e w n e s s   a n d   K u r t o s i s   f o r   c o l u m n s   t h a t   d e v i a t e   f r o m   n o r m a l .  
         " " "  
         i f   " a d v a n c e d _ s t a t s "   n o t   i n   a n a l y s i s _ r e s u l t s   o r   n o t   a n a l y s i s _ r e s u l t s [ " a d v a n c e d _ s t a t s " ] :  
                 r e t u r n   [ ]  
  
         s t o r y :   l i s t [ F l o w a b l e ]   =   [ ]  
         s t a t s   =   a n a l y s i s _ r e s u l t s [ " a d v a n c e d _ s t a t s " ]  
          
         #   F i l t e r   f o r   n o t a b l e   d e v i a t i o n s   ( a b s ( s k e w )   >   1   o r   a b s ( k u r t )   >   3 )  
         n o t a b l e   =   { k :   v   f o r   k ,   v   i n   s t a t s . i t e m s ( )   i f   a b s ( v . g e t ( " s k e w n e s s " ,   0 )   o r   0 )   >   1   o r   a b s ( v . g e t ( " k u r t o s i s " ,   0 )   o r   0 )   >   3 }  
          
         i f   n o t   n o t a b l e :  
                 r e t u r n   [ ]  
  
         s t o r y . a p p e n d ( P a r a g r a p h ( " D i s t r i b u t i o n   S h a p e   A n a l y s i s " ,   s t y l e s [ " S e c t i o n H e a d i n g " ] ) )  
         s t o r y . a p p e n d ( P a r a g r a p h (  
                 " T h e   f o l l o w i n g   c o l u m n s   e x h i b i t   s i g n i f i c a n t   s k e w   ( > 1 . 0 )   o r   h e a v y   t a i l s   ( K u r t o s i s   > 3 . 0 ) .   "  
                 " S t a n d a r d   m e a n / s t d   m e t r i c s   m a y   b e   m i s l e a d i n g   f o r   t h e s e   f e a t u r e s . " ,  
                 s t y l e s [ " B o d y " ]  
         ) )  
         s t o r y . a p p e n d ( S p a c e r ( 1 ,   0 . 1   *   i n c h ) )  
  
         t a b l e _ d a t a   =   [ [ " C o l u m n " ,   " S k e w n e s s " ,   " K u r t o s i s " ,   " S h a p e " ] ]  
         f o r   c o l ,   v a l   i n   n o t a b l e . i t e m s ( ) :  
                 s k e w   =   v a l . g e t ( " s k e w n e s s " ,   0 )  
                 k u r t   =   v a l . g e t ( " k u r t o s i s " ,   0 )  
                  
                 s h a p e _ d e s c   =   [ ]  
                 i f   s k e w   >   1 :   s h a p e _ d e s c . a p p e n d ( " R i g h t   S k e w e d " )  
                 e l i f   s k e w   <   - 1 :   s h a p e _ d e s c . a p p e n d ( " L e f t   S k e w e d " )  
                 i f   k u r t   >   3 :   s h a p e _ d e s c . a p p e n d ( " H e a v y   T a i l s " )  
                  
                 t a b l e _ d a t a . a p p e n d ( [  
                         c o l ,  
                         f " { s k e w : . 2 f } " ,  
                         f " { k u r t : . 2 f } " ,  
                         " ,   " . j o i n ( s h a p e _ d e s c )  
                 ] )  
  
         s t o r y . a p p e n d ( _ b u i l d _ s t y l e d _ t a b l e ( t a b l e _ d a t a ,   c o l _ w i d t h s = [ 2 . 5   *   i n c h ,   1 . 2   *   i n c h ,   1 . 2   *   i n c h ,   2 . 5   *   i n c h ] ) )  
         s t o r y . e x t e n d ( _ d i v i d e r ( ) )  
         m e t a . s e c t i o n s _ i n c l u d e d . a p p e n d ( " A d v a n c e d   S t a t i s t i c s " )  
         r e t u r n   s t o r y  
  
  
 d e f   _ b u i l d _ m u l t i c o l l i n e a r i t y _ s e c t i o n (  
         a n a l y s i s _ r e s u l t s :   d i c t [ s t r ,   A n y ] ,  
         s t y l e s :   d i c t [ s t r ,   P a r a g r a p h S t y l e ] ,  
         m e t a :   R e p o r t M e t a d a t a ,  
 )   - >   l i s t [ F l o w a b l e ] :  
         " " "  
         S h o w s   p o t e n t i a l   m u l t i c o l l i n e a r i t y   ( H i g h   V I F   P r o x y ) .  
         " " "  
         i f   " m u l t i c o l l i n e a r i t y "   n o t   i n   a n a l y s i s _ r e s u l t s   o r   n o t   a n a l y s i s _ r e s u l t s [ " m u l t i c o l l i n e a r i t y " ] :  
                 r e t u r n   [ ]  
  
         s t o r y :   l i s t [ F l o w a b l e ]   =   [ ]  
         m u l t i   =   a n a l y s i s _ r e s u l t s [ " m u l t i c o l l i n e a r i t y " ]  
  
         s t o r y . a p p e n d ( P a r a g r a p h ( " M u l t i c o l l i n e a r i t y   W a r n i n g " ,   s t y l e s [ " S e c t i o n H e a d i n g " ] ) )  
         s t o r y . a p p e n d ( P a r a g r a p h (  
                 " T h e   f o l l o w i n g   f e a t u r e   p a i r s   h a v e   v e r y   h i g h   c o r r e l a t i o n   ( > 0 . 9 5 ) ,   s u g g e s t i n g   t h e y   m a y   p r o v i d e   r e d u n d a n t   i n f o r m a t i o n . " ,  
                 s t y l e s [ " W a r n i n g T e x t " ]  
         ) )  
         s t o r y . a p p e n d ( S p a c e r ( 1 ,   0 . 1   *   i n c h ) )  
  
         t a b l e _ d a t a   =   [ [ " F e a t u r e   A " ,   " F e a t u r e   B " ,   " C o r r e l a t i o n " ,   " V e r d i c t " ] ]  
         f o r   i t e m   i n   m u l t i :  
                 t a b l e _ d a t a . a p p e n d ( [  
                         i t e m [ " f e a t u r e s " ] [ 0 ] ,  
                         i t e m [ " f e a t u r e s " ] [ 1 ] ,  
                         f " { i t e m [ ' c o r r e l a t i o n ' ] : . 2 f } " ,  
                         " R e d u n d a n t "  
                 ] )  
  
         s t o r y . a p p e n d ( _ b u i l d _ s t y l e d _ t a b l e ( t a b l e _ d a t a ,   c o l _ w i d t h s = [ 2 . 5   *   i n c h ,   2 . 5   *   i n c h ,   1 . 2   *   i n c h ,   1 . 2   *   i n c h ] ) )  
         s t o r y . e x t e n d ( _ d i v i d e r ( ) )  
         m e t a . s e c t i o n s _ i n c l u d e d . a p p e n d ( " M u l t i c o l l i n e a r i t y " )  
         r e t u r n   s t o r y  
  
  
 d e f   _ b u i l d _ t i m e _ s e r i e s _ s e c t i o n (  
         a n a l y s i s _ r e s u l t s :   d i c t [ s t r ,   A n y ] ,  
         s t y l e s :   d i c t [ s t r ,   P a r a g r a p h S t y l e ] ,  
         m e t a :   R e p o r t M e t a d a t a ,  
 )   - >   l i s t [ F l o w a b l e ] :  
         " " "  
         S h o w s   T i m e - S e r i e s   I n t e g r i t y   ( S o r t   o r d e r ,   D r i f t ) .  
         " " "  
         i f   " t i m e _ s e r i e s _ a n a l y s i s "   n o t   i n   a n a l y s i s _ r e s u l t s   o r   n o t   a n a l y s i s _ r e s u l t s [ " t i m e _ s e r i e s _ a n a l y s i s " ] :  
                 r e t u r n   [ ]  
  
         s t o r y :   l i s t [ F l o w a b l e ]   =   [ ]  
         t s   =   a n a l y s i s _ r e s u l t s [ " t i m e _ s e r i e s _ a n a l y s i s " ]  
  
         s t o r y . a p p e n d ( P a r a g r a p h ( " T i m e - S e r i e s   I n t e g r i t y " ,   s t y l e s [ " S e c t i o n H e a d i n g " ] ) )  
          
         #   S o r t   S t a t u s  
         i s _ s o r t e d   =   t s . g e t ( " i s _ s o r t e d " ,   F a l s e )  
         s o r t _ t e x t   =   " C h r o n o l o g i c a l l y   S o r t e d   ( G o o d ) "   i f   i s _ s o r t e d   e l s e   " N O T   S o r t e d   b y   T i m e   ( R i s k ) "  
         s o r t _ c o l o r   =   B r a n d . T E X T _ D A R K   i f   i s _ s o r t e d   e l s e   c o l o r s . H e x C o l o r ( " # b 9 1 c 1 c " )  
          
         s t o r y . a p p e n d ( P a r a g r a p h ( f " P r i m a r y   T i m e   C o l u m n :   < b > { t s . g e t ( ' p r i m a r y _ t i m e _ c o l ' ) } < / b > " ,   s t y l e s [ " B o d y " ] ) )  
         s t o r y . a p p e n d ( P a r a g r a p h ( f " S t a t u s :   < f o n t   c o l o r = { s o r t _ c o l o r } > { s o r t _ t e x t } < / f o n t > " ,   s t y l e s [ " B o d y " ] ) )  
         s t o r y . a p p e n d ( S p a c e r ( 1 ,   0 . 1   *   i n c h ) )  
  
         #   D r i f t   T a b l e  
         d r i f t s   =   t s . g e t ( " d r i f t _ d e t e c t e d " ,   [ ] )  
         i f   d r i f t s :  
                 s t o r y . a p p e n d ( P a r a g r a p h ( " < b > C o n c e p t u a l   D r i f t   D e t e c t e d   ( > 3 0 %   M e a n   S h i f t ) < / b > " ,   s t y l e s [ " W a r n i n g T e x t " ] ) )  
                 s t o r y . a p p e n d ( S p a c e r ( 1 ,   0 . 0 5   *   i n c h ) )  
                  
                 t a b l e _ d a t a   =   [ [ " C o l u m n " ,   " A v g   ( 1 s t   H a l f ) " ,   " A v g   ( 2 n d   H a l f ) " ,   " %   S h i f t " ] ]  
                 f o r   d   i n   d r i f t s :  
                         t a b l e _ d a t a . a p p e n d ( [  
                                 d [ " c o l u m n " ] ,  
                                 f " { d [ ' m e a n _ p 1 ' ] } " ,  
                                 f " { d [ ' m e a n _ p 2 ' ] } " ,  
                                 f " { d [ ' s h i f t _ p c t ' ] } % "  
                         ] )  
                 s t o r y . a p p e n d ( _ b u i l d _ s t y l e d _ t a b l e ( t a b l e _ d a t a ) )  
         e l s e :  
                 s t o r y . a p p e n d ( P a r a g r a p h ( " N o   s i g n i f i c a n t   d r i f t   d e t e c t e d   a c r o s s   t h e   t i m e   w i n d o w . " ,   s t y l e s [ " B o d y " ] ) )  
  
         s t o r y . e x t e n d ( _ d i v i d e r ( ) )  
         m e t a . s e c t i o n s _ i n c l u d e d . a p p e n d ( " T i m e - S e r i e s   A n a l y s i s " )  
         r e t u r n   s t o r y  
 