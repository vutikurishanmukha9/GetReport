"""
report_weasyprint.py
~~~~~~~~~~~~~~~~~~~~
WeasyPrint PDF engine — renders Jinja2 HTML/CSS templates into a PDF.

Used in production (Docker) where WeasyPrint system deps are available.
Falls back gracefully with a clear error if WeasyPrint is not installed.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from app.services.report_styles import ReportMetadata

logger = logging.getLogger(__name__)

# ─── Template directory ──────────────────────────────────────────────────────
_TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_pdf_weasyprint(
    analysis_results: dict[str, Any],
    charts: dict[str, Any],
    filename: str,
) -> tuple[BytesIO, ReportMetadata]:
    """
    Generate a PDF report using Jinja2 + WeasyPrint.

    Args:
        analysis_results: Full analysis dict (analyze_dataset + cleaning + insights).
        charts:           Dict of base64-encoded PNG chart images.
        filename:         Original uploaded filename.

    Returns:
        Tuple of (BytesIO buffer with the PDF, ReportMetadata).

    Raises:
        ImportError: If WeasyPrint is not installed (should only happen on local dev).
    """
    start_time = time.perf_counter()
    meta = ReportMetadata(filename=filename)
    logger.info("═══ PDF Report Generation Started (WeasyPrint) — '%s' ═══", filename)

    # ── 1. Render HTML from template ────────────────────────────────────────
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=False,  # We control the HTML ourselves
    )
    template = env.get_template("report.html")

    # Build template context
    metadata = analysis_results.get("metadata", {})
    context = {
        "filename": filename,
        "generated_at": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        "metadata": metadata,
        "analysis": analysis_results,
        "charts": charts or {},
    }

    html_content = template.render(**context)

    # Track which sections are present
    _track_sections(analysis_results, charts, meta)

    # ── 2. Convert HTML → PDF via WeasyPrint ────────────────────────────────
    try:
        from weasyprint import HTML, CSS
    except ImportError:
        logger.error(
            "WeasyPrint is not installed.  Set PDF_ENGINE=reportlab in .env "
            "for local dev, or install WeasyPrint system dependencies."
        )
        raise ImportError(
            "WeasyPrint is not installed. "
            "Set PDF_ENGINE=reportlab for local development."
        )

    css_path = _TEMPLATE_DIR / "report.css"
    pdf_bytes = HTML(
        string=html_content,
        base_url=str(_TEMPLATE_DIR),
    ).write_pdf(
        stylesheets=[CSS(filename=str(css_path))],
    )

    buffer = BytesIO(pdf_bytes)

    # ── 3. Finalize metadata ────────────────────────────────────────────────
    meta.timing_ms = (time.perf_counter() - start_time) * 1000
    meta.success = True

    logger.info(
        "═══ PDF Report Complete (WeasyPrint) — %.2f ms | "
        "Sections: %d included, %d skipped | Charts: %d included, %d skipped ═══",
        meta.timing_ms,
        len(meta.sections_included),
        len(meta.sections_skipped),
        meta.charts_included,
        meta.charts_skipped,
    )
    return buffer, meta


def _track_sections(
    analysis: dict[str, Any],
    charts: dict[str, Any],
    meta: ReportMetadata,
) -> None:
    """Track which sections were included/skipped for metadata reporting."""
    section_checks = {
        "Executive Summary": bool(analysis.get("confidence_scores")),
        "Dataset Overview": True,
        "Confidence Scores": bool(analysis.get("confidence_scores") and analysis["confidence_scores"].get("columns")),
        "Semantic Intelligence": bool(analysis.get("semantic_analysis")),
        "Analysis Decisions": bool(analysis.get("analysis_decisions")),
        "Cleaning Summary": bool(analysis.get("cleaning_report")),
        "Quality Comparison": bool(analysis.get("cleaning_report") and analysis["cleaning_report"].get("before_after")),
        "Summary Statistics": bool(analysis.get("summary")),
        "Advanced Statistics": bool(analysis.get("summary")),
        "Strong Correlations": bool(analysis.get("strong_correlations")),
        "Time Series": bool(analysis.get("time_series_analysis")),
        "AI Insights": bool(analysis.get("insights")),
        "Outlier Detection": bool(analysis.get("outliers")),
        "Categorical Distribution": bool(analysis.get("categorical_distribution")),
        "Missing Patterns": bool(analysis.get("missing_patterns")),
        "Feature Engineering": bool(analysis.get("feature_engineering")),
        "Smart Schema": bool(analysis.get("smart_schema")),
        "Recommendations": bool(analysis.get("recommendations") and analysis["recommendations"].get("items")),
        "Ranked Insights": bool(analysis.get("ranked_insights")),
        "Visualizations": bool(charts),
    }

    for section_name, present in section_checks.items():
        if present:
            meta.sections_included.append(section_name)
        else:
            meta.sections_skipped.append(section_name)

    if charts:
        meta.charts_included = sum(1 for v in charts.values() if v)
        meta.charts_skipped = sum(1 for v in charts.values() if not v)
