"""
report_generator.py
~~~~~~~~~~~~~~~~~~~
PDF report orchestrator.  Routes to one of two engines based on config:
  - PDF_ENGINE=reportlab  → existing ReportLab pipeline (default, local dev)
  - PDF_ENGINE=weasyprint  → Jinja2 HTML/CSS + WeasyPrint (production)
"""
from __future__ import annotations

import logging
import time
from io import BytesIO
from typing import Any

from app.core.config import settings

# ─── Re-exports (public API used by other modules) ──────────────────────────
from app.services.report_styles import (
    ReportMetadata,
    InvalidReportInputError,
)

# ─── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ─── Public Entry Point (Engine Factory) ─────────────────────────────────────
def generate_pdf_report(
    analysis_results: dict[str, Any],
    charts: dict[str, Any],
    filename: str,
) -> tuple[BytesIO, ReportMetadata]:
    """
    Generate the full GetReport PDF and return it as a BytesIO buffer.

    Routes to the correct engine based on ``settings.PDF_ENGINE``:
      - ``"reportlab"``  → pixel-level ReportLab (default, no system deps)
      - ``"weasyprint"`` → Jinja2 HTML/CSS → WeasyPrint PDF

    Returns:
        Tuple of (BytesIO buffer containing the PDF, ReportMetadata).
    """
    engine = settings.PDF_ENGINE.lower().strip()
    logger.info("═══ PDF Engine: %s ═══", engine)

    if engine == "weasyprint":
        from app.services.report_weasyprint import generate_pdf_weasyprint
        return generate_pdf_weasyprint(analysis_results, charts, filename)
    else:
        return _generate_pdf_reportlab(analysis_results, charts, filename)


# ─── ReportLab Engine (Original) ─────────────────────────────────────────────
def _generate_pdf_reportlab(
    analysis_results: dict[str, Any],
    charts: dict[str, Any],
    filename: str,
) -> tuple[BytesIO, ReportMetadata]:
    """Generate PDF using the original ReportLab pipeline."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate
    from reportlab.platypus.flowables import Flowable

    from app.services.report_styles import (
        _build_styles,
        _page_callback,
        _validate_inputs,
    )
    from app.services.report_sections import (
        _build_title_page,
        _build_executive_summary,
        _build_metadata_section,
        _build_confidence_scores_section,
        _build_semantic_analysis_section,
        _build_analysis_decisions_section,
        _build_cleaning_section,
        _build_comparison_section,
        _build_advanced_stats_section,
        _build_multicollinearity_section,
        _build_time_series_section,
        _build_insights_section,
        _build_correlations_section,
        _build_outliers_section,
        _build_categorical_section,
        _build_quality_flags_section,
        _build_feature_engineering_section,
        _build_smart_schema_section,
        _build_recommendations_section,
        _build_visualizations,
    )

    start_time = time.perf_counter()
    meta       = ReportMetadata(filename=filename)
    logger.info("═══ PDF Report Generation Started (ReportLab) — '%s' ═══", filename)

    # ── 1. Validate inputs ──────────────────────────────────────────────────
    _validate_inputs(analysis_results, charts, filename)

    # ── 2. Set up document ──────────────────────────────────────────────────
    buffer = BytesIO()
    doc    = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = _build_styles()
    story: list[Flowable] = []

    # ── 3. Build sections in order ──────────────────────────────────────────
    story.extend(_build_title_page(filename, styles))
    story.extend(_build_executive_summary(analysis_results, styles, meta))
    story.extend(_build_metadata_section(analysis_results, styles, meta))
    story.extend(_build_confidence_scores_section(analysis_results, styles, meta))
    story.extend(_build_semantic_analysis_section(analysis_results, styles, meta))
    story.extend(_build_analysis_decisions_section(analysis_results, styles, meta))
    story.extend(_build_cleaning_section(analysis_results, styles, meta))
    story.extend(_build_comparison_section(analysis_results, styles, meta))
    story.extend(_build_advanced_stats_section(analysis_results, styles, meta))
    story.extend(_build_multicollinearity_section(analysis_results, styles, meta))
    story.extend(_build_time_series_section(analysis_results, styles, meta))
    story.extend(_build_insights_section(analysis_results, styles, meta))
    story.extend(_build_correlations_section(analysis_results, styles, meta))
    story.extend(_build_outliers_section(analysis_results, styles, meta))
    story.extend(_build_categorical_section(analysis_results, styles, meta))
    story.extend(_build_quality_flags_section(analysis_results, styles, meta))
    story.extend(_build_feature_engineering_section(analysis_results, styles, meta))
    story.extend(_build_smart_schema_section(analysis_results, styles, meta))
    story.extend(_build_recommendations_section(analysis_results, styles, meta))
    story.extend(_build_visualizations(charts, styles, meta))

    # ── 4. Build PDF ────────────────────────────────────────────────────────
    try:
        doc.build(
            story,
            onFirstPage=_page_callback,
            onLaterPages=_page_callback,
        )
    except Exception as e:
        logger.error("PDF build failed: %s", str(e))
        raise

    # ── 5. Finalize ─────────────────────────────────────────────────────────
    buffer.seek(0)

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
    return buffer, meta

