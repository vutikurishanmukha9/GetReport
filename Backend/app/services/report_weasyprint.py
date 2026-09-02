"""
report_weasyprint.py
~~~~~~~~~~~~~~~~~~~~
WeasyPrint PDF engine — renders Jinja2 HTML/CSS templates into a PDF.

Used in production (Docker) where WeasyPrint system deps are available.
Falls back gracefully with a clear error if WeasyPrint is not installed.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from app.services.report_styles import ReportMetadata

logger = logging.getLogger(__name__)

# ─── Template directory & Jinja Environment ──────────────────────────────────
_TEMPLATE_DIR = Path(__file__).parent / "templates"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=True,
)

def _domain_label_filter(val: Any) -> str:
    """Format machine domain names to clean human-readable titles."""
    if not val or val == "Unknown":
        return "General Business / Generic"
    return str(val).replace("_", " ").title()

_JINJA_ENV.filters["domain_label"] = _domain_label_filter


class CSSCache:
    """
    Singleton cache for the parsed WeasyPrint CSS object.
    Parsing CSS is expensive; we should do it only once.
    """
    _css = None
    
    @classmethod
    def get(cls) -> Any:
        """
        Get the cached CSS object, parsing it if necessary.
        Must be called where weasyprint is known to be installed.
        """
        if cls._css is None:
            from weasyprint import CSS
            css_path = _TEMPLATE_DIR / "report.css"
            logger.info("Compiling and caching CSS from %s", css_path)
            cls._css = CSS(filename=str(css_path))
        return cls._css




def safe_url_fetcher(url: str, *args, **kwargs) -> dict:
    """
    Security URL fetcher for WeasyPrint to defeat SSRF (CWE-918) and LFI (CWE-73).
    - Allows data: URIs (inline images, fonts, charts).
    - Allows local files strictly within _TEMPLATE_DIR.
    - Blocks all remote network protocols (http://, https://, ftp://) and external file paths.
    """
    from urllib.parse import urlparse
    import base64

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    # On Windows, drive letters (e.g. D:, C:) are parsed as 1-char schemes by urlparse
    is_windows_drive = len(scheme) == 1 and scheme.isalpha() and len(url) > 1 and url[1] == ":"
    if is_windows_drive:
        scheme = "file"

    if scheme not in ("data", "file", ""):
        logger.warning(f"WeasyPrint SSRF attempt blocked: {url}")
        raise ValueError(f"External network request to '{url}' is forbidden for security.")

    if scheme in ("file", ""):
        if is_windows_drive:
            raw_path = url
        elif scheme == "file":
            raw_path = parsed.path
            if raw_path.startswith("/") and os.name == "nt" and len(raw_path) > 2 and raw_path[2] == ":":
                raw_path = raw_path[1:]
        else:
            raw_path = url
        target_path = os.path.realpath(raw_path)
        template_dir_real = os.path.realpath(str(_TEMPLATE_DIR))
        try:
            is_subpath = os.path.commonpath([template_dir_real, target_path]) == template_dir_real
        except ValueError:
            is_subpath = False

        if not (is_subpath and os.path.exists(target_path)):
            logger.warning(f"WeasyPrint local file access blocked: {url}")
            raise ValueError(f"Access to local file '{url}' is forbidden for security.")

    # URL is verified safe (data: or allowed local file).
    try:
        import weasyprint
        return weasyprint.default_url_fetcher(url, *args, **kwargs)
    except (ImportError, OSError):
        # Fallback when WeasyPrint C-libraries are not loaded locally
        if scheme == "data":
            header, _, data_part = url.partition(",")
            mime_type = header[5:].split(";")[0] if ";" in header else (header[5:] or "text/plain")
            content = base64.b64decode(data_part) if ";base64" in header else data_part.encode("utf-8")
            return {"string": content, "mime_type": mime_type}
        else:
            with open(target_path, "rb") as f:
                return {"string": f.read(), "mime_type": "text/css" if target_path.endswith(".css") else "application/octet-stream"}


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
    # Security: autoescape=True to prevent HTML injection from user data
    template = _JINJA_ENV.get_template("report.html")

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

    pdf_bytes = HTML(
        string=html_content,
        base_url=str(_TEMPLATE_DIR),
        url_fetcher=safe_url_fetcher,
    ).write_pdf(
        stylesheets=[CSSCache.get()],
    )

    import gc
    del html_content
    gc.collect()

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
        "Recommendations": bool(analysis.get("recommendations")),
        "Ranked Insights": bool(analysis.get("ranked_insights")),
        "Visualizations": bool(charts),
    }

    for section_name, present in section_checks.items():
        if present:
            meta.sections_included.append(section_name)
        else:
            meta.sections_skipped.append(section_name)

    if charts:
        meta.charts_included = _count_chart_items(charts, include_present=True)
        meta.charts_skipped = _count_chart_items(charts, include_present=False)


def _display_domain_name(domain: str) -> str:
    labels = {
        "logistics": "Supply Chain / Logistics",
        "supply_chain": "Supply Chain / Logistics",
        "sales_ecommerce": "Sales / E-Commerce",
        "hr_employee": "HR / Employee",
        "iot_sensor": "IoT / Sensor",
    }
    return labels.get(domain, str(domain).replace("_", " ").title())


def _count_chart_items(charts: dict[str, Any], include_present: bool) -> int:
    count = 0
    for value in charts.values():
        if isinstance(value, list):
            count += sum(1 for item in value if bool(item) is include_present)
        else:
            count += int(bool(value) is include_present)
    return count
