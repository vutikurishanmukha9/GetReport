"""
reports.py
~~~~~~~~~~
PDF Report Engine Factory & Fallback Pipeline.
Provides pre-flight availability checking (is_available) and render-time fallback chaining
(FallbackReportEngine) across Typst, WeasyPrint, and ReportLab.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Optional

from app.core.config import settings
from app.core.factories.registry import Registry
from app.services.report_styles import ReportMetadata

logger = logging.getLogger(__name__)


class BaseReportEngine(ABC):
    """Abstract Product: Contract for all PDF report engines."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if required dependencies, libraries, and binaries are present."""
        pass

    @abstractmethod
    def render_pdf(
        self,
        analysis_results: dict[str, Any],
        charts: dict[str, Any],
        filename: str,
    ) -> tuple[BytesIO, ReportMetadata]:
        pass


class TypstReportEngine(BaseReportEngine):
    """Ultra-fast native Rust Typst PDF compiler (<25MB RAM)."""

    @classmethod
    def is_available(cls) -> bool:
        try:
            import typst  # noqa: F401
            from app.services.report_typst import _FALLBACK_TEMPLATE, _PRIMARY_TEMPLATE
            return _PRIMARY_TEMPLATE.exists() or _FALLBACK_TEMPLATE.exists()
        except (ImportError, Exception):
            return False

    def render_pdf(
        self,
        analysis_results: dict[str, Any],
        charts: dict[str, Any],
        filename: str,
    ) -> tuple[BytesIO, ReportMetadata]:
        from app.services.report_typst import generate_pdf_typst
        logger.info("═══ Rendering via Typst (Rust Engine) ═══")
        return generate_pdf_typst(analysis_results, charts, filename)


class WeasyPrintReportEngine(BaseReportEngine):
    """HTML/CSS Paged Media rendering engine via WeasyPrint."""

    @classmethod
    def is_available(cls) -> bool:
        try:
            import weasyprint  # noqa: F401
            return True
        except (ImportError, Exception):
            return False

    def render_pdf(
        self,
        analysis_results: dict[str, Any],
        charts: dict[str, Any],
        filename: str,
    ) -> tuple[BytesIO, ReportMetadata]:
        from app.services.report_weasyprint import generate_pdf_weasyprint
        logger.info("═══ Rendering via WeasyPrint (HTML/CSS) ═══")
        return generate_pdf_weasyprint(analysis_results, charts, filename)


class ReportLabReportEngine(BaseReportEngine):
    """Pixel-precise pure Python ReportLab canvas engine (Universal fallback)."""

    @classmethod
    def is_available(cls) -> bool:
        try:
            import reportlab  # noqa: F401
            return True
        except ImportError:
            return False

    def render_pdf(
        self,
        analysis_results: dict[str, Any],
        charts: dict[str, Any],
        filename: str,
    ) -> tuple[BytesIO, ReportMetadata]:
        from app.services.report_generator import _generate_pdf_reportlab
        logger.info("═══ Rendering via ReportLab (Pure Python Fallback) ═══")
        return _generate_pdf_reportlab(analysis_results, charts, filename)


class FallbackReportEngine(BaseReportEngine):
    """
    Composite/Chain Engine: Tries candidate engines in sequence at render-time.
    If an engine fails during `render_pdf()`, logs the detailed exception
    and seamlessly proceeds to the next candidate in the chain.
    """

    def __init__(self, engines: list[BaseReportEngine]):
        if not engines:
            raise ValueError("FallbackReportEngine requires at least one candidate engine.")
        self.engines = engines

    @classmethod
    def is_available(cls) -> bool:
        return True

    def render_pdf(
        self,
        analysis_results: dict[str, Any],
        charts: dict[str, Any],
        filename: str,
    ) -> tuple[BytesIO, ReportMetadata]:
        last_error: Optional[Exception] = None
        for engine in self.engines:
            engine_name = engine.__class__.__name__
            try:
                logger.info(f"Attempting PDF generation using {engine_name}...")
                return engine.render_pdf(analysis_results, charts, filename)
            except Exception as e:
                logger.warning(
                    f"Report engine '{engine_name}' failed at render-time: {e}. "
                    f"Falling back to next engine in chain.",
                    exc_info=True,
                )
                last_error = e

        raise RuntimeError(f"All PDF rendering engines in fallback chain failed. Last error: {last_error}")


report_engine_registry = Registry[BaseReportEngine]("report_engines")
report_engine_registry.register_item("typst", TypstReportEngine, is_default=True)
report_engine_registry.register_item("weasyprint", WeasyPrintReportEngine)
report_engine_registry.register_item("reportlab", ReportLabReportEngine)


class ReportEngineFactory:
    """Creator: Registry-based factory assembling available and fallback engines."""

    @classmethod
    def create_engine(
        cls,
        preferred_engine: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> BaseReportEngine:
        pref = (preferred_engine or getattr(settings, "PDF_ENGINE", "typst") or "typst").lower().strip()

        # Deduplicated priority order starting with the preferred engine
        priority = [pref]
        for engine_key in ["typst", "weasyprint", "reportlab"]:
            if engine_key not in priority:
                priority.append(engine_key)

        # Pre-flight filter: Only consider engines where is_available() is True
        available_instances: list[BaseReportEngine] = []
        for key in priority:
            if report_engine_registry.contains(key):
                engine_cls = report_engine_registry.get(key)
                if hasattr(engine_cls, "is_available") and engine_cls.is_available():
                    available_instances.append(engine_cls())

        if not available_instances:
            raise RuntimeError(
                f"No PDF rendering engines are available on this system. Candidates tested: {priority}"
            )

        if not allow_fallback or len(available_instances) == 1:
            return available_instances[0]

        return FallbackReportEngine(available_instances)
