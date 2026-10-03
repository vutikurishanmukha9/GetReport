"""
app.core.factories
~~~~~~~~~~~~~~~~~~
Registry-based factories, strategy dispatchers, and template method pipelines.
Explicitly loads all module registrations upon import to eliminate import side-effects.
"""
from app.core.factories.registry import Registry
from app.core.factories.storage import (
    StorageProviderFactory,
    storage_registry,
    get_storage_provider,
)
from app.core.factories.reports import (
    BaseReportEngine,
    TypstReportEngine,
    WeasyPrintReportEngine,
    ReportLabReportEngine,
    FallbackReportEngine,
    ReportEngineFactory,
    report_engine_registry,
)
from app.core.factories.exporters import (
    BaseDataExporter,
    CsvDataExporter,
    ExcelDataExporter,
    ParquetDataExporter,
    JsonDataExporter,
    DataExporterFactory,
    exporter_registry,
)

from app.core.factories.remediation import (
    remediation_registry,
    apply_remediation,
)

__all__ = [
    "Registry",
    "StorageProviderFactory",
    "storage_registry",
    "get_storage_provider",
    "BaseReportEngine",
    "TypstReportEngine",
    "WeasyPrintReportEngine",
    "ReportLabReportEngine",
    "FallbackReportEngine",
    "ReportEngineFactory",
    "report_engine_registry",
    "BaseDataExporter",
    "CsvDataExporter",
    "ExcelDataExporter",
    "ParquetDataExporter",
    "JsonDataExporter",
    "DataExporterFactory",
    "exporter_registry",
    "remediation_registry",
    "apply_remediation",
]
