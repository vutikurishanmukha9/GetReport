"""
exporters.py
~~~~~~~~~~~~
Universal Security-Hardened Data Exporter (Template Method Pattern).
Enforces CWE-1236 spreadsheet formula neutralization across ALL string columns
for both CSV and Excel outputs. Streams directly to disk/temp files to bound RAM consumption.
"""
from __future__ import annotations

import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import polars as pl

from app.core.factories.registry import Registry

logger = logging.getLogger(__name__)


class BaseDataExporter(ABC):
    """
    Template Method Abstract Exporter.
    Universally neutralizes formula injection across ALL string columns before delegating serialization.
    Writes output to a secure temporary file stream to prevent in-memory OOM.
    """

    def export(
        self,
        df: pl.DataFrame,
        filename: str = "export",
        **kwargs: Any,
    ) -> tuple[Path, str, str]:
        """
        Template Method:
        1. Sanitizes string columns against CWE-1236 formula injection.
        2. Delegates format-specific disk streaming to `_serialize()`.
        Returns: (file_path_on_disk, mime_type, file_extension)
        """
        sanitized_df = self._neutralize_formulas(df)
        return self._serialize(sanitized_df, filename, **kwargs)

    def _neutralize_formulas(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Prefixes formula trigger characters (=, +, -, @, \\t, \\r) with a neutralizing
        single quote (').
        Applied universally to string/text columns.
        """
        exprs = []
        for col_name, dtype in zip(df.columns, df.dtypes):
            if dtype in (pl.Utf8, pl.String, pl.Object):
                condition = (
                    pl.col(col_name).str.starts_with("=")
                    | pl.col(col_name).str.starts_with("+")
                    | pl.col(col_name).str.starts_with("-")
                    | pl.col(col_name).str.starts_with("@")
                    | pl.col(col_name).str.starts_with("\t")
                    | pl.col(col_name).str.starts_with("\r")
                )
                neutralized = (
                    pl.when(condition)
                    .then(pl.lit("'") + pl.col(col_name))
                    .otherwise(pl.col(col_name))
                    .alias(col_name)
                )
                exprs.append(neutralized)
            else:
                exprs.append(pl.col(col_name))
        return df.with_columns(exprs)

    @abstractmethod
    def _serialize(
        self,
        df: pl.DataFrame,
        filename: str,
        **kwargs: Any,
    ) -> tuple[Path, str, str]:
        """Concrete serialization hook writing directly to temp file path."""
        pass


class CsvDataExporter(BaseDataExporter):
    """CSV Exporter writing directly to disk stream."""

    def _serialize(
        self,
        df: pl.DataFrame,
        filename: str,
        **kwargs: Any,
    ) -> tuple[Path, str, str]:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp.close()
        temp_path = Path(temp.name)
        df.write_csv(temp_path)
        return temp_path, "text/csv", ".csv"


class ExcelDataExporter(BaseDataExporter):
    """Excel (.xlsx) Exporter writing directly to disk stream."""

    def _serialize(
        self,
        df: pl.DataFrame,
        filename: str,
        **kwargs: Any,
    ) -> tuple[Path, str, str]:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        temp.close()
        temp_path = Path(temp.name)
        df.write_excel(temp_path)
        return (
            temp_path,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xlsx",
        )


class ParquetDataExporter(BaseDataExporter):
    """Columnar Parquet Exporter with snappy compression."""

    def _serialize(
        self,
        df: pl.DataFrame,
        filename: str,
        **kwargs: Any,
    ) -> tuple[Path, str, str]:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
        temp.close()
        temp_path = Path(temp.name)
        df.write_parquet(temp_path, compression="snappy")
        return temp_path, "application/vnd.apache.parquet", ".parquet"


class JsonDataExporter(BaseDataExporter):
    """JSON Line Exporter writing to disk stream."""

    def _serialize(
        self,
        df: pl.DataFrame,
        filename: str,
        **kwargs: Any,
    ) -> tuple[Path, str, str]:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        temp.close()
        temp_path = Path(temp.name)
        df.write_json(temp_path, pretty=True)
        return temp_path, "application/json", ".json"


exporter_registry = Registry[BaseDataExporter]("data_exporters")
exporter_registry.register_item("csv", CsvDataExporter, is_default=True)
exporter_registry.register_item("excel", ExcelDataExporter)
exporter_registry.register_item("xlsx", ExcelDataExporter)
exporter_registry.register_item("parquet", ParquetDataExporter)
exporter_registry.register_item("json", JsonDataExporter)


class DataExporterFactory:
    """Creator: Registry-based factory for data format exporters."""

    @classmethod
    def create_exporter(cls, format_type: Optional[str] = None) -> BaseDataExporter:
        target = (format_type or "csv").lower().strip()
        return exporter_registry.create(target)
