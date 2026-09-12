"""
Backend/app/services/duckdb_engine.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
High-performance in-process OLAP execution layer powered by DuckDB.
Provides zero-copy queries over cleaned Parquet files and Polars DataFrames.
"""
from __future__ import annotations

import os
import re
import logging
from typing import Any, Dict, List, Optional
import duckdb
import polars as pl

logger = logging.getLogger(__name__)


class DuckDBSecurityError(Exception):
    """Raised when a query attempts a non-read-only operation or administrative command."""
    pass


class DuckDBAnalyticalSession:
    """
    Manages an ephemeral, in-process DuckDB session for ad-hoc analytical queries.
    Enforces strict read-only execution, memory caps, and thread limits.
    """

    DISALLOWED_KEYWORDS = {
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
        "CREATE", "TRUNCATE", "RENAME", "ATTACH", "DETACH",
        "PRAGMA", "COPY", "EXPORT", "INSTALL", "LOAD", "SET",
        "READ_CSV", "READ_CSV_AUTO", "READ_PARQUET", "READ_JSON",
        "READ_JSON_AUTO", "READ_BLOB", "READ_TEXT", "SCAN_PARQUET",
        "SCAN_CSV", "GLOB", "HTTPFS", "POSTGRES_SCAN", "SQLITE_SCAN",
        "ICU_SORT_KEY", "PARQUET_SCAN"
    }

    def __init__(self, memory_limit: str = "2GB", threads: int = 4):
        self.conn = duckdb.connect(database=":memory:")
        self.conn.execute(f"SET memory_limit = '{memory_limit}';")
        self.conn.execute(f"SET threads = {threads};")
        # Enforce C++ engine-level sandboxing: block all local file and network operations
        self.conn.execute("SET enable_external_access = false;")
        self._registered_tables: set[str] = set()
        self._arrow_tables: Dict[str, Any] = {}

    def register_parquet(self, table_name: str, file_path: str) -> None:
        """Register a Parquet file as an analytical view via in-memory zero-copy Arrow."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet source file not found: {file_path}")
        safe_table_name = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
        # Load Parquet via Polars into Arrow memory table (avoids DuckDB external filesystem access)
        arrow_table = pl.read_parquet(file_path).to_arrow()
        self._arrow_tables[safe_table_name] = arrow_table
        self.conn.register(safe_table_name, arrow_table)
        self._registered_tables.add(safe_table_name)
        logger.info(f"Registered Parquet view '{safe_table_name}' from {file_path}")

    def register_polars(self, table_name: str, df: pl.DataFrame) -> None:
        """Zero-copy view registration from a Polars DataFrame via Apache Arrow."""
        arrow_table = df.to_arrow()
        safe_table_name = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
        self._arrow_tables[safe_table_name] = arrow_table
        self.conn.register(safe_table_name, arrow_table)
        self._registered_tables.add(safe_table_name)
        logger.info(f"Registered in-memory Polars DataFrame view '{safe_table_name}' ({df.height} rows)")

    def validate_sql_safety(self, sql: str) -> None:
        """
        Validates that the SQL query is strictly read-only.
        Blocks statement stacking, DDL, DML, filesystem access, and dangerous extensions.
        """
        clean_sql = sql.strip()
        if not clean_sql:
            raise DuckDBSecurityError("SQL query cannot be empty.")

        # Disallow semicolon statement chaining
        statements = [s.strip() for s in clean_sql.split(";") if s.strip()]
        if len(statements) > 1:
            raise DuckDBSecurityError("Multiple stacked SQL statements are forbidden.")

        primary_stmt = statements[0]
        first_token = primary_stmt.split()[0].upper()
        if first_token not in ("SELECT", "WITH", "DESCRIBE", "EXPLAIN", "SHOW", "VALUES"):
            raise DuckDBSecurityError(
                f"Security Violation: Statement type '{first_token}' is not permitted. Only analytical read queries are allowed."
            )

        # Tokenize statement into words to catch forbidden keywords and table functions
        tokens = set(re.findall(r"\b[A-Za-z_]+\b", primary_stmt.upper()))
        forbidden_found = tokens.intersection(self.DISALLOWED_KEYWORDS)
        if forbidden_found:
            raise DuckDBSecurityError(
                f"Security Violation: Forbidden keyword(s) or function(s) detected: {', '.join(sorted(forbidden_found))}"
            )

        # Disallow file path references or extension patterns inside string literals
        file_path_pattern = re.compile(
            r"""['"](?:(?:\.\.?[/\\]|[/\\]|[a-zA-Z]:[/\\])?.*?\.(?:csv|tsv|parquet|json|jsonl|ndjson|arrow|feather|txt|ini|env|py|sh|db|sqlite|log|conf|yml|yaml))['"]""",
            re.IGNORECASE
        )
        if file_path_pattern.search(primary_stmt):
            raise DuckDBSecurityError("Security Violation: Direct file path references are forbidden in analytical queries.")

    def execute_read_query(
        self, sql: str, params: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute an analytical SQL query with security validation.
        Guarantees that no mutating or administrative commands can execute.
        Returns records as a list of Python dictionaries via native Polars conversion.
        """
        self.validate_sql_safety(sql)
        try:
            if params:
                result_pl = self.conn.execute(sql, params).pl()
            else:
                result_pl = self.conn.execute(sql).pl()
            return result_pl.to_dicts()
        except Exception as e:
            logger.error(f"DuckDB query execution error: {e} | Query: {sql}")
            raise

    def get_column_profile(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Calculates exact statistical distributions at hardware memory speeds."""
        safe_table = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
        # Escape column name with double quotes
        escaped_col = f'"{column_name}"'
        query = f"""
        SELECT 
            COUNT(*) AS total_records,
            COUNT({escaped_col}) AS non_null_count,
            COUNT(*) - COUNT({escaped_col}) AS null_count,
            APPROX_COUNT_DISTINCT({escaped_col}) AS approx_distinct,
            MIN({escaped_col}) AS min_value,
            MAX({escaped_col}) AS max_value,
            AVG(TRY_CAST({escaped_col} AS DOUBLE)) AS mean_value,
            STDDEV_SAMP(TRY_CAST({escaped_col} AS DOUBLE)) AS std_deviation
        FROM {safe_table};
        """
        results = self.conn.execute(query).pl().to_dicts()
        return results[0] if results else {}

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Returns column names, types, and nullability for a registered table."""
        safe_table = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
        query = f"DESCRIBE {safe_table};"
        return self.conn.execute(query).pl().to_dicts()

    def list_tables(self) -> List[str]:
        """Returns all currently registered views and tables."""
        res = self.conn.execute("SHOW TABLES;").fetchall()
        return [row[0] for row in res]

    def close(self) -> None:
        """Release session resources."""
        try:
            self.conn.close()
        except Exception as e:
            logger.warning(f"Error closing DuckDB connection: {e}")
