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
        self, sql: str, params: Optional[List[Any]] = None, max_rows: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute an analytical SQL query with security validation.
        Guarantees that no mutating or administrative commands can execute.
        Returns records as a list of Python dictionaries via native Polars conversion.
        Caps rows via Polars .head() before materializing into Python dicts to prevent OOM.
        """
        self.validate_sql_safety(sql)
        try:
            if params:
                result_pl = self.conn.execute(sql, params).pl()
            else:
                result_pl = self.conn.execute(sql).pl()
            if max_rows is not None and max_rows > 0:
                result_pl = result_pl.head(min(max_rows, 1000))
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

    def get_schema_metadata(self) -> Dict[str, Any]:
        """
        Apache Superset SQL Lab-style Schema Tree Introspection.
        Returns complete metadata for all tables, columns, data types, and sample values
        optimized for SQL Lab editor autocomplete and schema tree browsers.
        """
        tables_meta = []
        for table in self.list_tables():
            safe_table = re.sub(r"[^a-zA-Z0-9_]", "_", table)
            row_cnt = self.conn.execute(f"SELECT COUNT(*) FROM {safe_table};").fetchone()[0]
            desc = self.get_table_schema(table)

            # Sample first 3 records for value preview
            try:
                samples_pl = self.conn.execute(f"SELECT * FROM {safe_table} LIMIT 3;").pl()
                sample_dict = samples_pl.to_dict(as_series=False)
            except Exception:
                sample_dict = {}

            columns = []
            for col in desc:
                col_name = col.get("column_name")
                col_type = col.get("column_type")
                nullable = col.get("null", "YES") == "YES"
                samples = sample_dict.get(col_name, [])[:3]

                columns.append({
                    "name": col_name,
                    "type": str(col_type),
                    "nullable": nullable,
                    "sample_values": [str(s) for s in samples if s is not None]
                })

            tables_meta.append({
                "table_name": table,
                "row_count": row_cnt,
                "column_count": len(columns),
                "columns": columns
            })

        return {
            "engine": "DuckDB-InProcess",
            "total_tables": len(tables_meta),
            "tables": tables_meta
        }

    def execute_parameterized_query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        max_rows: int = 1000
    ) -> Dict[str, Any]:
        """
        Executes a parameterized SQL query (Apache Superset Jinja/mustache pattern).
        Supports {{ variable_name }} substitutions safely with parameter binding.
        """
        import time
        t_start = time.perf_counter()

        processed_sql = sql
        binding_params = []

        if params and isinstance(params, dict):
            # Replace {{ param }} with ? placeholders
            for key, val in params.items():
                pattern = re.compile(rf"\{{\{{\s*{re.escape(key)}\s*\}}\}}")
                if pattern.search(processed_sql):
                    processed_sql = pattern.sub("?", processed_sql)
                    binding_params.append(val)

        # Security check on processed SQL
        self.validate_sql_safety(processed_sql)

        # Enforce analytical limit if not explicitly limited
        if not re.search(r"\bLIMIT\s+\d+", processed_sql, re.IGNORECASE):
            processed_sql = f"{processed_sql.rstrip(';')} LIMIT {max_rows};"

        try:
            if binding_params:
                res_pl = self.conn.execute(processed_sql, binding_params).pl()
            else:
                res_pl = self.conn.execute(processed_sql).pl()

            records = res_pl.to_dicts()
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)

            return {
                "success": True,
                "sql": processed_sql,
                "duration_ms": duration_ms,
                "row_count": len(records),
                "columns": res_pl.columns,
                "column_types": [str(t) for t in res_pl.dtypes],
                "data": records
            }
        except Exception as ex:
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
            logger.error(f"Parameterized SQL execution failed: {ex}")
            return {
                "success": False,
                "sql": processed_sql,
                "duration_ms": duration_ms,
                "error": str(ex),
                "data": []
            }

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
