"""
Backend/app/services/duckdb_engine.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
High-performance in-process OLAP execution layer powered by DuckDB.
Provides zero-copy queries over cleaned Parquet files and Polars DataFrames,
with sqlglot AST read-only validation, LRU Arrow view caching, parameterized
bindings, EXPLAIN plan inspection, and column profiling.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime, time as dt_time
from decimal import Decimal
import json
import logging
import math
import os
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import duckdb
import polars as pl
import sqlglot
from sqlglot import exp
from sqlglot.tokens import TokenType

logger = logging.getLogger(__name__)

_ARROW_CACHE_LOCK = threading.Lock()
_ARROW_CACHE: "OrderedDict[Tuple[str, int, int], Any]" = OrderedDict()
_MAX_CACHED_DATASETS = 6
_MAX_CACHE_TABLE_BYTES = 512 * 1024 * 1024  # 512 MB cap per cached table

_PARAM_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
_FILE_PATH_PATTERN = re.compile(
    r"""(?:^|[\s'"])(?:(?:\.\.?[/\\]|[/\\]|[a-zA-Z]:[/\\]).*?\.(?:csv|tsv|parquet|json|jsonl|ndjson|arrow|feather|txt|ini|env|py|sh|db|sqlite|log|conf|yml|yaml)|[^/\\\s]+\.(?:csv|tsv|parquet|json|jsonl|ndjson|arrow|feather|ini|env|db|sqlite))$""",
    re.IGNORECASE,
)


def invalidate_arrow_cache() -> None:
    """Clear the cached Arrow dataset registry."""
    with _ARROW_CACHE_LOCK:
        _ARROW_CACHE.clear()


def _load_cached_arrow_table(file_path: str) -> Any:
    """Load a Parquet or tabular dataset into an Arrow table using an bounded LRU cache."""
    abs_path = os.path.abspath(file_path)
    stat = os.stat(abs_path)
    cache_key = (abs_path, stat.st_mtime_ns, stat.st_size)

    with _ARROW_CACHE_LOCK:
        cached = _ARROW_CACHE.get(cache_key)
        if cached is not None:
            _ARROW_CACHE.move_to_end(cache_key)
            return cached

    if abs_path.lower().endswith(".parquet"):
        arrow_table = pl.read_parquet(abs_path).to_arrow()
    else:
        from app.services.data_processing import load_dataframe
        arrow_table = load_dataframe(abs_path).to_arrow()

    nbytes = getattr(arrow_table, "nbytes", 0)
    if nbytes <= _MAX_CACHE_TABLE_BYTES:
        with _ARROW_CACHE_LOCK:
            _ARROW_CACHE[cache_key] = arrow_table
            _ARROW_CACHE.move_to_end(cache_key)
            while len(_ARROW_CACHE) > _MAX_CACHED_DATASETS:
                _ARROW_CACHE.popitem(last=False)

    return arrow_table


def _serialize_cell(value: Any) -> Any:
    """Normalize Polars/DuckDB scalar values into clean JSON-safe Python primitives."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None
    if isinstance(value, (datetime, date, dt_time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return [_serialize_cell(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _serialize_cell(v) for k, v in value.items()}
    return value


def _serialize_records(df: pl.DataFrame) -> List[Dict[str, Any]]:
    raw = df.to_dicts()
    return [{str(k): _serialize_cell(v) for k, v in row.items()} for row in raw]


def _cursor_to_polars(cursor: Any) -> pl.DataFrame:
    """
    Safely convert a DuckDB query result cursor to a Polars DataFrame.
    Prefers native .pl() / zero-copy Arrow conversion with graceful fallbacks.
    """
    if hasattr(cursor, "pl"):
        try:
            return cursor.pl()
        except Exception:
            pass

    if hasattr(cursor, "to_arrow_table"):
        try:
            return pl.from_arrow(cursor.to_arrow_table())
        except Exception:
            pass

    if hasattr(cursor, "arrow"):
        try:
            arrow_obj = cursor.arrow()
            if hasattr(arrow_obj, "read_all"):
                arrow_obj = arrow_obj.read_all()
            return pl.from_arrow(arrow_obj)
        except Exception:
            pass

    if hasattr(cursor, "fetch_arrow_table"):
        try:
            return pl.from_arrow(cursor.fetch_arrow_table())
        except Exception:
            pass

    if hasattr(cursor, "df"):
        try:
            return pl.from_pandas(cursor.df())
        except Exception:
            pass

    if hasattr(cursor, "fetchdf"):
        try:
            return pl.from_pandas(cursor.fetchdf())
        except Exception:
            pass

    if hasattr(cursor, "fetchall") and hasattr(cursor, "description") and cursor.description:
        cols = [desc[0] for desc in cursor.description]
        return pl.DataFrame(cursor.fetchall(), schema=cols)

    return pl.DataFrame()


def format_sql_query(sql: str) -> str:
    """Format a DuckDB SQL statement cleanly using sqlglot AST pretty-printing."""
    cleaned = (sql or "").strip()
    if not cleaned:
        return ""
    try:
        formatted = sqlglot.transpile(cleaned, read="duckdb", write="duckdb", pretty=True)
        if formatted and len(formatted) == 1 and formatted[0].strip():
            trailing_semi = ";" if cleaned.rstrip().endswith(";") else ""
            return formatted[0].strip() + trailing_semi
    except Exception:
        pass
    return cleaned


class DuckDBSecurityError(Exception):
    """Raised when a query attempts a non-read-only operation or administrative command."""


class DuckDBAnalyticalSession:
    """
    Manages an ephemeral, in-process DuckDB session for ad-hoc analytical queries.
    Enforces strict read-only execution, memory caps, and thread limits.
    """

    DISALLOWED_KEYWORDS = {
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "MERGE",
        "CREATE", "TRUNCATE", "RENAME", "ATTACH", "DETACH",
        "PRAGMA", "COPY", "EXPORT", "INSTALL", "LOAD", "SET",
        "READ_CSV", "READ_CSV_AUTO", "READ_PARQUET", "READ_JSON",
        "READ_JSON_AUTO", "READ_NDJSON", "READ_NDJSON_AUTO", "READ_BLOB", "READ_TEXT",
        "SCAN_PARQUET", "SCAN_CSV", "GLOB", "HTTPFS", "POSTGRES_SCAN", "SQLITE_SCAN",
        "ICU_SORT_KEY", "PARQUET_SCAN", "EXPORT_DATABASE", "WRITE_PARQUET",
    }

    DISALLOWED_FUNCTIONS = {
        "READ_CSV", "READ_CSV_AUTO", "READ_PARQUET", "READ_JSON",
        "READ_JSON_AUTO", "READ_NDJSON", "READ_NDJSON_AUTO", "READ_BLOB", "READ_TEXT",
        "SCAN_PARQUET", "SCAN_CSV", "PARQUET_SCAN", "GLOB", "HTTPFS", "POSTGRES_SCAN",
        "SQLITE_SCAN", "ICU_SORT_KEY", "EXPORT_DATABASE", "WRITE_PARQUET",
    }

    def __init__(self, memory_limit: str = "2GB", threads: int = 4):
        self.conn = duckdb.connect(database=":memory:")
        self.conn.execute(f"SET memory_limit = '{memory_limit}';")
        self.conn.execute(f"SET threads = {threads};")
        self.conn.execute("SET enable_external_access = false;")
        self._registered_tables: set[str] = set()
        self._arrow_tables: Dict[str, Any] = {}

    def register_parquet(self, table_name: str, file_path: str) -> None:
        """Register a Parquet file as an analytical view via LRU-cached Apache Arrow."""
        self.register_file(table_name, file_path)

    def register_file(self, table_name: str, file_path: str) -> None:
        """Register any supported dataset artifact via cached zero-copy Arrow."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset artifact file not found: {file_path}")
        safe_table_name = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
        arrow_table = _load_cached_arrow_table(file_path)
        self._arrow_tables[safe_table_name] = arrow_table
        self.conn.register(safe_table_name, arrow_table)
        self._registered_tables.add(safe_table_name)

    def register_polars(self, table_name: str, df: pl.DataFrame) -> None:
        """Zero-copy view registration from a Polars DataFrame via Apache Arrow."""
        arrow_table = df.to_arrow()
        safe_table_name = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
        self._arrow_tables[safe_table_name] = arrow_table
        self.conn.register(safe_table_name, arrow_table)
        self._registered_tables.add(safe_table_name)
        logger.info("Registered in-memory Polars DataFrame view '%s' (%d rows)", safe_table_name, df.height)

    def _assert_single_statement_tokens(self, clean_sql: str) -> List[sqlglot.Token]:
        """Lex SQL using sqlglot tokenizer so string literals with ';' are never misclassified."""
        tokens = sqlglot.tokenize(clean_sql, read="duckdb")
        non_trivial = [t for t in tokens if t.token_type not in (TokenType.SPACE, TokenType.BREAK)]
        if not non_trivial:
            raise DuckDBSecurityError("SQL query cannot be empty.")

        for idx, tok in enumerate(non_trivial):
            if tok.token_type == TokenType.SEMICOLON and idx < len(non_trivial) - 1:
                raise DuckDBSecurityError("Multiple stacked SQL statements are forbidden.")
        return non_trivial

    def _validate_explain_command(self, raw_sql: str) -> None:
        """Validate EXPLAIN / EXPLAIN ANALYZE / EXPLAIN (FORMAT ...) by inspecting the inner query."""
        stripped = raw_sql.strip().rstrip(";").strip()
        remainder = re.sub(
            r"^EXPLAIN\s+(?:ANALYZE\s+)?(?:\([^)]*\)\s+)?",
            "",
            stripped,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        if not remainder or remainder.upper() == stripped.upper():
            raise DuckDBSecurityError("EXPLAIN statement must target a valid read-only SQL query.")
        self.validate_sql_safety(remainder)

    def validate_sql_safety(self, sql: str) -> None:
        """
        AST-backed strict read-only SQL validation via sqlglot.
        Avoids false positives on column names or string literals while blocking
        statement stacking, DDL/DML, extensions, and external file functions.
        """
        clean_sql = (sql or "").strip()
        if not clean_sql:
            raise DuckDBSecurityError("SQL query cannot be empty.")

        tokens = self._assert_single_statement_tokens(clean_sql)
        first_word = tokens[0].text.upper()

        if first_word == "EXPLAIN":
            self._validate_explain_command(clean_sql)
            return

        allowed_roots = (
            exp.Select,
            exp.Union,
            exp.Intersect,
            exp.Except,
            exp.Subquery,
            exp.Values,
            exp.Describe,
            exp.Show,
        )

        disallowed_nodes = tuple(
            node_cls
            for node_cls in (
                getattr(exp, "Drop", None),
                getattr(exp, "Delete", None),
                getattr(exp, "Update", None),
                getattr(exp, "Insert", None),
                getattr(exp, "Alter", None),
                getattr(exp, "Create", None),
                getattr(exp, "Merge", None),
                getattr(exp, "TruncateTable", None),
                getattr(exp, "Copy", None),
                getattr(exp, "Attach", None),
                getattr(exp, "Detach", None),
                getattr(exp, "Pragma", None),
                getattr(exp, "Set", None),
                getattr(exp, "Use", None),
                getattr(exp, "Transaction", None),
                getattr(exp, "Commit", None),
                getattr(exp, "Rollback", None),
            )
            if node_cls is not None
        )

        try:
            expressions = [e for e in sqlglot.parse(clean_sql, read="duckdb") if e is not None]
        except sqlglot.errors.ParseError as pe:
            # Fallback token-level check only if DuckDB-specific read syntax isn't recognized by sqlglot
            if first_word not in ("SELECT", "WITH", "DESCRIBE", "SHOW", "VALUES", "FROM", "SUMMARIZE"):
                raise DuckDBSecurityError(
                    f"Security Violation: Statement type '{first_word}' is not permitted. Only analytical read queries are allowed."
                ) from pe
            for tok in tokens:
                if tok.token_type not in (TokenType.STRING, TokenType.NUMBER) and not tok.text.startswith('"'):
                    if tok.text.upper() in self.DISALLOWED_KEYWORDS:
                        raise DuckDBSecurityError(
                            f"Security Violation: Forbidden keyword or function detected: {tok.text.upper()}"
                        ) from pe
            return

        if len(expressions) > 1:
            raise DuckDBSecurityError("Multiple stacked SQL statements are forbidden.")
        if not expressions:
            raise DuckDBSecurityError("SQL query cannot be empty.")

        root = expressions[0]
        if isinstance(root, exp.Command):
            cmd_name = str(root.this or first_word).upper()
            if cmd_name not in ("DESCRIBE", "SHOW", "SUMMARIZE"):
                raise DuckDBSecurityError(
                    f"Security Violation: Statement type '{cmd_name}' is not permitted. Only analytical read queries are allowed."
                )
            cmd_tokens = sqlglot.tokenize(str(root.expression or ""), read="duckdb")
            for tok in cmd_tokens:
                if tok.token_type != TokenType.STRING and tok.text.upper() in self.DISALLOWED_KEYWORDS:
                    raise DuckDBSecurityError(f"Security Violation: Forbidden keyword detected: {tok.text.upper()}")
            return

        if not isinstance(root, allowed_roots):
            raise DuckDBSecurityError(
                f"Security Violation: Statement type '{first_word}' is not permitted. Only analytical read queries are allowed."
            )

        for node in root.walk():
            if disallowed_nodes and isinstance(node, disallowed_nodes):
                raise DuckDBSecurityError(
                    f"Security Violation: Mutating or administrative operation '{type(node).__name__.upper()}' is forbidden."
                )
            if isinstance(node, exp.Command):
                raise DuckDBSecurityError("Security Violation: Nested command syntax is forbidden.")
            if isinstance(node, (exp.Func, exp.Anonymous)):
                candidates = (
                    getattr(node, "name", None),
                    node.sql_name() if hasattr(node, "sql_name") else None,
                )
                for cand in candidates:
                    fn_name = str(cand or "").strip().upper()
                    if fn_name and fn_name in self.DISALLOWED_FUNCTIONS:
                        raise DuckDBSecurityError(
                            f"Security Violation: Forbidden keyword or table/file function '{fn_name}'."
                        )
            if isinstance(node, exp.Literal) and node.is_string:
                lit_val = str(node.this or "")
                if _FILE_PATH_PATTERN.search(lit_val):
                    raise DuckDBSecurityError(
                        "Security Violation: Direct file path references are forbidden in analytical queries."
                    )

    def _substitute_template_params(
        self, sql: str, params: Optional[Dict[str, Any]]
    ) -> Tuple[str, List[Any]]:
        """Replace {{ variable_name }} occurrences with positional '?' bindings safely."""
        if not _PARAM_PATTERN.search(sql):
            return sql, []

        param_map = params or {}
        bound: List[Any] = []

        def _replacer(match: re.Match) -> str:
            key = match.group(1)
            if key not in param_map:
                raise DuckDBSecurityError(f"Missing parameter value for '{{{{ {key} }}}}'")
            bound.append(param_map[key])
            return "?"

        return _PARAM_PATTERN.sub(_replacer, sql), bound

    def execute_read_query(
        self, sql: str, params: Optional[List[Any]] = None, max_rows: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Execute read-only SQL and return serialized record dictionaries."""
        self.validate_sql_safety(sql)
        clean_sql = sql.rstrip(";").strip()
        capped_limit = min(max_rows, 1000) if (max_rows is not None and max_rows > 0) else None
        first_token = clean_sql.split()[0].upper()

        if capped_limit and first_token in ("SELECT", "WITH", "VALUES", "FROM"):
            wrapped_sql = f"SELECT * FROM ({clean_sql}) AS _subquery LIMIT {capped_limit}"
        else:
            wrapped_sql = clean_sql

        try:
            cursor = self.conn.execute(wrapped_sql, params) if params else self.conn.execute(wrapped_sql)
            result_pl = _cursor_to_polars(cursor)
            if capped_limit:
                result_pl = result_pl.head(capped_limit)
            return _serialize_records(result_pl)
        except Exception as e:
            logger.error("DuckDB query execution error: %s | Query: %s", e, sql)
            raise

    def execute_studio_query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        max_rows: int = 500,
        max_cap: int = 1000,
    ) -> Dict[str, Any]:
        """
        Execute an interactive Studio query with accurate limit+1 truncation detection,
        optional {{ param }} prepared bindings, and Arrow schema column types.
        """
        t_start = time.perf_counter()
        processed_sql, bound_params = self._substitute_template_params(sql, params)
        self.validate_sql_safety(processed_sql)

        clean_sql = processed_sql.rstrip(";").strip()
        capped_limit = max(1, min(max_rows, max_cap))
        first_token = sqlglot.tokenize(clean_sql, read="duckdb")[0].text.upper()
        supports_subquery_wrap = first_token in ("SELECT", "WITH", "VALUES", "FROM")

        if supports_subquery_wrap:
            wrapped_sql = f"SELECT * FROM ({clean_sql}) AS _subquery LIMIT {capped_limit + 1}"
        else:
            wrapped_sql = clean_sql

        cursor = self.conn.execute(wrapped_sql, bound_params) if bound_params else self.conn.execute(wrapped_sql)
        result_pl = _cursor_to_polars(cursor)
        capped = result_pl.height > capped_limit
        if capped:
            result_pl = result_pl.head(capped_limit)

        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
        records = _serialize_records(result_pl)
        columns = list(result_pl.columns)
        column_types = [str(dtype) for dtype in result_pl.dtypes]

        return {
            "columns": columns,
            "column_types": column_types,
            "records": records,
            "total_returned": len(records),
            "capped": capped,
            "execution_ms": duration_ms,
        }

    def explain_studio_query(
        self, sql: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return DuckDB physical execution plan in structured JSON format."""
        processed_sql, bound_params = self._substitute_template_params(sql, params)
        self.validate_sql_safety(processed_sql)

        clean_sql = processed_sql.rstrip(";").strip()
        # Strip leading EXPLAIN if caller supplied it so we consistently format JSON
        inner_sql = re.sub(
            r"^EXPLAIN\s+(?:ANALYZE\s+)?(?:\([^)]*\)\s+)?",
            "",
            clean_sql,
            count=1,
            flags=re.IGNORECASE,
        ).strip() or clean_sql

        first_token = sqlglot.tokenize(inner_sql, read="duckdb")[0].text.upper()
        explain_sql = (
            f"EXPLAIN (FORMAT JSON) {inner_sql}"
            if first_token in ("SELECT", "WITH", "VALUES", "FROM")
            else f"EXPLAIN {inner_sql}"
        )

        cursor = self.conn.execute(explain_sql, bound_params) if bound_params else self.conn.execute(explain_sql)
        rows = cursor.fetchall()
        plan_nodes: List[Dict[str, Any]] = []
        raw_text = ""

        if rows and len(rows[0]) >= 2:
            raw_text = str(rows[0][1] or "")
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, list):
                    plan_nodes = parsed
            except json.JSONDecodeError:
                plan_nodes = []

        return {
            "sql": inner_sql,
            "plan": plan_nodes,
            "raw_plan": raw_text,
        }

    def iter_export_batches(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        max_rows: int = 50000,
    ) -> Tuple[List[str], Iterable[Dict[str, Any]]]:
        """Execute read-only SQL up to an elevated streaming export ceiling."""
        res = self.execute_studio_query(sql, params=params, max_rows=max_rows, max_cap=50000)
        return res["columns"], res["records"]

    def get_column_profile(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Calculate exact statistical distributions for an existing column."""
        safe_table = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
        schema_rows = self.get_table_schema(safe_table)
        valid_cols = {row.get("column_name"): row.get("column_type") for row in schema_rows}
        if column_name not in valid_cols:
            raise ValueError(f"Column '{column_name}' does not exist in '{safe_table}'.")

        safe_col = column_name.replace('"', '""')
        escaped_col = f'"{safe_col}"'
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
        df = _cursor_to_polars(self.conn.execute(query))
        results = _serialize_records(df)
        profile = results[0] if results else {}
        profile["column"] = column_name
        profile["column_type"] = str(valid_cols[column_name] or "UNKNOWN")
        return profile

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Return column names, types, and nullability for a registered table."""
        safe_table = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
        return _serialize_records(_cursor_to_polars(self.conn.execute(f"DESCRIBE {safe_table};")))

    def get_schema_metadata(self) -> Dict[str, Any]:
        """Return tables, column metadata, and sample values for SQL autocomplete and inspector."""
        tables_meta = []
        for table in self.list_tables():
            safe_table = re.sub(r"[^a-zA-Z0-9_]", "_", table)
            row_cnt = self.conn.execute(f"SELECT COUNT(*) FROM {safe_table};").fetchone()[0]
            desc = self.get_table_schema(table)

            try:
                samples_pl = _cursor_to_polars(self.conn.execute(f"SELECT * FROM {safe_table} LIMIT 3;"))
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
                    "sample_values": [str(_serialize_cell(s)) for s in samples if s is not None],
                })

            tables_meta.append({
                "table_name": table,
                "row_count": int(row_cnt or 0),
                "column_count": len(columns),
                "columns": columns,
            })

        return {
            "engine": "DuckDB-InProcess",
            "total_tables": len(tables_meta),
            "tables": tables_meta,
        }

    def execute_parameterized_query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        max_rows: int = 1000,
    ) -> Dict[str, Any]:
        """Execute a {{ variable }} parameterized query cleanly."""
        try:
            res = self.execute_studio_query(sql, params=params, max_rows=max_rows, max_cap=max_rows)
            return {
                "success": True,
                "sql": sql,
                "duration_ms": res["execution_ms"],
                "row_count": res["total_returned"],
                "columns": res["columns"],
                "column_types": res["column_types"],
                "data": res["records"],
            }
        except DuckDBSecurityError:
            raise
        except Exception as ex:
            logger.error("Parameterized SQL execution failed: %s", ex)
            return {
                "success": False,
                "sql": sql,
                "duration_ms": 0.0,
                "error": str(ex),
                "data": [],
            }

    def list_tables(self) -> List[str]:
        """Return all currently registered views and tables."""
        res = self.conn.execute("SHOW TABLES;").fetchall()
        return [row[0] for row in res]

    def close(self) -> None:
        """Release session resources."""
        try:
            self.conn.close()
        except Exception as e:
            logger.warning("Error closing DuckDB connection: %s", e)
