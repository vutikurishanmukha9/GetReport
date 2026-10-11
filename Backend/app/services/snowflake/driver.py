"""
Snowflake Ingestion Driver Interface and Implementations.
Includes RealSnowflakeDriver (production with snowflake-connector-python)
and MockSnowflakeDriver (for isolated automated testing and sandbox environments).
"""
from __future__ import annotations

import abc
import datetime
import decimal
import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pyarrow as pa

from app.services.snowflake.query_guard import quote_ident
from app.services.snowflake.schema import compile_target_arrow_schema

logger = logging.getLogger(__name__)


class SnowflakeIngestDriver(abc.ABC):
    """Abstract interface exposing the Snowflake ingestion lifecycle."""

    @abc.abstractmethod
    def test_connection(self, config: Dict[str, Any]) -> bool:
        """Verifies authentication and warehouse availability."""
        ...

    @abc.abstractmethod
    def discover_databases(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Lists available databases."""
        ...

    @abc.abstractmethod
    def discover_schemas(self, config: Dict[str, Any], database_name: str) -> List[Dict[str, Any]]:
        """Lists schemas in database."""
        ...

    @abc.abstractmethod
    def discover_objects(
        self,
        config: Dict[str, Any],
        database_name: str,
        schema_name: str,
        limit: int = 500,
        from_last: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lists tables and views in schema."""
        ...

    @abc.abstractmethod
    def preview_query(
        self,
        config: Dict[str, Any],
        query: str,
        limit: int = 100,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Returns rows (as dicts) and column names for preview."""
        ...

    @abc.abstractmethod
    def submit_query_async(self, config: Dict[str, Any], query: str) -> str:
        """Asynchronously executes query and returns query_id (sfqid)."""
        ...

    @abc.abstractmethod
    def is_query_running(self, config: Dict[str, Any], query_id: str) -> bool:
        """Returns True if the async query is still in progress."""
        ...

    @abc.abstractmethod
    def abort_query(self, config: Dict[str, Any], query_id: str) -> bool:
        """Aborts an active running query."""
        ...

    @abc.abstractmethod
    def fetch_arrow_batches(
        self,
        config: Dict[str, Any],
        query_id: str,
    ) -> Tuple[Iterator[pa.RecordBatch], Sequence[Any]]:
        """Returns generator of Arrow RecordBatches and cursor.description."""
        ...


class RealSnowflakeDriver(SnowflakeIngestDriver):
    """Production driver utilizing snowflake-connector-python with single-threaded prefetch."""

    def _create_connection(self, config: Dict[str, Any], auto_connect: bool = True):
        import snowflake.connector

        conn_params = {
            "account": config["account"],
            "user": config["user"],
            "warehouse": config.get("warehouse"),
            "database": config.get("database"),
            "schema": config.get("schema"),
            "role": config.get("role"),
            "client_prefetch_threads": 1,  # Memory-constrained container safety
            "query_tag": "getreport-ingest",
        }

        if config.get("private_key_der"):
            conn_params["private_key"] = config["private_key_der"]
            conn_params["authenticator"] = "snowflake_jwt"
        elif config.get("password"):
            conn_params["password"] = config["password"]
            conn_params["authenticator"] = "snowflake"
        else:
            raise ValueError("Authentication credentials (private key or password) are required.")

        # Clean optional None params
        conn_params = {k: v for k, v in conn_params.items() if v is not None}
        return snowflake.connector.connect(**conn_params)

    def test_connection(self, config: Dict[str, Any]) -> bool:
        try:
            with self._create_connection(config) as conn:
                cur = conn.cursor()
                cur.execute("SELECT CURRENT_VERSION(), CURRENT_WAREHOUSE()")
                return True
        except Exception as e:
            logger.error("Snowflake connection test failed: %s", e)
            raise

    def discover_databases(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        with self._create_connection(config) as conn:
            cur = conn.cursor()
            cur.execute("SHOW DATABASES;")
            rows = cur.fetchall()
            # SHOW DATABASES returns: (created_on, name, is_default, is_current, origin, owner, comment, ...)
            return [{"name": r[1], "owner": r[5]} for r in rows]

    def discover_schemas(self, config: Dict[str, Any], database_name: str) -> List[Dict[str, Any]]:
        with self._create_connection(config) as conn:
            cur = conn.cursor()
            cur.execute(f"SHOW SCHEMAS IN DATABASE {quote_ident(database_name)};")
            rows = cur.fetchall()
            return [{"name": r[1], "database": database_name, "owner": r[5]} for r in rows]

    def discover_objects(
        self,
        config: Dict[str, Any],
        database_name: str,
        schema_name: str,
        limit: int = 500,
        from_last: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        schema_ident = f"{quote_ident(database_name)}.{quote_ident(schema_name)}"
        objects: List[Dict[str, Any]] = []

        with self._create_connection(config) as conn:
            cur = conn.cursor()
            # 1. Fetch tables
            tbl_sql = f"SHOW TABLES IN SCHEMA {schema_ident} LIMIT {limit}"
            if from_last:
                tbl_sql += f" FROM '{from_last}'"
            cur.execute(tbl_sql)
            for r in cur.fetchall():
                # SHOW TABLES returns: (created_on, name, database_name, schema_name, kind, comment, cluster_by, rows, bytes, owner, ...)
                objects.append({
                    "name": r[1],
                    "type": "TABLE",
                    "rows": r[7],
                    "bytes": r[8],
                })

            # 2. Fetch views
            view_sql = f"SHOW VIEWS IN SCHEMA {schema_ident} LIMIT {limit}"
            cur.execute(view_sql)
            for r in cur.fetchall():
                objects.append({
                    "name": r[1],
                    "type": "VIEW",
                    "rows": None,
                    "bytes": None,
                })

        return objects

    def preview_query(
        self,
        config: Dict[str, Any],
        query: str,
        limit: int = 100,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        with self._create_connection(config) as conn:
            cur = conn.cursor()
            # Wrap in subquery with explicit limit for safety
            wrapped_sql = f"SELECT * FROM ({query}) AS preview_src LIMIT {limit}"
            cur.execute(wrapped_sql)
            cols = [desc[0] for desc in cur.description]
            raw_rows = cur.fetchall()
            records = [dict(zip(cols, row)) for row in raw_rows]
            return records, cols

    def submit_query_async(self, config: Dict[str, Any], query: str) -> str:
        conn = self._create_connection(config)
        cur = conn.cursor()
        cur.execute_async(query)
        sfqid = cur.sfqid
        return sfqid

    def is_query_running(self, config: Dict[str, Any], query_id: str) -> bool:
        with self._create_connection(config) as conn:
            status = conn.get_query_status(query_id)
            return conn.is_still_running(status)

    def abort_query(self, config: Dict[str, Any], query_id: str) -> bool:
        try:
            with self._create_connection(config) as conn:
                cur = conn.cursor()
                cur.abort_query(query_id)
                return True
        except Exception as e:
            logger.warning("Failed to abort Snowflake query %s: %s", query_id, e)
            return False

    def fetch_arrow_batches(
        self,
        config: Dict[str, Any],
        query_id: str,
    ) -> Tuple[Iterator[pa.RecordBatch], Sequence[Any]]:
        conn = self._create_connection(config)
        cur = conn.cursor()
        cur.get_results_from_sfqid(query_id)
        desc = cur.description
        batches = cur.fetch_arrow_batches()
        return batches, desc


class MockSnowflakeDriver(SnowflakeIngestDriver):
    """
    Mock driver for hermetic unit testing, sandbox demos, and CI environments.
    Simulates databases, schemas, tables, async polling, and PyArrow record batches.
    """

    def __init__(self):
        self._active_queries: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    def test_connection(self, config: Dict[str, Any]) -> bool:
        if config.get("account") == "invalid-account":
            raise ConnectionError("Account could not be resolved.")
        return True

    def discover_databases(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"name": "ANALYTICS_DB", "owner": "SYSADMIN"},
            {"name": "RAW_VAULT_DB", "owner": "DATA_ENGINEER"},
        ]

    def discover_schemas(self, config: Dict[str, Any], database_name: str) -> List[Dict[str, Any]]:
        return [
            {"name": "PUBLIC", "database": database_name, "owner": "SYSADMIN"},
            {"name": "FINANCE", "database": database_name, "owner": "FINANCE_ROLE"},
        ]

    def discover_objects(
        self,
        config: Dict[str, Any],
        database_name: str,
        schema_name: str,
        limit: int = 500,
        from_last: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        all_objs = [
            {"name": "TRANSACTIONS", "type": "TABLE", "rows": 15000, "bytes": 4500000, "database": database_name, "schema": schema_name},
            {"name": "CUSTOMER_PROFILES", "type": "TABLE", "rows": 5000, "bytes": 1200000, "database": database_name, "schema": schema_name},
            {"name": "V_DAILY_METRICS", "type": "VIEW", "rows": None, "bytes": None, "database": database_name, "schema": schema_name},
        ]
        if from_last:
            idx = next((i for i, o in enumerate(all_objs) if o["name"] == from_last), -1)
            if idx != -1:
                all_objs = all_objs[idx + 1:]
            else:
                all_objs = []
        return all_objs[:limit]

    def preview_query(
        self,
        config: Dict[str, Any],
        query: str,
        limit: int = 100,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        cols = ["id", "customer_name", "amount", "is_active", "created_at"]
        rows = [
            {
                "id": i + 1,
                "customer_name": f"Enterprise Client {i + 1}",
                "amount": float(100.50 + i * 25.0),
                "is_active": i % 2 == 0,
                "created_at": "2026-10-10 12:00:00",
            }
            for i in range(min(limit, 5))
        ]
        return rows, cols

    def submit_query_async(self, config: Dict[str, Any], query: str) -> str:
        self._counter += 1
        query_id = f"sfqid-mock-{self._counter}-{int(time.time())}"
        self._active_queries[query_id] = {
            "query": query,
            "status": "RUNNING",
            "created_at": time.time(),
        }
        return query_id

    def is_query_running(self, config: Dict[str, Any], query_id: str) -> bool:
        q = self._active_queries.get(query_id)
        if not q or q["status"] == "ABORTED":
            return False
        # Complete execution after 1 second
        if time.time() - q["created_at"] > 0.5:
            q["status"] = "SUCCESS"
            return False
        return True

    def abort_query(self, config: Dict[str, Any], query_id: str) -> bool:
        if query_id in self._active_queries:
            self._active_queries[query_id]["status"] = "ABORTED"
            return True
        return False

    def fetch_arrow_batches(
        self,
        config: Dict[str, Any],
        query_id: str,
    ) -> Tuple[Iterator[pa.RecordBatch], Sequence[Any]]:
        # Cursor description tuple: (name, type_code, display_size, internal_size, precision, scale, null_ok)
        desc = [
            ("TRANSACTION_ID", 0, None, None, 38, 0, False),  # FIXED int64
            ("MERCHANT_NAME", 2, None, None, None, None, True),  # TEXT
            ("AMOUNT", 0, None, None, 18, 2, False),  # FIXED decimal128(18, 2)
            ("FEE", 1, None, None, None, None, True),  # REAL float64
            ("SETTLED_DATE", 3, None, None, None, None, True),  # DATE
            ("IS_VERIFIED", 13, None, None, None, None, False),  # BOOLEAN
            ("EVENT_TIMESTAMP", 8, None, None, None, None, True),  # TIMESTAMP_NTZ
        ]

        def _generator() -> Iterator[pa.RecordBatch]:
            # Generate 2 synthetic Arrow batches
            for b_idx in range(2):
                n_rows = 50
                batch_data = {
                    "TRANSACTION_ID": pa.array([1000 + b_idx * n_rows + i for i in range(n_rows)], type=pa.int64()),
                    "MERCHANT_NAME": pa.array([f"Vendor {i % 10}" for i in range(n_rows)], type=pa.string()),
                    "AMOUNT": pa.array([decimal.Decimal(f"{100 + i}.50") for i in range(n_rows)], type=pa.decimal128(18, 2)),
                    "FEE": pa.array([float(1.5 + (i % 5)) for i in range(n_rows)], type=pa.float64()),
                    "SETTLED_DATE": pa.array([datetime.date(2026, 10, 1) + datetime.timedelta(days=i % 10) for i in range(n_rows)], type=pa.date32()),
                    "IS_VERIFIED": pa.array([i % 2 == 0 for i in range(n_rows)], type=pa.bool_()),
                    "EVENT_TIMESTAMP": pa.array([datetime.datetime(2026, 10, 10, 12, 0, 0) + datetime.timedelta(seconds=i * 60) for i in range(n_rows)], type=pa.timestamp("us")),
                }
                yield pa.RecordBatch.from_pydict(batch_data)

        return _generator(), desc


_global_driver: Optional[SnowflakeIngestDriver] = None


def get_snowflake_driver(use_mock: bool = False) -> SnowflakeIngestDriver:
    """Returns the singleton driver instance (real or mock)."""
    global _global_driver
    if use_mock:
        return MockSnowflakeDriver()
    if _global_driver is None:
        try:
            import snowflake.connector
            _global_driver = RealSnowflakeDriver()
        except ImportError:
            logger.info("Snowflake connector not available in current environment; using MockSnowflakeDriver.")
            _global_driver = MockSnowflakeDriver()
    return _global_driver
