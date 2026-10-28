"""
Superset Companion Service for GetReport.

Handles communication with Apache Superset:
- Health checking and companion status
- Direct database export of cleaned Polars datasets to Superset
- Registering dataset tables via Superset REST API (/api/v1/dataset/)
- Admin authentication & access token caching
- Scoped Guest Token generation for embedded dashboards
- Safe fallbacks when Superset is offline
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
import httpx
import polars as pl

from app.core.config import settings

logger = logging.getLogger(__name__)


def _polars_to_pg_type(dtype: pl.DataType) -> str:
    """Map Polars DataType to standard PostgreSQL column type."""
    if dtype in (pl.Int8, pl.Int16, pl.Int32):
        return "INTEGER"
    elif dtype == pl.Int64:
        return "BIGINT"
    elif dtype in (pl.Float32, pl.Float64):
        return "DOUBLE PRECISION"
    elif dtype == pl.Boolean:
        return "BOOLEAN"
    elif dtype in (pl.Date,):
        return "DATE"
    elif dtype in (pl.Datetime,):
        return "TIMESTAMP"
    else:
        return "TEXT"


class SupersetService:
    def __init__(self):
        self.base_url = settings.SUPERSET_URL.rstrip("/")
        self.admin_username = settings.SUPERSET_ADMIN_USERNAME
        self.admin_password = settings.SUPERSET_ADMIN_PASSWORD
        self.db_url = settings.SUPERSET_DB_URL
        self._cached_access_token: Optional[str] = None
        self._token_expiry_timestamp: float = 0

    async def check_health(self) -> Dict[str, Any]:
        """Check if the Apache Superset companion container is alive."""
        if not settings.SUPERSET_ENABLED:
            return {
                "enabled": False,
                "status": "disabled",
                "message": "Superset integration is disabled in settings.",
                "superset_url": self.base_url,
            }

        try:
            async with httpx.AsyncClient(timeout=2.5) as client:
                res = await client.get(f"{self.base_url}/health")
                is_ok = res.status_code in (200, 302, 204)
                return {
                    "enabled": True,
                    "status": "online" if is_ok else "unhealthy",
                    "status_code": res.status_code,
                    "superset_url": self.base_url,
                    "message": "Superset companion is online and ready." if is_ok else "Superset returned a non-200 status code.",
                }
        except Exception as e:
            return {
                "enabled": True,
                "status": "offline",
                "superset_url": self.base_url,
                "message": (
                    "Superset companion service is not running. "
                    "Start it via: docker compose -f docker-compose.superset.yml up -d"
                ),
                "error": str(e),
            }

    async def get_admin_access_token(self) -> Optional[str]:
        """Obtain or return cached JWT access token for Superset admin."""
        now = time.time()
        if self._cached_access_token and now < (self._token_expiry_timestamp - 60):
            return self._cached_access_token

        login_url = f"{self.base_url}/api/v1/security/login"
        payload = {
            "username": self.admin_username,
            "password": self.admin_password,
            "provider": "db",
            "refresh": True,
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(login_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    token = data.get("access_token")
                    if token:
                        self._cached_access_token = token
                        self._token_expiry_timestamp = now + 240
                        return token
                logger.warning(
                    f"Superset login failed with status {response.status_code}: {response.text}"
                )
        except Exception as e:
            logger.error(f"Error authenticating with Superset: {e}")

        return None

    def _sync_df_to_postgres(self, table_name: str, df: pl.DataFrame) -> bool:
        """Synchronously write Polars DataFrame into Superset PostgreSQL database."""
        try:
            import psycopg2
            from psycopg2.extras import execute_batch

            conn = psycopg2.connect(self.db_url, connect_timeout=3)
            cur = conn.cursor()

            # Sanitize column names for SQL identifiers
            cols = [re.sub(r"[^a-zA-Z0-9_]", "_", col).lower() for col in df.columns]
            col_defs = []
            for col_name, dtype in zip(cols, df.dtypes):
                pg_type = _polars_to_pg_type(dtype)
                col_defs.append(f'"{col_name}" {pg_type}')

            create_sql = f'CREATE TABLE IF NOT EXISTS public."{table_name}" ({", ".join(col_defs)});'
            cur.execute(create_sql)
            cur.execute(f'TRUNCATE TABLE public."{table_name}";')

            # Prepare insert query
            placeholders = ", ".join(["%s"] * len(cols))
            insert_sql = f'INSERT INTO public."{table_name}" ({", ".join([f"{c}" for c in cols])}) VALUES ({placeholders});'

            # Convert to rows in memory
            rows = df.iter_rows()
            batch_size = 5000
            batch = []
            for row in rows:
                batch.append(row)
                if len(batch) >= batch_size:
                    execute_batch(cur, insert_sql, batch)
                    batch = []
            if batch:
                execute_batch(cur, insert_sql, batch)

            conn.commit()
            cur.close()
            conn.close()
            logger.info(f"Successfully synced {len(df)} rows into Superset PostgreSQL table: {table_name}")
            return True
        except Exception as e:
            logger.warning(f"Could not connect or write to Superset PostgreSQL ({e}). Dataset saved locally.")
            return False

    async def sync_dataset_to_superset(
        self,
        task_id: str,
        df: Optional[pl.DataFrame] = None,
        file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Uploads and registers an audited dataset in Apache Superset:
        1. Loads cleaned DataFrame (from memory or Parquet).
        2. Exports into Superset's PostgreSQL analytical store.
        3. Registers table via Superset REST API (/api/v1/dataset/).
        """
        # Resolve DataFrame
        if df is None:
            if not file_path:
                from app.services.task_manager import title_task_manager
                from app.services.storage import get_storage_provider
                job = title_task_manager.get_job(task_id)
                if job and job.result:
                    file_ref = job.result.get("cleaned_file_ref") or job.result.get("_file_ref")
                    if file_ref:
                        storage = get_storage_provider()
                        file_path = storage.get_absolute_path(file_ref)

            if not file_path:
                from app.services.storage import get_storage_provider
                storage = get_storage_provider()
                cleaned_ref = f"cleaned_{task_id}.parquet"
                abs_path = storage.get_absolute_path(cleaned_ref)
                if os.path.exists(abs_path):
                    file_path = abs_path

            if file_path and os.path.exists(file_path):
                try:
                    df = pl.read_parquet(file_path) if file_path.endswith(".parquet") else pl.read_csv(file_path)
                except Exception as e:
                    logger.error(f"Failed to load dataset for sync: {e}")
                    return {"status": "error", "message": f"Could not read source dataset: {e}"}
            else:
                return {
                    "status": "not_found",
                    "message": "Cleaned dataset not found for this task ID. Run audit analysis first.",
                }

        # Safe table name
        safe_task_id = re.sub(r"[^a-zA-Z0-9_]", "_", task_id)
        table_name = f"dataset_{safe_task_id[:16]}"

        # Write to database
        db_synced = self._sync_df_to_postgres(table_name, df)

        # Check Superset API registration
        token = await self.get_admin_access_token()
        api_registered = False
        dataset_id = None

        if token:
            try:
                dataset_url = f"{self.base_url}/api/v1/dataset/"
                headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
                payload = {
                    "database": 1,
                    "schema": "public",
                    "table_name": table_name,
                }
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(dataset_url, json=payload, headers=headers)
                    if resp.status_code in (200, 201):
                        api_registered = True
                        data = resp.json()
                        dataset_id = data.get("id")
                    elif "already exists" in resp.text:
                        api_registered = True
            except Exception as reg_err:
                logger.warning(f"Failed to register dataset via Superset API: {reg_err}")

        return {
            "status": "synced" if db_synced else "ready",
            "task_id": task_id,
            "table_name": table_name,
            "rows_synced": len(df),
            "columns": df.columns,
            "db_synced": db_synced,
            "api_registered": api_registered,
            "dataset_id": dataset_id,
            "dashboard_id": settings.SUPERSET_DEFAULT_DASHBOARD_ID,
            "message": (
                f"Dataset '{table_name}' with {len(df)} rows prepared for Superset analytics."
                if not db_synced else
                f"Dataset successfully synchronized to Superset table '{table_name}' ({len(df)} rows)."
            ),
        }

    async def create_guest_token(
        self,
        dashboard_id: Optional[str] = None,
        task_id: Optional[str] = None,
        user_info: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Request a scoped Guest Token from Superset to embed a dashboard.
        """
        target_dashboard_id = dashboard_id or settings.SUPERSET_DEFAULT_DASHBOARD_ID
        health = await self.check_health()

        if health.get("status") != "online":
            return {
                "status": "offline",
                "dashboard_id": target_dashboard_id,
                "task_id": task_id,
                "superset_url": self.base_url,
                "guest_token": None,
                "message": health.get("message"),
                "instructions": "Run: docker compose -f docker-compose.superset.yml up -d",
            }

        access_token = await self.get_admin_access_token()
        if not access_token:
            return {
                "status": "auth_error",
                "dashboard_id": target_dashboard_id,
                "task_id": task_id,
                "superset_url": self.base_url,
                "guest_token": None,
                "message": "Failed to authenticate with Superset admin API.",
            }

        guest_token_url = f"{self.base_url}/api/v1/security/guest_token/"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        user_data = user_info or {
            "username": "getreport_guest",
            "first_name": "GetReport",
            "last_name": "Analyst",
        }

        payload = {
            "user": user_data,
            "resources": [{"type": "dashboard", "id": target_dashboard_id}],
            "rls": [],
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.post(guest_token_url, json=payload, headers=headers)
                if res.status_code == 200:
                    token_data = res.json()
                    guest_token = token_data.get("token")
                    return {
                        "status": "online",
                        "dashboard_id": target_dashboard_id,
                        "task_id": task_id,
                        "superset_url": self.base_url,
                        "guest_token": guest_token,
                        "message": "Guest token generated successfully.",
                    }
                logger.error(
                    f"Superset guest token creation failed [{res.status_code}]: {res.text}"
                )
                return {
                    "status": "token_generation_failed",
                    "dashboard_id": target_dashboard_id,
                    "task_id": task_id,
                    "superset_url": self.base_url,
                    "guest_token": None,
                    "message": f"Superset rejected token request ({res.status_code}).",
                    "details": res.text,
                }
        except Exception as e:
            logger.error(f"Error requesting Superset guest token: {e}")
            return {
                "status": "error",
                "dashboard_id": target_dashboard_id,
                "task_id": task_id,
                "superset_url": self.base_url,
                "guest_token": None,
                "message": str(e),
            }



    def compute_aggregation(
        self,
        task_id: str,
        dimension: str,
        metric: Optional[str] = None,
        aggregation: str = "count",
        secondary_dimension: Optional[str] = None,
        limit: int = 15,
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """
        Computes high-speed grouped aggregations over the cleaned dataset for interactive BI charts.
        """
        from app.services.task_manager import title_task_manager
        from app.services.storage import get_storage_provider
        
        job = title_task_manager.get_job(task_id)
        if not job or not job.result:
            raise ValueError(f"Task {task_id} not found or results missing")
        
        file_ref = job.result.get("cleaned_file_ref") or job.result.get("_file_ref")
        if not file_ref:
            raise ValueError(f"Cleaned dataset file not found for task {task_id}")
            
        storage = get_storage_provider()
        file_path = storage.get_absolute_path(file_ref)
        if not os.path.exists(file_path):
            raise ValueError(f"File path does not exist on disk: {file_path}")
            
        df = pl.read_parquet(file_path) if file_path.endswith(".parquet") else pl.read_csv(file_path)
        
        # Validate dimension
        if dimension not in df.columns:
            raise ValueError(f"Dimension '{dimension}' not found in dataset columns")

        # Build aggregation expression
        agg_func = aggregation.lower()
        if metric and metric in df.columns:
            if agg_func == "sum":
                val_expr = pl.col(metric).sum().alias("value")
            elif agg_func in ("avg", "mean"):
                val_expr = pl.col(metric).mean().alias("value")
            elif agg_func == "min":
                val_expr = pl.col(metric).min().alias("value")
            elif agg_func == "max":
                val_expr = pl.col(metric).max().alias("value")
            elif agg_func == "median":
                val_expr = pl.col(metric).median().alias("value")
            else:
                val_expr = pl.col(metric).count().alias("value")
        else:
            metric = "records"
            val_expr = pl.len().alias("value")

        group_cols = [dimension]
        if secondary_dimension and secondary_dimension in df.columns:
            group_cols.append(secondary_dimension)

        grouped = df.filter(pl.col(dimension).is_not_null()).group_by(group_cols).agg(
            val_expr,
            pl.len().alias("count")
        )

        descending = sort_order.lower() == "desc"
        sorted_df = grouped.sort("value", descending=descending).head(limit)

        # Calculate dataset KPI summary
        total_rows = len(df)
        if metric != "records" and metric in df.columns:
            metric_col = df[metric]
            kpis = {
                "metric_name": metric,
                "dimension_name": dimension,
                "total_rows": total_rows,
                "metric_sum": float(metric_col.sum()) if metric_col.is_numeric() else None,
                "metric_mean": round(float(metric_col.mean()), 2) if metric_col.is_numeric() else None,
                "metric_max": float(metric_col.max()) if metric_col.is_numeric() else None,
                "metric_min": float(metric_col.min()) if metric_col.is_numeric() else None,
                "unique_categories": df[dimension].n_unique(),
            }
        else:
            kpis = {
                "metric_name": "Records",
                "dimension_name": dimension,
                "total_rows": total_rows,
                "metric_sum": total_rows,
                "metric_mean": None,
                "metric_max": None,
                "metric_min": None,
                "unique_categories": df[dimension].n_unique(),
            }

        records = sorted_df.to_dicts()
        cleaned_records = []
        for r in records:
            row = {}
            for k, v in r.items():
                if v is None:
                    row[k] = None
                elif isinstance(v, (int, float)):
                    row[k] = round(v, 2) if isinstance(v, float) and abs(v) < 1e12 else (v if v == v else 0)
                else:
                    row[k] = str(v)
            cleaned_records.append(row)

        return {
            "task_id": task_id,
            "dimension": dimension,
            "metric": metric,
            "aggregation": agg_func,
            "secondary_dimension": secondary_dimension,
            "kpis": kpis,
            "records": cleaned_records,
            "total_groups": len(grouped),
            "columns": df.columns,
            "numeric_columns": [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()],
            "categorical_columns": [c for c, dt in zip(df.columns, df.dtypes) if dt in (pl.String, pl.Categorical)],
        }


superset_service = SupersetService()
