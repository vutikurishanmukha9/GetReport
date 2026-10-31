"""
Analytics & Superset BI Companion Routes.

Provides endpoints for:
- Checking Superset companion service health
- Synchronizing uploaded & audited datasets directly into Superset
- Retrieving available embedded dashboards
- Issuing scoped Guest Tokens for @superset-ui/embedded-sdk
- Introspecting schemas with column classification & sample values
- Executing high-speed filtered analytical aggregations with ECharts option trees
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.auth import verify_api_key
from app.core.config import settings
from app.core.limiter import limiter
from app.services.superset_service import superset_service
from app.services.superset_engine import superset_engine, SupersetExploreQuery

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])

ANALYTICS_QUERY_LIMIT = "60/minute"


class GuestTokenRequest(BaseModel):
    dashboard_id: Optional[str] = Field(
        default=None,
        description="The Superset dashboard UUID or integer ID to embed. Defaults to settings.SUPERSET_DEFAULT_DASHBOARD_ID."
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Optional GetReport task ID to associate with the embedded dashboard."
    )
    username: Optional[str] = Field(default="getreport_user", description="Identifier for guest user session")


class GuestTokenResponse(BaseModel):
    status: str
    dashboard_id: str
    task_id: Optional[str] = None
    superset_url: str
    guest_token: Optional[str] = None
    message: str
    instructions: Optional[str] = None


class CompanionStatusResponse(BaseModel):
    enabled: bool
    status: str
    superset_url: str
    message: str
    status_code: Optional[int] = None
    instructions: Optional[str] = None


class DatasetSyncResponse(BaseModel):
    status: str
    task_id: str
    table_name: Optional[str] = None
    rows_synced: Optional[int] = None
    columns: Optional[List[str]] = None
    db_synced: bool
    api_registered: bool
    dataset_id: Optional[int] = None
    dashboard_id: Optional[str] = None
    message: str


@router.get("/status", response_model=CompanionStatusResponse)
async def get_analytics_status(
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    """
    Check the connectivity status of the Apache Superset BI companion service.
    """
    health = await superset_service.check_health()
    instructions = (
        "Run `docker compose -f docker/docker-compose.superset.yml up -d` in the project root to start the companion."
        if health.get("status") != "online" else None
    )
    return CompanionStatusResponse(
        enabled=health.get("enabled", True),
        status=health.get("status", "unknown"),
        superset_url=health.get("superset_url", settings.SUPERSET_URL),
        message=health.get("message", ""),
        status_code=health.get("status_code"),
        instructions=instructions,
    )


@router.post("/sync/{task_id}", response_model=DatasetSyncResponse)
@limiter.limit("20/minute")
async def sync_uploaded_dataset(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Synchronize an uploaded and audited GetReport dataset into Apache Superset.
    Exports the cleaned dataset to the analytical database and registers the table.
    """
    res = await superset_service.sync_dataset_to_superset(task_id=task_id)
    if res.get("status") == "error":
        raise HTTPException(status_code=500, detail=res.get("message", "Sync failed"))
    if res.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=res.get("message", "Dataset not found"))

    return DatasetSyncResponse(
        status=res.get("status", "synced"),
        task_id=task_id,
        table_name=res.get("table_name"),
        rows_synced=res.get("rows_synced"),
        columns=res.get("columns"),
        db_synced=res.get("db_synced", False),
        api_registered=res.get("api_registered", False),
        dataset_id=res.get("dataset_id"),
        dashboard_id=res.get("dashboard_id", settings.SUPERSET_DEFAULT_DASHBOARD_ID),
        message=res.get("message", "Dataset synced"),
    )


@router.post("/guest-token", response_model=GuestTokenResponse)
@limiter.limit("30/minute")
async def generate_guest_token(
    request: Request,
    payload: Optional[GuestTokenRequest] = None,
    _auth: None = Depends(verify_api_key),
):
    """
    Generate a secure Guest Token for embedding a Superset dashboard in the GetReport UI.
    If Superset is offline, returns an informative status payload rather than raising an uncaught exception.
    """
    req_payload = payload or GuestTokenRequest()
    user_info = {
        "username": req_payload.username or "getreport_user",
        "first_name": "GetReport",
        "last_name": "Viewer",
    }

    result = await superset_service.create_guest_token(
        dashboard_id=req_payload.dashboard_id,
        task_id=req_payload.task_id,
        user_info=user_info,
    )

    return GuestTokenResponse(
        status=result.get("status", "error"),
        dashboard_id=result.get("dashboard_id", settings.SUPERSET_DEFAULT_DASHBOARD_ID),
        task_id=result.get("task_id"),
        superset_url=result.get("superset_url", settings.SUPERSET_URL),
        guest_token=result.get("guest_token"),
        message=result.get("message", ""),
        instructions=result.get("instructions"),
    )


@router.get("/dashboards")
async def list_available_dashboards(
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    """
    Returns configured and certified dashboards available for viewing.
    """
    return {
        "dashboards": [
            {
                "id": settings.SUPERSET_DEFAULT_DASHBOARD_ID,
                "title": "Executive Data Intelligence & BI Overview",
                "description": "High-level visual breakdown of key metrics, distributions, and trends.",
                "certified": True,
            }
        ]
    }


class AggregateRequest(BaseModel):
    dimension: str = Field(..., description="Categorical or date column name to group by")
    metric: Optional[str] = Field(default=None, description="Numeric column to aggregate. Defaults to row count.")
    aggregation: str = Field(default="count", description="Aggregation method: count, sum, avg, min, max, median")
    secondary_dimension: Optional[str] = Field(default=None, description="Optional secondary grouping column")
    limit: int = Field(default=15, ge=1, le=100, description="Top N limit")
    sort_order: str = Field(default="desc", description="Sort order: desc or asc")


@router.get("/datasets")
async def list_recent_datasets(
    request: Request,
    limit: int = 10,
    _auth: None = Depends(verify_api_key),
):
    """
    Returns recent audited datasets from GetReport jobs for interactive BI exploration.
    """
    from app.db import get_db_connection
    from app.services.task_manager import title_task_manager

    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT task_id, filename, created_at FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            ("completed", limit)
        ).fetchall()

    datasets = []
    for r in rows:
        item = dict(r) if hasattr(r, "keys") else {"task_id": r[0], "filename": r[1], "created_at": r[2]}
        job = title_task_manager.get_job(item["task_id"])
        if job and job.result:
            info = job.result.get("info", {})
            item["rows"] = info.get("rows", 0)
            item["columns_count"] = len(info.get("columns", []))
            item["columns"] = info.get("columns", [])
            item["numeric_columns"] = [
                c for c in info.get("columns", [])
                if any(t in (info.get("dtypes", {}).get(c, "")).lower() for t in ("int", "float", "double", "num"))
            ]
            item["categorical_columns"] = [
                c for c in info.get("columns", [])
                if c not in item["numeric_columns"]
            ]
        datasets.append(item)

    return {"datasets": datasets}


@router.post("/aggregate/{task_id}")
@limiter.limit(ANALYTICS_QUERY_LIMIT)
async def aggregate_dataset_metrics(
    request: Request,
    task_id: str,
    body: AggregateRequest,
    _auth: None = Depends(verify_api_key),
):
    """
    Computes high-speed grouped aggregations over the cleaned dataset for interactive BI charts.
    """
    try:
        res = superset_service.compute_aggregation(
            task_id=task_id,
            dimension=body.dimension,
            metric=body.metric,
            aggregation=body.aggregation,
            secondary_dimension=body.secondary_dimension,
            limit=body.limit,
            sort_order=body.sort_order,
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Aggregation error for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Aggregation calculation failed: {e}")


@router.get("/explore/{task_id}/schema")
@limiter.limit(ANALYTICS_QUERY_LIMIT)
async def get_superset_dataset_schema(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Introspects the uploaded dataset using Apache Superset's column and data type classification.
    """
    try:
        schema = superset_engine.introspect_dataset(task_id)
        return schema
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error introspecting dataset for Superset Explore {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explore/query")
@limiter.limit(ANALYTICS_QUERY_LIMIT)
async def execute_superset_explore_query(
    request: Request,
    query: SupersetExploreQuery,
    _auth: None = Depends(verify_api_key),
):
    """
    Executes a Superset Explore analytical query on the uploaded dataset
    with optional filters and temporal bucketing, returning ECharts visualization options and KPIs.
    """
    try:
        result = superset_engine.execute_explore_query(query)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error executing Superset Explore query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explore query failed: {e}")
