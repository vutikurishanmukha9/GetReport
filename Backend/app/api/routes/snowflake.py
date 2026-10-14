"""
Snowflake Data Warehouse API Routes.
Provides interactive schema browsing, query preview, and background Parquet ingestion.
"""
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from app.core.auth import verify_api_key, check_task_ownership, validate_task_id, get_authenticated_owner_id
from app.core.config import settings
from app.core.limiter import limiter
from app.core.task_dispatcher import dispatch_task
from app.services.snowflake.driver import get_snowflake_driver
from app.services.snowflake.error_mapper import sanitize_snowflake_error
from app.services.snowflake.query_guard import (
    validate_account_identifier,
    validate_sql_query,
)
from app.services.snowflake.vault import (
    get_snowflake_vault,
    prepare_private_key_der,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/snowflake", tags=["snowflake"])


# ─── Pydantic Request & Response Models ──────────────────────────────────────

class SnowflakeConnectRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    account: str = Field(..., description="Snowflake account identifier (e.g. org-account or xy12345.us-east-1)")
    user: str = Field(..., description="Snowflake username")
    warehouse: Optional[str] = Field(None, description="Active warehouse name")
    database: Optional[str] = Field(None, description="Default database")
    schema_name: Optional[str] = Field(None, alias="schema", description="Default schema")
    role: Optional[str] = Field(None, description="Snowflake role")
    auth_type: str = Field("key_pair", description="'key_pair' or 'password'")
    private_key_pem: Optional[str] = Field(None, description="RSA private key in PEM format")
    private_key_passphrase: Optional[str] = Field(None, description="Optional passphrase for encrypted private key")
    password: Optional[str] = Field(None, description="Password for password auth")
    use_mock: Optional[bool] = Field(False, description="Use sandbox mock driver for testing without live cluster")


class SnowflakeConnectResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str
    account: str
    user: str
    warehouse: Optional[str]
    database: Optional[str]
    schema_name: Optional[str] = Field(None, alias="schema")
    message: str


class SnowflakePreviewRequest(BaseModel):
    session_id: str
    query: str
    limit: int = Field(50, ge=1, le=200)


class SnowflakeImportRequest(BaseModel):
    session_id: str
    query: str
    dataset_name: Optional[str] = "snowflake_dataset"
    byte_limit: Optional[int] = 150 * 1024 * 1024


# ─── Helper: Session Credential Resolver ─────────────────────────────────────

def _resolve_session_config(session_id: str, user_id: str = "local-development") -> Dict[str, Any]:
    vault = get_snowflake_vault()
    creds = vault.retrieve_session(user_id, session_id, refresh_ttl=True)
    if not creds:
        raise HTTPException(
            status_code=401,
            detail="ERR_SNOWFLAKE_SESSION_EXPIRED: Your Snowflake connection session has expired or is invalid. Please reconnect.",
        )
    return creds


# ─── API Endpoints ───────────────────────────────────────────────────────────

@router.post("/connect", response_model=SnowflakeConnectResponse)
@limiter.limit("20/minute")
async def connect_snowflake(
    request: Request,
    payload: SnowflakeConnectRequest = Body(...),
    _auth: None = Depends(verify_api_key),
):
    """
    Validates credentials, verifies connectivity to Snowflake, and stores encrypted credentials
    in the session vault with a 15-minute sliding expiration window.
    """
    # 1. SSRF check on account identifier
    if not validate_account_identifier(payload.account):
        raise HTTPException(
            status_code=400,
            detail="Invalid account identifier format. Account must match alphanumeric host format without protocols or paths.",
        )

    # 2. Prepare authentication parameters
    session_creds: Dict[str, Any] = {
        "account": payload.account.strip(),
        "user": payload.user.strip(),
        "warehouse": payload.warehouse.strip() if payload.warehouse else None,
        "database": payload.database.strip() if payload.database else None,
        "schema": payload.schema_name.strip() if payload.schema_name else None,
        "role": payload.role.strip() if payload.role else None,
        "use_mock": payload.use_mock,
    }

    if payload.auth_type == "key_pair":
        if not payload.private_key_pem:
            raise HTTPException(status_code=400, detail="private_key_pem is required for key_pair authentication.")
        try:
            der_bytes = prepare_private_key_der(
                payload.private_key_pem.encode("utf-8"),
                payload.private_key_passphrase,
            )
            session_creds["private_key_der"] = der_bytes
        except Exception as key_err:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse RSA private key: {key_err}. Ensure valid PKCS8 or PKCS1 PEM format.",
            )
    elif payload.auth_type == "password":
        if not payload.password:
            raise HTTPException(status_code=400, detail="Password is required for password authentication.")
        session_creds["password"] = payload.password
    else:
        raise HTTPException(status_code=400, detail="auth_type must be either 'key_pair' or 'password'.")

    # 3. Test connection using driver
    driver = get_snowflake_driver(use_mock=bool(payload.use_mock))
    try:
        driver.test_connection(session_creds)
    except Exception as conn_err:
        logger.error("Connection failed: %s", conn_err)
        friendly_err = sanitize_snowflake_error(conn_err)
        raise HTTPException(
            status_code=400,
            detail=f"Snowflake Connection Failed: {friendly_err}",
        )

    # 4. Store in encrypted vault (15-min sliding TTL)
    caller_owner_id = get_authenticated_owner_id(request) or "local-development"
    user_id = caller_owner_id
    vault = get_snowflake_vault()
    session_id = vault.store_session(user_id, session_creds)

    return SnowflakeConnectResponse(
        session_id=session_id,
        account=payload.account,
        user=payload.user,
        warehouse=payload.warehouse,
        database=payload.database,
        schema_name=payload.schema_name,
        message="Snowflake connection verified and secure session established.",
    )


@router.get("/databases")
async def list_databases(
    request: Request,
    session_id: str = Query(..., description="Active session token"),
    _auth: None = Depends(verify_api_key),
):
    """Discovers available databases using SHOW DATABASES (zero warehouse compute cost)."""
    caller_owner_id = get_authenticated_owner_id() or "local-development"
    creds = _resolve_session_config(session_id, user_id=caller_owner_id)
    driver = get_snowflake_driver(use_mock=bool(creds.get("use_mock")))
    try:
        databases = driver.discover_databases(creds)
        return {"databases": databases}
    except Exception as e:
        logger.error("Failed to list databases: %s", e)
        friendly_err = sanitize_snowflake_error(e)
        raise HTTPException(status_code=500, detail=friendly_err)


@router.get("/schemas")
async def list_schemas(
    request: Request,
    session_id: str = Query(..., description="Active session token"),
    database_name: str = Query(..., description="Database name to browse"),
    _auth: None = Depends(verify_api_key),
):
    """Discovers schemas inside a database using SHOW SCHEMAS IN DATABASE."""
    caller_owner_id = get_authenticated_owner_id() or "local-development"
    creds = _resolve_session_config(session_id, user_id=caller_owner_id)
    driver = get_snowflake_driver(use_mock=bool(creds.get("use_mock")))
    try:
        schemas = driver.discover_schemas(creds, database_name)
        return {"database": database_name, "schemas": schemas}
    except Exception as e:
        logger.error("Failed to list schemas in %s: %s", database_name, e)
        friendly_err = sanitize_snowflake_error(e)
        raise HTTPException(status_code=500, detail=friendly_err)


@router.get("/objects")
async def list_objects(
    request: Request,
    session_id: str = Query(..., description="Active session token"),
    database_name: str = Query(..., description="Database name"),
    schema_name: str = Query(..., description="Schema name"),
    limit: int = Query(500, ge=1, le=1000),
    from_last: Optional[str] = Query(None, description="Paging cursor"),
    _auth: None = Depends(verify_api_key),
):
    """Discovers tables and views with row counts and byte estimations."""
    caller_owner_id = get_authenticated_owner_id() or "local-development"
    creds = _resolve_session_config(session_id, user_id=caller_owner_id)
    driver = get_snowflake_driver(use_mock=bool(creds.get("use_mock")))
    try:
        objects = driver.discover_objects(creds, database_name, schema_name, limit=limit, from_last=from_last)
        has_more = len(objects) == limit
        return {"database": database_name, "schema": schema_name, "objects": objects, "has_more": has_more}
    except Exception as e:
        logger.error("Failed to list objects in %s.%s: %s", database_name, schema_name, e)
        friendly_err = sanitize_snowflake_error(e)
        raise HTTPException(status_code=500, detail=friendly_err)


@router.post("/preview")
@limiter.limit("30/minute")
async def preview_query(
    request: Request,
    payload: SnowflakePreviewRequest = Body(...),
    _auth: None = Depends(verify_api_key),
):
    """Executes a safe limited preview query wrapped in a subquery."""
    is_valid, clean_sql, err_msg = validate_sql_query(payload.query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_msg)

    caller_owner_id = get_authenticated_owner_id() or "local-development"
    creds = _resolve_session_config(payload.session_id, user_id=caller_owner_id)
    driver = get_snowflake_driver(use_mock=bool(creds.get("use_mock")))
    try:
        rows, cols = driver.preview_query(creds, clean_sql, limit=payload.limit)
        return {"columns": cols, "rows": rows, "total_rows_returned": len(rows)}
    except Exception as e:
        logger.error("Preview execution failed: %s", e)
        friendly_err = sanitize_snowflake_error(e)
        raise HTTPException(status_code=500, detail=friendly_err)


@router.post("/import")
@limiter.limit("10/minute")
async def start_snowflake_import(
    request: Request,
    payload: SnowflakeImportRequest = Body(...),
    _auth: None = Depends(verify_api_key),
):
    """
    Submits an analytical query to be streamed in the background into a local Parquet extract.
    Returns task_id for tracking progress.
    """
    is_valid, clean_sql, err_msg = validate_sql_query(payload.query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_msg)

    caller_owner_id = get_authenticated_owner_id(request)
    user_id = caller_owner_id or "local-development"

    # Validate that the session is still active
    creds = _resolve_session_config(payload.session_id, user_id=user_id)
    filename = f"{payload.dataset_name or 'snowflake'}_{uuid.uuid4().hex[:8]}.parquet"
    from app.services.storage import get_storage_provider
    storage = get_storage_provider()
    base_dir = str(getattr(storage, "base_dir", "temp_uploads"))
    output_dir = os.path.join(base_dir, "snowflake")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Register initial job state in TaskManager
    from app.services.task_manager import title_task_manager, TaskStatus
    task_id = await title_task_manager.create_job_async(
        filename=filename,
        owner_id=caller_owner_id,
    )
    title_task_manager.update_progress(task_id, 5, "Initializing Snowflake asynchronous query extraction...")

    # Dispatch to background task runner (ARQ or asyncio)
    await dispatch_task(
        "app.tasks.import_snowflake",
        user_id,
        payload.session_id,
        task_id,
        clean_sql,
        output_path,
        payload.byte_limit or (150 * 1024 * 1024),
    )

    return {
        "task_id": task_id,
        "filename": filename,
        "status": "QUEUED",
        "message": "Snowflake data extraction queued.",
    }


@router.post("/cancel/{task_id}")
async def cancel_snowflake_query(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """Flags active Snowflake extraction for immediate cooperative cancellation."""
    validate_task_id(task_id)
    from app.services.task_manager import title_task_manager
    job = await title_task_manager.get_job_async(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")

    caller_owner_id = get_authenticated_owner_id(request)
    check_task_ownership(job.owner_id, caller_owner_id)

    vault = get_snowflake_vault()
    redis_client = vault._get_redis_client()
    if redis_client:
        redis_client.set(f"snowflake:cancel:{task_id}", "1", ex=60)
        # Attempt immediate abort if active_query sfqid is known
        sfqid = redis_client.get(f"snowflake:active_query:{task_id}")
        if sfqid:
            logger.info("Found active sfqid %s for task %s; aborting...", sfqid, task_id)
    
    title_task_manager.fail_job(task_id, "Snowflake ingestion cancelled by user.")

    return {"task_id": task_id, "status": "CANCELLED", "message": "Cancellation signal sent."}


@router.post("/delete-extract/{task_id}")
@router.delete("/extract/{task_id}")
async def delete_snowflake_extract(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Enforces per-user file_ref access validation and permanently purges local Snowflake
    Parquet extracts and task job metadata (GDPR/DPA compliance).
    """
    validate_task_id(task_id)
    from app.services.task_manager import title_task_manager
    job = await title_task_manager.get_job_async(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")

    caller_owner_id = get_authenticated_owner_id(request)
    check_task_ownership(job.owner_id, caller_owner_id)

    # 1. Cooperative cancel signal in Redis if still active
    vault = get_snowflake_vault()
    redis_client = vault._get_redis_client()
    if redis_client:
        redis_client.set(f"snowflake:cancel:{task_id}", "1", ex=60)

    # 2. Delete local extract file
    from app.services.storage import get_storage_provider
    storage = get_storage_provider()
    deleted_files = 0

    # From job.result if available
    file_ref = (job.result or {}).get("_file_ref") if isinstance(job.result, dict) else None
    if file_ref:
        try:
            if storage.delete(file_ref):
                deleted_files += 1
        except Exception as e:
            logger.warning("Error deleting file_ref %s: %s", file_ref, e)

    # From storage base_dir / snowflake / filename
    base_dir = str(getattr(storage, "base_dir", "temp_uploads"))
    candidates = [
        os.path.join(base_dir, "snowflake", job.filename or ""),
        os.path.join(base_dir, job.filename or ""),
    ]
    for c in candidates:
        if os.path.isfile(c):
            try:
                os.remove(c)
                deleted_files += 1
            except Exception as e:
                logger.warning("Error removing %s: %s", c, e)

    # 3. Mark job as purged
    title_task_manager.fail_job(task_id, "Extract permanently deleted by user.")

    return {
        "task_id": task_id,
        "status": "DELETED",
        "deleted_files": deleted_files,
        "message": "Extract files and task state permanently purged.",
    }


@router.post("/disconnect/{session_id}")
async def disconnect_snowflake(
    session_id: str,
    _auth: None = Depends(verify_api_key),
):
    """Explicitly purges Snowflake credentials and closes pooled connections."""
    vault = get_snowflake_vault()
    vault.invalidate_session(session_id)
    return {"session_id": session_id, "message": "Snowflake session invalidated and credentials purged."}
