import uuid
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from app.db import get_db_connection, get_async_db_connection, init_db
from app.core.config import settings
from app.services.storage import get_storage_provider
try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)
storage = get_storage_provider()

# Redis Client for PubSub (Status Updates)
redis_client = None
if redis:
    try:
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=1)
        redis_client.ping()
        logger.info(f"Connected to Redis for Pub/Sub at {settings.REDIS_URL}")
    except Exception as e:
        logger.warning(f"Redis not available ({e}). WebSockets will fallback to polling.")
else:
    logger.warning("Redis module not found. Install 'redis' for real-time updates.")

# Async Redis (if needed later) or just use Sync Redis client for publish (it's fast enough)
# For now, we stick to sync redis publish even in async methods unless it blocks too much.
# Usually publish is very fast (0.5ms).

def publish_update(task_id: str, data: Dict[str, Any]):
    """Publish update to Redis channel"""
    if redis_client:
        try:
            channel = f"task:{task_id}"
            redis_client.publish(channel, json.dumps(data))
        except Exception as e:
            logger.error(f"Redis publish failed: {e}")

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Job:
    id: str
    status: TaskStatus
    message: str
    progress: int = 0
    filename: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    report_path: Optional[str] = None
    result_path: Optional[str] = None
    version: int = 0  # For Optimistic Locking

import io

# ─── Storage-Based Result Management ─────────────────────────────────────────

def _save_result_to_storage(task_id: str, result: Dict[str, Any]) -> str:
    """Save result JSON to storage (Local or S3). Returns reference."""
    json_bytes = json.dumps(result, indent=2, default=str).encode("utf-8")
    file_obj = io.BytesIO(json_bytes)
    filename = f"{task_id}_result.json"
    file_ref = storage.save_upload(file_obj, filename)
    logger.info(f"Result saved to storage: {file_ref}")
    return file_ref

def _load_result_from_storage(file_ref: str) -> Optional[Dict[str, Any]]:
    """Load result JSON from storage reference."""
    if not file_ref: return None
    try:
        local_path = storage.get_absolute_path(file_ref)
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read result from storage {file_ref}: {e}")
    return None

class TaskManager:
    """
    Manages job states using Database persistence.
    Supports both Sync (Celery) and Async (FastAPI) contexts.
    """

    # ─── ASYNC METHODS (For FastAPI) ─────────────────────────────────────────

    async def create_job_async(self, filename: str) -> str:
        task_id = str(uuid.uuid4())
        initial_status = TaskStatus.PENDING
        initial_message = "Job created"
        
        # No Self-Healing here. DB Schema must be correct.
        query = "INSERT INTO jobs (task_id, status, filename, message, progress, version) VALUES (?, ?, ?, ?, ?, ?)"
        args = (task_id, initial_status, filename, initial_message, 0, 1)

        async with get_async_db_connection() as conn:
            await conn.execute(query, args)
            await conn.commit()
            
        logger.info(f"Job {task_id} created in DB (Async).")
        return task_id

    async def get_job_async(self, task_id: str) -> Optional[Job]:
        async with get_async_db_connection() as conn:
            # We assume wrapper returns dict-like or row-like object
            cursor = await conn.execute("SELECT * FROM jobs WHERE task_id = ?", (task_id,))
            row = await cursor.fetchone()
            
        if not row:
            return None
            
        # Helper to safely access row by name (supports dict or sqlite Row)
        def get_col(r, keys):
            # asyncpg Record behaves like mapping. aiosqlite Row behaves like mapping.
            return r[keys]

        # Load result from storage
        result_data = None
        result_ref = row["result_path"]
        
        # IO Block: Reading from storage (Synchronous IO).
        # Theoretically we should make storage async too, but for local files it's fast.
        # For S3 it's an HTTP request (blocking). 
        # TODO: Make storage provider async. For now, we assume it's acceptable for small JSONs.
        if result_ref:
            result_data = _load_result_from_storage(result_ref)
        
        return Job(
            id=row["task_id"],
            status=TaskStatus(row["status"]),
            message=row["message"],
            progress=row["progress"],
            filename=row["filename"],
            result=result_data,
            error=row["error"],
            report_path=row["report_path"],
            result_path=result_ref,
            version=row["version"]
        )

    async def update_status_async(self, task_id: str, status: TaskStatus, result: Optional[Dict[str, Any]] = None):
        """
        Updates status. 
        Note: We define 'Optimistic Locking' here as simply ensuring we increment version.
        For strict concurrency, we would need to pass expected_version, but API doesn't support it yet.
        So we just Atomically Increment.
        """
        result_ref = None
        if result:
            result_ref = _save_result_to_storage(task_id, result)

        async with get_async_db_connection() as conn:
            if result_ref:
                await conn.execute(
                    """
                    UPDATE jobs 
                    SET status = ?, result_path = ?, updated_at = CURRENT_TIMESTAMP, version = version + 1
                    WHERE task_id = ?
                    """,
                    (status, result_ref, task_id)
                )
            else:
                await conn.execute(
                    """
                    UPDATE jobs 
                    SET status = ?, updated_at = CURRENT_TIMESTAMP, version = version + 1
                    WHERE task_id = ?
                    """,
                    (status, task_id)
                )
            await conn.commit()
            
        publish_update(task_id, {
            "task_id": task_id,
            "status": status,
            "progress": 100 if status == TaskStatus.COMPLETED else None,
            "result": result
        })

    async def update_result_async(self, task_id: str, result: Dict[str, Any]):
        """Async update just the result JSON without changing status."""
        result_ref = _save_result_to_storage(task_id, result)
        async with get_async_db_connection() as conn:
            await conn.execute(
                "UPDATE jobs SET result_path = ?, updated_at = CURRENT_TIMESTAMP, version = version + 1 WHERE task_id = ?",
                (result_ref, task_id)
            )
            await conn.commit()
            
        publish_update(task_id, {
            "task_id": task_id,
            "type": "ledger_update",
            "result": result
        })

    # ─── SYNC METHODS (Legacy/Celery) ────────────────────────────────────────

    def create_job(self, filename: str) -> str:
        """Sync version for testing or legacy calls"""
        task_id = str(uuid.uuid4())
        initial_status = TaskStatus.PENDING
        initial_message = "Job created"
        
        args = (task_id, initial_status, filename, initial_message, 0, 1)
        query = "INSERT INTO jobs (task_id, status, filename, message, progress, version) VALUES (?, ?, ?, ?, ?, ?)"

        with get_db_connection() as conn:
            conn.execute(query, args)
            conn.commit()
            
        logger.info(f"Job {task_id} created in DB (Sync).")
        return task_id

    def get_job(self, task_id: str) -> Optional[Job]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE task_id = ?", (task_id,)).fetchone()
            
        if not row: return None
            
        result_data = None
        # Access via dict (psycopg2) or Row (sqlite)
        result_ref = row["result_path"] if "result_path" in row.keys() else None
        
        if result_ref:
            result_data = _load_result_from_storage(result_ref)
        elif "result_json" in row.keys() and row["result_json"]:
             try: result_data = json.loads(row["result_json"])
             except: pass
                
        return Job(
            id=row["task_id"],
            status=TaskStatus(row["status"]),
            message=row["message"],
            progress=row["progress"],
            filename=row["filename"],
            result=result_data,
            error=row["error"],
            report_path=row["report_path"],
            result_path=result_ref,
            version=row.get("version", 0) if hasattr(row, "get") else row["version"]
        )

    def update_progress(self, task_id: str, progress: int, message: Optional[str] = None):
        with get_db_connection() as conn:
            if message:
                conn.execute(
                    "UPDATE jobs SET progress = ?, message = ?, updated_at = CURRENT_TIMESTAMP, version = version + 1 WHERE task_id = ?",
                    (progress, message, task_id)
                )
            else:
                conn.execute(
                    "UPDATE jobs SET progress = ?, updated_at = CURRENT_TIMESTAMP, version = version + 1 WHERE task_id = ?",
                    (progress, task_id)
                )
            conn.commit()
            
        publish_update(task_id, {
            "task_id": task_id,
            "status": "processing",
            "progress": progress,
            "message": message or "Processing..."
        })

    def update_status(self, task_id: str, status: TaskStatus, result: Optional[Dict[str, Any]] = None):
        with get_db_connection() as conn:
            if result:
                result_ref = _save_result_to_storage(task_id, result)
                conn.execute(
                    "UPDATE jobs SET status = ?, result_path = ?, updated_at = CURRENT_TIMESTAMP, version = version + 1 WHERE task_id = ?",
                    (status, result_ref, task_id)
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP, version = version + 1 WHERE task_id = ?",
                    (status, task_id)
                )
            conn.commit()
            
        publish_update(task_id, {
            "task_id": task_id,
            "status": status,
            "progress": 100 if status == TaskStatus.COMPLETED else None,
            "result": result
        })

    def fail_job(self, task_id: str, error_msg: str):
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE jobs 
                SET status = ?, 
                    message = 'Failed', 
                    error = ?,
                    updated_at = CURRENT_TIMESTAMP,
                    version = version + 1
                WHERE task_id = ?
                """,
                (TaskStatus.FAILED, error_msg, task_id)
            )
            conn.commit()
            logger.error(f"Job {task_id} marked as FAILED in DB: {error_msg}")
            
        publish_update(task_id, {
            "task_id": task_id,
            "status": TaskStatus.FAILED,
            "error": error_msg
        })
    
    def complete_job(self, task_id: str, result: Dict[str, Any], report_path: Optional[str] = None):
        result_ref = _save_result_to_storage(task_id, result)
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE jobs 
                SET status = ?, 
                    progress = 100, 
                    message = 'Completed', 
                    result_path = ?, 
                    report_path = ?,
                    updated_at = CURRENT_TIMESTAMP, 
                    version = version + 1
                WHERE task_id = ?
                """,
                (TaskStatus.COMPLETED, result_ref, report_path, task_id)
            )
            conn.commit()
            
        publish_update(task_id, {
            "task_id": task_id,
            "status": TaskStatus.COMPLETED,
            "progress": 100,
            "message": "Completed",
            "result": result
        })

title_task_manager = TaskManager()
