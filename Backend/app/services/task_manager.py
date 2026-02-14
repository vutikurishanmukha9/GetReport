import uuid
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from app.db import get_db_connection, init_db
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

def publish_update(task_id: str, data: Dict[str, Any]):
    """Publish update to Redis channel"""
    if redis_client:
        try:
            channel = f"task:{task_id}"
            redis_client.publish(channel, json.dumps(data))
        except Exception as e:
            logger.error(f"Redis publish failed: {e}")

# Ensure DB is initialized on module load (or app startup)
# For simplicity, we call it here, but ideally Main.py calls it.
try:
    init_db()
except Exception as e:
    logger.error(f"DB Init failed: {e}")

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
    result_path: Optional[str] = None  # Path to result JSON file

import io

# ─── Storage-Based Result Management ─────────────────────────────────────────

def _save_result_to_storage(task_id: str, result: Dict[str, Any]) -> str:
    """
    Save result JSON to storage (Local or S3).
    Returns the file reference (path or key).
    """
    # Serialize to JSON bytes
    json_bytes = json.dumps(result, indent=2, default=str).encode("utf-8")
    file_obj = io.BytesIO(json_bytes)
    
    # Filename for metadata (storage provider generates unique key usually)
    # We prefix with task_id to make it identifiable if storage allows
    filename = f"{task_id}_result.json"
    
    # Save
    file_ref = storage.save_upload(file_obj, filename)
    logger.info(f"Result saved to storage: {file_ref}")
    return file_ref

def _load_result_from_storage(file_ref: str) -> Optional[Dict[str, Any]]:
    """
    Load result JSON from storage reference.
    """
    if not file_ref:
        return None
        
    try:
        # get_absolute_path ensures it's available locally (downloads if needed)
        local_path = storage.get_absolute_path(file_ref)
        
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read result from storage {file_ref}: {e}")
    return None

class TaskManager:
    """
    Manages job states using SQLite persistence.
    Replaces the In-Memory Dictionary implementation.
    """

    def create_job(self, filename: str) -> str:
        task_id = str(uuid.uuid4())
        initial_status = TaskStatus.PENDING
        initial_message = "Job created"
        
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO jobs (task_id, status, filename, message, progress) VALUES (?, ?, ?, ?, ?)",
                (task_id, initial_status, filename, initial_message, 0)
            )
            conn.commit()
            
        logger.info(f"Job {task_id} created in DB.")
        return task_id

    def get_job(self, task_id: str) -> Optional[Job]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE task_id = ?", (task_id,)).fetchone()
            
        if not row:
            return None
            
        # Load result from storage first (preferred)
        result_data = None
        result_ref = row.get("result_path") if hasattr(row, 'get') else (row["result_path"] if "result_path" in row.keys() else None)
        
        if result_ref:
            result_data = _load_result_from_storage(result_ref)
        
        # Fallback: read from result_json column if storage ref not found (Legacy Support)
        # Note: We are removing result_json usage, but keeping read capability for old records is safe.
        if result_data is None and "result_json" in row.keys() and row["result_json"]:
            try:
                result_data = json.loads(row["result_json"])
            except:
                pass
                
        return Job(
            id=row["task_id"],
            status=TaskStatus(row["status"]),
            message=row["message"],
            progress=row["progress"],
            filename=row["filename"],
            result=result_data,
            error=row["error"],
            report_path=row["report_path"],
            result_path=result_ref
        )

    def update_progress(self, task_id: str, progress: int, message: Optional[str] = None):
        with get_db_connection() as conn:
            if message:
                conn.execute(
                    "UPDATE jobs SET progress = ?, message = ?, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (progress, message, task_id)
                )
            else:
                conn.execute(
                    "UPDATE jobs SET progress = ?, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (progress, task_id)
                )
            conn.commit()
            
        # Publish real-time update
        publish_update(task_id, {
            "task_id": task_id,
            "status": "processing", # Assumed if progress updates
            "progress": progress,
            "message": message or "Processing..."
        })

    def update_status(self, task_id: str, status: TaskStatus, result: Optional[Dict[str, Any]] = None):
        with get_db_connection() as conn:
            if result:
                # Save result to storage
                result_ref = _save_result_to_storage(task_id, result)
                conn.execute(
                    "UPDATE jobs SET status = ?, result_path = ?, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (status, result_ref, task_id)
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (status, task_id)
                )
            conn.commit()
            
        # Publish real-time update
        publish_update(task_id, {
            "task_id": task_id,
            "status": status,
            "progress": 100 if status == TaskStatus.COMPLETED else None,
            "result": result
        })

    def update_result(self, task_id: str, result: Dict[str, Any]):
        """Update just the result JSON without changing status. Used by Issue Ledger endpoints."""
        result_ref = _save_result_to_storage(task_id, result)
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE jobs SET result_path = ?, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                (result_ref, task_id)
            )
            conn.commit()
            
        # Publish real-time update (Issue Ledger change)
        publish_update(task_id, {
            "task_id": task_id,
            "type": "ledger_update",
            "result": result
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
                    updated_at = CURRENT_TIMESTAMP 
                WHERE task_id = ?
                """,
                (TaskStatus.COMPLETED, result_ref, report_path, task_id)
            )
            conn.commit()
            
        # Publish real-time update
        publish_update(task_id, {
            "task_id": task_id,
            "status": TaskStatus.COMPLETED,
            "progress": 100,
            "message": "Completed",
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
                    updated_at = CURRENT_TIMESTAMP 
                WHERE task_id = ?
                """,
                (TaskStatus.FAILED, error_msg, task_id)
            )
            conn.commit()
            logger.error(f"Job {task_id} marked as FAILED in DB: {error_msg}")
            
        # Publish real-time update
        publish_update(task_id, {
            "task_id": task_id,
            "status": TaskStatus.FAILED,
            "error": error_msg
        })

title_task_manager = TaskManager()
