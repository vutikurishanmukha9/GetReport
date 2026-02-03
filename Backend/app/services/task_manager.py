import uuid
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from app.db import get_db_connection, init_db

logger = logging.getLogger(__name__)

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
            
        # Parse result JSON if present
        result_data = None
        if row["result_json"]:
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
            report_path=row["report_path"]
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

    def update_status(self, task_id: str, status: TaskStatus, result: Optional[Dict[str, Any]] = None):
        with get_db_connection() as conn:
            if result:
                result_json = json.dumps(result)
                conn.execute(
                    "UPDATE jobs SET status = ?, result_json = ?, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (status, result_json, task_id)
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (status, task_id)
                )
            conn.commit()

    def complete_job(self, task_id: str, result: Dict[str, Any], report_path: Optional[str] = None):
        result_json = json.dumps(result)
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE jobs 
                SET status = ?, 
                    progress = 100, 
                    message = 'Completed', 
                    result_json = ?, 
                    report_path = ?,
                    updated_at = CURRENT_TIMESTAMP 
                WHERE task_id = ?
                """,
                (TaskStatus.COMPLETED, result_json, report_path, task_id)
            )
            conn.commit()
            
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

title_task_manager = TaskManager()
