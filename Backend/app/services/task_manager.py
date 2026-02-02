import uuid
import logging
from typing import Dict, Any, Optional

from app.db.base import SessionLocal, engine, Base
from app.db.models import Job

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

logger = logging.getLogger(__name__)

class TaskStatus:
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TaskManager:
    """
    Persisted TaskManager using SQLite + SQLAlchemy.
    Replaces the previous in-memory dictionary implementation.
    """
    
    def create_job(self, filename: str = "") -> str:
        db = SessionLocal()
        try:
            job_id = str(uuid.uuid4())
            new_job = Job(id=job_id, filename=filename, status=TaskStatus.PENDING, progress=0)
            db.add(new_job)
            db.commit()
            db.refresh(new_job)
            logger.info(f"Job {job_id} created in DB.")
            return job_id
        finally:
            db.close()

    def update_status(self, job_id: str, status: str, result: Dict[str, Any] = None):
        """Specifically update status and optional partial result."""
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = status
                if result:
                    job.result = result
                db.commit()
        finally:
            db.close()

    def update_progress(self, job_id: str, progress: int, message: str):
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                # Only auto-switch to PROCESSING if we are not in a final or paused state
                if job.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
                    job.status = TaskStatus.PROCESSING
                
                job.progress = max(0, min(100, progress))
                job.message = message
                db.commit()
        finally:
            db.close()

    def complete_job(self, job_id: str, result: Dict[str, Any], report_path: str = None):
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = TaskStatus.COMPLETED
                job.progress = 100
                job.message = "Analysis Complete"
                job.result = result
                job.report_path = report_path
                db.commit()
                logger.info(f"Job {job_id} marked COMPLETED in DB. Report: {report_path}")
        finally:
            db.close()

    def fail_job(self, job_id: str, error_message: str):
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = TaskStatus.FAILED
                job.error = error_message
                job.message = "Failed"
                db.commit()
                logger.error(f"Job {job_id} FAILED: {error_message}")
        finally:
            db.close()

    def get_job(self, job_id: str) -> Optional[Job]:
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            # Return model directly (pydantic will serialize it)
            return job
        finally:
            db.close()

# Singleton instance
title_task_manager = TaskManager()
