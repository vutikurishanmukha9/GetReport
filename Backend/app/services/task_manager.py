from typing import Dict, Any, Optional
import uuid
from enum import Enum
from dataclasses import dataclass, field
import datetime

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Job:
    id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    message: str = "Queued"
    result: Optional[Dict[str, Any]] = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    error: Optional[str] = None

class TaskManager:
    """
    Simple in-memory task manager. 
    In a real production app, this should be replaced by Redis + Celery/RQ.
    """
    def __init__(self):
        self._jobs: Dict[str, Job] = {}

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = Job(id=job_id)
        return job_id

    def update_progress(self, job_id: str, progress: int, message: str):
        if job_id in self._jobs:
            self._jobs[job_id].status = TaskStatus.PROCESSING
            self._jobs[job_id].progress = max(0, min(100, progress))
            self._jobs[job_id].message = message

    def complete_job(self, job_id: str, result: Dict[str, Any]):
        if job_id in self._jobs:
            self._jobs[job_id].status = TaskStatus.COMPLETED
            self._jobs[job_id].progress = 100
            self._jobs[job_id].message = "Analysis Complete"
            self._jobs[job_id].result = result

    def fail_job(self, job_id: str, error_message: str):
        if job_id in self._jobs:
            self._jobs[job_id].status = TaskStatus.FAILED
            self._jobs[job_id].error = error_message
            self._jobs[job_id].message = "Failed"

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

# Global instance
title_task_manager = TaskManager()
