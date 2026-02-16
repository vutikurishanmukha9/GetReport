"""
Upload Route — File ingestion endpoint.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
import logging
import os
import re

from app.core.limiter import limiter, UPLOAD_LIMIT
from app.core.config import settings
from app.services.task_manager import title_task_manager
from app.services.storage import get_storage_provider
from app.tasks import inspect_file_task
from app.core.file_validation import validate_file_signature

storage = get_storage_provider()
logger = logging.getLogger(__name__)
router = APIRouter()

# Global state for lazy cleanup (DoS prevention)
_last_cleanup_time = 0

class TaskResponse(BaseModel):
    task_id: str
    message: str

@router.post("/upload", response_model=TaskResponse)
@limiter.limit(UPLOAD_LIMIT)
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Initiates processing using Streaming Upload (RAM Safe).
    Returns Task ID immediately.
    """
    try:
        # Pre-validate extension
        if not file.filename.lower().endswith(('.csv', '.xls', '.xlsx')):
             raise HTTPException(400, "Invalid file type. Only CSV and Excel supported.")
             
        # Content Validation (Phase 4 Security Hardening)
        await validate_file_signature(file)
         
        # Enforce file size limit (streaming — avoids loading entire file to RAM)
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        size = 0
        CHUNK_SIZE = 64 * 1024  # 64KB chunks
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB")
        await file.seek(0)  # Reset for downstream read
             
        # Sanitize Filename (Security Fix)
        base_name = os.path.basename(file.filename)
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name)
        
        if not safe_filename:
            safe_filename = "unnamed_file.csv"
             
        # Create Task
        task_id = await title_task_manager.create_job_async(safe_filename)
        
        # Save file via Storage Service
        file_ref = storage.save_upload(file.file, safe_filename)
        
        # Start Inspection Task (Phase 1) - VIA CELERY
        inspect_file_task.delay(task_id, file_ref, safe_filename)
        
        # Schedule cleanup for old reports (Lazy Cleanup: Max once per hour)
        # Prevents separate thread per request (DoS mitigation)
        from app.services.cleanup import cleanup_old_files
        import time
        
        global _last_cleanup_time
        try:
            _last_cleanup_time
        except NameError:
            _last_cleanup_time = 0
            
        now = time.time()
        if now - _last_cleanup_time > 3600:
            output_dir = os.path.join(os.getcwd(), "outputs")
            background_tasks.add_task(cleanup_old_files, output_dir, 86400)
            _last_cleanup_time = now
        
        return TaskResponse(
            task_id=task_id, 
            message="File uploaded. Processing started."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
