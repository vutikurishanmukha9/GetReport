"""
Upload Route — File ingestion endpoint.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request, Depends
from pydantic import BaseModel
import logging
import os
import re

from app.core.limiter import limiter, UPLOAD_LIMIT
from app.core.config import settings
from app.core.auth import verify_api_key
from app.services.task_manager import title_task_manager
from app.services.storage import get_storage_provider
from app.tasks import inspect_file_task
from app.core.file_validation import validate_file_signature

storage = get_storage_provider()
logger = logging.getLogger(__name__)
router = APIRouter()

# Global state for lazy cleanup (DoS prevention)
_last_cleanup_time = 0

ALLOWED_EXTENSIONS_TUPLE = (
    '.csv', '.xls', '.xlsx', '.parquet', '.json', '.jsonl', 
    '.ndjson', '.tsv', '.feather', '.arrow', '.gz'
)

class TaskResponse(BaseModel):
    task_id: str
    message: str

class BatchTaskResponse(BaseModel):
    task_ids: list[str]
    tasks: list[dict]
    message: str

async def _compute_file_hash(file: UploadFile) -> str:
    """Compute SHA-256 hash of file content without consuming buffer."""
    hasher = hashlib.sha256()
    await file.seek(0)
    chunk = await file.read(64 * 1024)
    while chunk:
        hasher.update(chunk)
        chunk = await file.read(64 * 1024)
    await file.seek(0)
    return hasher.hexdigest()

@router.post("/upload", response_model=TaskResponse)
@limiter.limit(UPLOAD_LIMIT)
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _auth: None = Depends(verify_api_key),
):
    """
    Initiates processing using Streaming Upload (RAM Safe).
    Returns Task ID immediately.
    """
    try:
        # Pre-validate extension
        if not file.filename.lower().endswith(ALLOWED_EXTENSIONS_TUPLE):
             raise HTTPException(400, "Invalid file type. Supported formats: CSV, TSV, Excel, Parquet, JSON, JSONL, Feather, GZ.")
             
        # Content Validation (Phase 4 Security Hardening)
        await validate_file_signature(file)
        file_hash = await _compute_file_hash(file)
         
        # Enforce file size limit in O(1) time using pointer seek
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        try:
            await file.seek(0, 2)  # Seek to end
            size = await file.tell()
            await file.seek(0)  # Reset to start
        except Exception as seek_err:
            # Fallback if async seek(0, 2)/tell() is unsupported by the FastAPI/Starlette wrapper version
            logger.warning(f"Async seek/tell size check failed ({seek_err}), falling back to streaming read.")
            size = 0
            CHUNK_SIZE = 64 * 1024
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB")
            await file.seek(0)

        if size > max_bytes:
            raise HTTPException(413, f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB")

             
        # Sanitize Filename (Security Fix)
        base_name = os.path.basename(file.filename)
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name)
        
        if not safe_filename:
            safe_filename = "unnamed_file.csv"
             
        # Create Task
        task_id = await title_task_manager.create_job_async(safe_filename, file_hash=file_hash)
        
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
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/upload/batch", response_model=BatchTaskResponse)
@limiter.limit(UPLOAD_LIMIT)
async def upload_files_batch(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    _auth: None = Depends(verify_api_key),
):
    """
    Ingests multiple datasets at once (Batch Upload).
    Returns list of Task IDs immediately.
    """
    if not files or len(files) == 0:
        raise HTTPException(400, "No files provided.")

    if len(files) > 10:
        raise HTTPException(400, "Batch upload limit is 10 files per request.")

    task_ids = []
    task_details = []
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"

    for file in files:
        if not file.filename.lower().endswith(ALLOWED_EXTENSIONS_TUPLE):
            raise HTTPException(400, f"Invalid file type for '{file.filename}'.")

        await validate_file_signature(file)
        file_hash = await _compute_file_hash(file)

        base_name = os.path.basename(file.filename)
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name) or "unnamed_file.csv"

        task_id = await title_task_manager.create_job_async(safe_filename, batch_id=batch_id, file_hash=file_hash)
        file_ref = storage.save_upload(file.file, safe_filename)
        inspect_file_task.delay(task_id, file_ref, safe_filename)

        task_ids.append(task_id)
        task_details.append({"task_id": task_id, "batch_id": batch_id, "filename": safe_filename, "file_hash": file_hash[:12]})

    return BatchTaskResponse(
        task_ids=task_ids,
        tasks=task_details,
        message=f"Uploaded {len(files)} files successfully under Batch '{batch_id}'. Inspection tasks started."
    )

