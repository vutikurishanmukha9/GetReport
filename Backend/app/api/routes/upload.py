"""
Upload Route — Single-Pass Zero-Copy Ingestion Endpoint.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Depends
from pydantic import BaseModel
import logging
import os
import re
import hashlib
import uuid

from app.core.limiter import limiter, UPLOAD_LIMIT
from app.core.config import settings
from app.core.auth import verify_api_key
from app.services.task_manager import title_task_manager
from app.services.storage import get_storage_provider
from app.tasks import inspect_file_task

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


async def _ensure_file_size(file: UploadFile) -> int:
    """Validate per-file size limit and return total bytes."""
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    size = getattr(file, "size", None)
    if size is None:
        try:
            file.file.seek(0, 2)
            size = file.file.tell()
            file.file.seek(0)
        except Exception:
            size = 0
            while chunk := await file.read(64 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB")
            await file.seek(0)
    if size > max_bytes:
        raise HTTPException(413, f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB")
    return size


async def _validate_upload_sizes(files: list[UploadFile]) -> None:
    """Apply aggregate file size limits before batch operations."""
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    total_size = 0
    for file in files:
        total_size += await _ensure_file_size(file)
        if total_size > max_bytes:
            raise HTTPException(413, f"Combined upload too large. Max total size: {settings.MAX_UPLOAD_SIZE_MB}MB")


@router.post("/upload", response_model=TaskResponse)
@limiter.limit(UPLOAD_LIMIT)
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _auth: None = Depends(verify_api_key),
):
    """
    Initiates processing using Single-Pass Streaming Ingestion (Zero-Copy & RAM Safe).
    Returns Task ID immediately.
    """
    try:
        # Pre-validate extension
        if not file.filename.lower().endswith(ALLOWED_EXTENSIONS_TUPLE):
            raise HTTPException(400, "Invalid file type. Supported formats: CSV, TSV, Excel, Parquet, JSON, JSONL, Feather, GZ.")

        # Sanitize Filename
        base_name = os.path.basename(file.filename)
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name)
        if not safe_filename:
            safe_filename = "unnamed_file.csv"

        # Stream upload in single pass (calculates hash, validates magic bytes, bounds size, saves to storage)
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        file_ref, file_hash, total_bytes = await storage.save_upload_streaming(
            file,
            safe_filename,
            max_bytes
        )

        # Create Task
        task_id = await title_task_manager.create_job_async(safe_filename, file_hash=file_hash)

        # Start Inspection Task (Phase 1) - VIA CELERY
        inspect_file_task.delay(task_id, file_ref, safe_filename)

        # Schedule cleanup for old reports (Lazy Cleanup: Max once per hour)
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
    Ingests multiple datasets at once using Single-Pass Streaming.
    Returns list of Task IDs immediately.
    """
    if not files or len(files) == 0:
        raise HTTPException(400, "No files provided.")

    if len(files) > 10:
        raise HTTPException(400, "Batch upload limit is 10 files per request.")

    task_ids = []
    task_details = []
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    for file in files:
        if not file.filename.lower().endswith(ALLOWED_EXTENSIONS_TUPLE):
            raise HTTPException(400, f"Invalid file type for '{file.filename}'.")

        base_name = os.path.basename(file.filename)
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name) or "unnamed_file.csv"

        file_ref, file_hash, total_bytes = await storage.save_upload_streaming(
            file,
            safe_filename,
            max_bytes
        )

        task_id = await title_task_manager.create_job_async(safe_filename, batch_id=batch_id, file_hash=file_hash)
        inspect_file_task.delay(task_id, file_ref, safe_filename)

        task_ids.append(task_id)
        task_details.append({"task_id": task_id, "batch_id": batch_id, "filename": safe_filename, "file_hash": file_hash[:12]})

    return BatchTaskResponse(
        task_ids=task_ids,
        tasks=task_details,
        message=f"Uploaded {len(files)} files successfully under Batch '{batch_id}'. Inspection tasks started."
    )


@router.post("/upload/join", response_model=TaskResponse)
@limiter.limit(UPLOAD_LIMIT)
async def upload_and_join_files(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    join_key: str = Form(...),
    join_type: str = Form("inner"),
    _auth: None = Depends(verify_api_key),
):
    """
    Ingests multiple files, joins them on a primary key column, and starts inspection on the joined dataset.
    """
    from app.services.data_processing import load_dataframe, join_datasets
    import io

    if not files or len(files) < 2:
        raise HTTPException(400, "At least 2 files are required for a joined analysis.")

    if len(files) > 5:
        raise HTTPException(400, "Maximum 5 files allowed for joined analysis.")

    if join_type not in {"inner", "left", "full", "outer", "semi", "anti", "cross"}:
        raise HTTPException(400, "Invalid join type.")

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    dfs_map = {}
    temp_files_to_remove = []

    try:
        for file in files:
            if not file.filename.lower().endswith(ALLOWED_EXTENSIONS_TUPLE):
                raise HTTPException(400, f"Invalid file type for '{file.filename}'.")

            base_name = os.path.basename(file.filename)
            safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name) or "unnamed_file.csv"

            file_ref, file_hash, total_bytes = await storage.save_upload_streaming(
                file,
                safe_filename,
                max_bytes
            )
            temp_files_to_remove.append(file_ref)

            df = load_dataframe(storage.get_absolute_path(file_ref))
            dfs_map[safe_filename] = df

        # Perform Multi-Dataset Join
        joined_df = join_datasets(dfs_map, join_key=join_key, how=join_type)

        if joined_df.height == 0:
            raise HTTPException(400, f"Joined dataset resulted in 0 rows. Verify join key '{join_key}' and join mode '{join_type}'.")

        # Save joined dataframe as CSV
        joined_filename = f"joined_{join_type}_{uuid.uuid4().hex[:8]}.csv"
        csv_bytes = io.BytesIO()
        joined_df.write_csv(csv_bytes)
        csv_bytes.seek(0)

        file_hash = hashlib.sha256(csv_bytes.getvalue()).hexdigest()
        task_id = await title_task_manager.create_job_async(joined_filename, file_hash=file_hash)

        joined_file_ref = storage.save_upload(csv_bytes, joined_filename)
        inspect_file_task.delay(task_id, joined_file_ref, joined_filename)

        return TaskResponse(
            task_id=task_id,
            message=f"Successfully joined {len(files)} files on '{join_key}' ({join_type} join). Inspection started."
        )

    except KeyError as ke:
        raise HTTPException(400, str(ke).strip("'"))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Joined upload failed: {str(e)}")
        raise HTTPException(500, f"Joined upload failed: {str(e)}")
    finally:
        # Clean up intermediate staged component files
        for tmp_ref in temp_files_to_remove:
            try:
                storage.delete(tmp_ref)
            except Exception:
                pass
