"""
API Endpoint Aggregator
Imports and includes all route sub-modules.
Each module is a focused, single-responsibility router.
"""
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.limiter import limiter, ANALYZE_LIMIT
from app.core.auth import verify_api_key, validate_task_id
from app.core.task_dispatcher import dispatch_task
from app.services.task_manager import title_task_manager, TaskStatus
from app.services.analysis_config import AnalysisConfig

# Import sub-routers
from app.api.routes import upload, status, report, chat, issues

logger = logging.getLogger(__name__)
router = APIRouter()

# ─── Include Sub-Routers ─────────────────────────────────────────────────────
router.include_router(upload.router, tags=["upload"])
router.include_router(status.router, tags=["status"])
router.include_router(report.router, tags=["report"])
router.include_router(chat.router, tags=["chat"])
router.include_router(issues.router, tags=["issues"])


class AnalysisRulesRequest(BaseModel):
    rules: Dict[str, Any]
    analysis_config: Optional[AnalysisConfig] = None


# ─── Start Analysis (remains here as it bridges upload → analysis) ───────────

@router.post("/jobs/{task_id}/analyze")
@limiter.limit(ANALYZE_LIMIT)
async def start_analysis(
    request: Request,
    task_id: str,
    body: AnalysisRulesRequest,
    background_tasks: BackgroundTasks,
    _auth: None = Depends(verify_api_key),
):
    """
    Stage 2: User approves cleaning rules and starts full analysis pipeline.
    Dispatches via unified task dispatcher (ARQ worker in production or
    native in-memory asyncio coroutine in local dev).
    """
    validate_task_id(task_id)
    logger.info(f"Received start_analysis for {task_id}. Rules keys: {list(body.rules.keys())}")

    # Job Retrieval (Async)
    job = await title_task_manager.get_job_async(task_id)

    if not job:
        logger.error(f"Job {task_id} NOT FOUND.")
        raise HTTPException(404, "Job not found")

    logger.info(f"Job {task_id} status: {job.status}")

    if job.status != TaskStatus.WAITING_FOR_USER:
        msg = f"Job is not waiting for input. Current status: {job.status}"
        logger.warning(msg)
        return JSONResponse(status_code=409, content={"message": msg})

    # Start Analysis Pipeline (Phase 2) via Unified Task Dispatcher
    analysis_config = body.analysis_config.model_dump() if body.analysis_config else None
    await dispatch_task("app.tasks.resume_analysis", task_id, body.rules, analysis_config)
    return {"message": "Analysis started"}
