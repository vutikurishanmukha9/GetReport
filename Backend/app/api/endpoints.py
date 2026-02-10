"""
API Endpoint Aggregator
Imports and includes all route sub-modules.
Each module is a focused, single-responsibility router.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

from app.core.limiter import limiter, ANALYZE_LIMIT
from app.services.task_manager import title_task_manager, TaskStatus
from app.tasks import resume_analysis_task

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


# ─── Data Models (shared) ───────────────────────────────────────────────────

class AnalysisRulesRequest(BaseModel):
    rules: Dict[str, Any]
    analysis_config: Optional[Dict[str, Any]] = None


# ─── Start Analysis (remains here as it bridges upload → analysis) ───────────

@router.post("/jobs/{task_id}/analyze")
@limiter.limit(ANALYZE_LIMIT)
async def start_analysis(
    request: Request,
    task_id: str, 
    body: AnalysisRulesRequest, 
    background_tasks: BackgroundTasks
):
    """
    Stage 2: User approves cleaning rules and starts full analysis.
    """
    logger.info(f"Received start_analysis for {task_id}. Rules keys: {list(body.rules.keys())}")
    
    job = title_task_manager.get_job(task_id)
    if not job:
        logger.error(f"Job {task_id} NOT FOUND.")
        raise HTTPException(404, "Job not found")
        
    logger.info(f"Job {task_id} status: {job.status}")
    
    if job.status != TaskStatus.WAITING_FOR_USER:
       msg = f"Job is not waiting for input. Current status: {job.status}"
       logger.warning(msg)
       return JSONResponse(status_code=400, content={"message": msg})
        
    # Start Analysis Task (Phase 2) - VIA CELERY
    resume_analysis_task.delay(task_id, body.rules, body.analysis_config)
    return {"message": "Analysis started"}
