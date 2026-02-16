"""
Issues Route — Issue Ledger CRUD and lifecycle management.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
import logging

from app.services.task_manager import title_task_manager, TaskStatus

logger = logging.getLogger(__name__)
router = APIRouter()

class IssueActionRequest(BaseModel):
    note: str = ""

class IssueModifyRequest(BaseModel):
    fix_code: str
    note: str = ""


def _recalc_summary(issues: list) -> dict:
    """Recalculate issue summary counts."""
    summary = {"pending": 0, "approved": 0, "rejected": 0, "modified": 0, "total": len(issues)}
    for issue in issues:
        status = issue.get("status", "pending")
        if status in summary:
            summary[status] += 1
    return summary


async def _get_job_and_ledger(task_id: str):
    """Helper: get job and validate it's in approval phase with a ledger."""
    job = await title_task_manager.get_job_async(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    if job.status != TaskStatus.WAITING_FOR_USER:
        raise HTTPException(400, "Job is not in approval phase")
    
    ledger_data = job.result.get("issue_ledger")
    if not ledger_data:
        raise HTTPException(400, "No issue ledger found")
    
    return job, ledger_data


@router.get("/jobs/{task_id}/issues")
async def get_issues(task_id: str):
    """Get the issue ledger for a task."""
    job = await title_task_manager.get_job_async(task_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    if not job.result:
        raise HTTPException(400, "No inspection data available")
    
    issue_ledger = job.result.get("issue_ledger")
    if not issue_ledger:
        return {"issues": [], "summary": {"total": 0}, "locked": False}
    
    return issue_ledger


@router.post("/jobs/{task_id}/issues/{issue_id}/approve")
async def approve_issue(task_id: str, issue_id: str, request: IssueActionRequest = None):
    """Approve a single issue for execution."""
    job, ledger_data = await _get_job_and_ledger(task_id)
    
    for issue in ledger_data.get("issues", []):
        if issue["id"] == issue_id:
            issue["status"] = "approved"
            if request and request.note:
                issue["user_note"] = request.note
            ledger_data["summary"] = _recalc_summary(ledger_data["issues"])
            await title_task_manager.update_result_async(task_id, job.result)
            return {"message": "Issue approved", "issue_id": issue_id}
    
    raise HTTPException(404, "Issue not found")


@router.post("/jobs/{task_id}/issues/{issue_id}/reject")
async def reject_issue(task_id: str, issue_id: str, request: IssueActionRequest = None):
    """Reject an issue - fix will not be applied."""
    job, ledger_data = await _get_job_and_ledger(task_id)
    
    for issue in ledger_data.get("issues", []):
        if issue["id"] == issue_id:
            issue["status"] = "rejected"
            if request and request.note:
                issue["user_note"] = request.note
            ledger_data["summary"] = _recalc_summary(ledger_data["issues"])
            await title_task_manager.update_result_async(task_id, job.result)
            return {"message": "Issue rejected", "issue_id": issue_id}
    
    raise HTTPException(404, "Issue not found")


@router.post("/jobs/{task_id}/issues/approve-all")
async def approve_all_issues(task_id: str):
    """Approve all pending issues."""
    job, ledger_data = await _get_job_and_ledger(task_id)
    
    count = 0
    for issue in ledger_data.get("issues", []):
        if issue["status"] == "pending":
            issue["status"] = "approved"
            count += 1
    
    ledger_data["summary"] = _recalc_summary(ledger_data["issues"])
    await title_task_manager.update_result_async(task_id, job.result)
    return {"message": f"Approved {count} issues", "count": count}


@router.post("/jobs/{task_id}/issues/reject-all")
async def reject_all_issues(task_id: str):
    """Reject all pending issues."""
    job, ledger_data = await _get_job_and_ledger(task_id)
    
    count = 0
    for issue in ledger_data.get("issues", []):
        if issue["status"] == "pending":
            issue["status"] = "rejected"
            count += 1
    
    ledger_data["summary"] = _recalc_summary(ledger_data["issues"])
    await title_task_manager.update_result_async(task_id, job.result)
    return {"message": f"Rejected {count} issues", "count": count}


@router.post("/jobs/{task_id}/issues/lock")
async def lock_issues(task_id: str):
    """Lock the issue ledger - no more changes allowed."""
    job, ledger_data = await _get_job_and_ledger(task_id)
    
    pending = sum(1 for i in ledger_data.get("issues", []) if i["status"] == "pending")
    if pending > 0:
        raise HTTPException(400, f"Cannot lock: {pending} issues still pending. Approve or reject all first.")
    
    ledger_data["locked"] = True
    ledger_data["locked_at"] = datetime.now().isoformat()
    
    await title_task_manager.update_result_async(task_id, job.result)
    return {"message": "Issue ledger locked", "summary": ledger_data["summary"]}
