from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import os
import asyncio
from concurrent.futures import ProcessPoolExecutor # NEW

from app.services.data_processing import (
    load_dataframe, clean_data, get_dataset_info, 
    UnsupportedFileTypeError, FileTooLargeError, EmptyFileError, ParseError
)
from app.services.visualization import generate_charts
from app.services.report_renderer import generate_pdf_report
from app.services.llm_insight import generate_insights
from app.services.task_manager import title_task_manager, TaskStatus
from app.services.rag_service import rag_service
from app.services.issue_ledger import detect_issues, IssueLedger
from app.services.storage import get_storage_provider
from app.tasks import inspect_file_task, resume_analysis_task, generate_pdf_task

storage = get_storage_provider()

logger = logging.getLogger(__name__)
router = APIRouter()

# Global Process Pool REMOVED - Using Celery

# ─── Data Models ─────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    question: str

class TaskResponse(BaseModel):
    task_id: str
    message: str

class StatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    report_download_url: Optional[str] = None

class AnalysisRulesRequest(BaseModel):
    rules: Dict[str, Any]
    analysis_config: Optional[Dict[str, Any]] = None # Tier 5

class IssueActionRequest(BaseModel):
    note: str = ""

class IssueModifyRequest(BaseModel):
    fix_code: str
    note: str = ""

# ─── Background Processor ────────────────────────────────────────────────────





# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/upload", response_model=TaskResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Initiates processing using Streaming Upload (RAM Safe).
    Returns Task ID immediately.
    """
    import shutil
    import tempfile
    import re
    
    try:
        # Pre-validate extension
        if not file.filename.lower().endswith(('.csv', '.xls', '.xlsx')):
             raise HTTPException(400, "Invalid file type. Only CSV and Excel supported.")
             
        # Sanitize Filename (Security Fix)
        # Remove path separators and non-alphanumeric chars (except .-_)
        base_name = os.path.basename(file.filename) # Defend against ../../ attacks
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name)
        
        # Ensure it's not empty
        if not safe_filename:
            safe_filename = "unnamed_file.csv"
             
        # Create Task
        task_id = title_task_manager.create_job(safe_filename)
        
        # Save file via Storage Service
        try:
            file_ref = storage.save_upload(file.file, safe_filename)
        except Exception as write_err:
            raise write_err
        
        # Start Inspection Task (Phase 1) - VIA CELERY
        inspect_file_task.delay(
            task_id, 
            file_ref, 
            safe_filename
        )
        
        # Also schedule strict hygiene cleanup for old reports
        from app.services.cleanup import cleanup_old_files
        output_dir = os.path.join(os.getcwd(), "outputs")
        # Note: We should ideally also clean up the storage provider, but for LocalStorage it handles its own folder
        background_tasks.add_task(cleanup_old_files, output_dir, 86400) # 24h retention
        
        return TaskResponse(
            task_id=task_id, 
            message="File uploaded. Processing started."
        )

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/jobs/{task_id}/analyze")
async def start_analysis(
    task_id: str, 
    request: AnalysisRulesRequest, 
    background_tasks: BackgroundTasks
):
    """
    Stage 2: User approves cleaning rules and starts full analysis.
    """
    logger.info(f"Received start_analysis for {task_id}. Rules keys: {list(request.rules.keys())}")
    
    job = title_task_manager.get_job(task_id)
    if not job:
        logger.error(f"Job {task_id} NOT FOUND.")
        raise HTTPException(404, "Job not found")
        
    logger.info(f"Job {task_id} status: {job.status}")
    
    # Check if job is in correct state (WAITING_FOR_USER)
    if job.status != TaskStatus.WAITING_FOR_USER:
       msg = f"Job is not waiting for input. Current status: {job.status}"
       logger.warning(msg)
       return JSONResponse(status_code=400, content={"message": msg})
        
    # Start Analysis Task (Phase 2) - VIA CELERY
    resume_analysis_task.delay(task_id, request.rules, request.analysis_config)
    return {"message": "Analysis started"}

@router.get("/status/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str):
    """
    Check the progress of a processing task.
    """
    job = title_task_manager.get_job(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return StatusResponse(
        task_id=job.id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        result=job.result,
        error=job.error,
        report_download_url=f"/api/jobs/{job.id}/report" if job.report_path else None
    )

@router.websocket("/ws/status/{task_id}")
async def websocket_status(websocket: WebSocket, task_id: str):
    """
    Real-time status updates via WebSockets + Redis PubSub.
    """
    await websocket.accept()
    
    # Check if task exists
    job = title_task_manager.get_job(task_id)
    if not job:
        await websocket.close(code=4004, reason="Task not found")
        return

    # Subscribe to Redis Channel
    from app.services.task_manager import redis_client
    
    if not redis_client:
        # Fallback: Polling if no Redis
        try:
            while True:
                job = title_task_manager.get_job(task_id)
                if job:
                    data = {
                        "task_id": job.id,
                        "status": job.status,
                        "progress": job.progress,
                        "message": job.message,
                        "result": job.result
                    }
                    await websocket.send_json(data)
                    if job.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        break
                await asyncio.sleep(2)
        except WebSocketDisconnect:
            pass
        return

    # Redis PubSub Logic
    pubsub = redis_client.pubsub()
    channel = f"task:{task_id}"
    pubsub.subscribe(channel)
    
    try:
        # Send initial state
        initial_data = {
            "task_id": job.id,
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "result": job.result
        }
        await websocket.send_json(initial_data)
        
        while True:
            # Check for messages
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                data = json.loads(message["data"])
                await websocket.send_json(data)
                
                # Close if terminal state
                status = data.get("status")
                if status in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]:
                    break
            
            # Keep alive / Heartbeat can be handled here if needed
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        pass
    finally:
        pubsub.unsubscribe(channel)

@router.post("/jobs/{task_id}/report")
async def generate_persistent_report(task_id: str):
    """
    Generates PDF from the saved Job result and persists it to disk.
    """
    job = title_task_manager.get_job(task_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.result:
        raise HTTPException(400, "Job analysis not ready yet")
    
    try:
        # Extract data from stored result
        result = job.result
        # Ensure outputs dir exists
        output_dir = os.path.join(os.getcwd(), "outputs")  # Backend/outputs
        os.makedirs(output_dir, exist_ok=True)
        
        filename = result.get("filename", "unknown")
        pdf_name = f"{task_id}_{filename}.pdf"
        pdf_path = os.path.join(output_dir, pdf_name)
        
        # Prepare Data for PDF
        analysis_data = result.get("analysis", {}).copy()
        
        # Inject Insights (if available)
        insights_data = result.get("insights", {})
        if insights_data:
            analysis_data["insights"] = insights_data
            
        # Inject Cleaning Report (if available)
        cleaning_data = result.get("cleaning_report", {})
        if cleaning_data:
            analysis_data["cleaning_report"] = cleaning_data
        
        # Generate PDF (Via Celery Task now)
        # However, the endpoint expects to return the path *immediately*? 
        # Actually, generate_persistent_report is usually called by frontend polling until ready.
        # But this endpoint generates ON DEMAND.
        # If we move to Celery, this endpoint becomes "start generation".
        # BUT: For backward compatibility with "Restart & Die" section, let's look at the critique.
        # Critique said "ProcessPoolExecutor(max_workers=3)" is a bottleneck.
        # So we SHOULD use Celery.
        
        # Trigger Celery Task
        generate_pdf_task.delay(task_id)
        
        return {"message": "Report generation started. Check status.", "path": None}
        
    except Exception as e:
        logger.error(f"Persistent report generation failed: {str(e)}")
        raise HTTPException(500, str(e))
        
        # Save to disk
        with open(pdf_path, "wb") as f:
            f.write(pdf_buffer.getbuffer())
        
        # Update Job
        title_task_manager.complete_job(task_id, job.result, report_path=pdf_path)
        
        return {"message": "Report generated", "path": pdf_path}
    
    except Exception as e:
        logger.error(f"Persistent report generation failed: {str(e)}")
        raise HTTPException(500, str(e))


# @router.get("/jobs/{task_id}/report")
# async def download_report(task_id: str):
#     """
#     Downloads the persisted PDF report.
#     """
#     job = title_task_manager.get_job(task_id)
#     if not job or not job.report_path:
#         raise HTTPException(404, "Report not found. Generate it first.")
#     
#     if not os.path.exists(job.report_path):
#         raise HTTPException(404, "Report file missing from disk.")
#         
#     return FileResponse(
#         job.report_path, 
#         media_type="application/pdf", 
#         filename=os.path.basename(job.report_path)
#     )


@router.post("/jobs/{task_id}/chat")
async def chat_with_job(task_id: str, request: ChatRequest):
    """
    Chat with the analyzed data (RAG).
    """
    job = title_task_manager.get_job(task_id)
    if not job or job.status != TaskStatus.COMPLETED:
         raise HTTPException(400, "Job is not completed yet.")
         
    response = await rag_service.chat_with_report(task_id, request.question)
    # response is now a dict with 'answer', 'sources', 'metrics'
    return response


# ─── Issue Ledger Endpoints ──────────────────────────────────────────────────

@router.get("/jobs/{task_id}/issues")
async def get_issues(task_id: str):
    """
    Get the issue ledger for a task.
    """
    job = title_task_manager.get_job(task_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    # Issue ledger should be in job result during WAITING_FOR_USER phase
    if not job.result:
        raise HTTPException(400, "No inspection data available")
    
    issue_ledger = job.result.get("issue_ledger")
    if not issue_ledger:
        return {"issues": [], "summary": {"total": 0}, "locked": False}
    
    return issue_ledger


@router.post("/jobs/{task_id}/issues/{issue_id}/approve")
async def approve_issue(task_id: str, issue_id: str, request: IssueActionRequest = None):
    """
    Approve a single issue for execution.
    """
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    if job.status != TaskStatus.WAITING_FOR_USER:
        raise HTTPException(400, "Job is not in approval phase")
    
    ledger_data = job.result.get("issue_ledger")
    if not ledger_data:
        raise HTTPException(400, "No issue ledger found")
    
    # Find and update the issue
    for issue in ledger_data.get("issues", []):
        if issue["id"] == issue_id:
            issue["status"] = "approved"
            if request and request.note:
                issue["user_note"] = request.note
            # Update the summary
            ledger_data["summary"] = _recalc_summary(ledger_data["issues"])
            # Persist to database
            title_task_manager.update_result(task_id, job.result)
            return {"message": "Issue approved", "issue_id": issue_id}
    
    raise HTTPException(404, "Issue not found")


@router.post("/jobs/{task_id}/issues/{issue_id}/reject")
async def reject_issue(task_id: str, issue_id: str, request: IssueActionRequest = None):
    """
    Reject an issue - fix will not be applied.
    """
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    if job.status != TaskStatus.WAITING_FOR_USER:
        raise HTTPException(400, "Job is not in approval phase")
    
    ledger_data = job.result.get("issue_ledger")
    if not ledger_data:
        raise HTTPException(400, "No issue ledger found")
    
    for issue in ledger_data.get("issues", []):
        if issue["id"] == issue_id:
            issue["status"] = "rejected"
            if request and request.note:
                issue["user_note"] = request.note
            ledger_data["summary"] = _recalc_summary(ledger_data["issues"])
            # Persist to database
            title_task_manager.update_result(task_id, job.result)
            return {"message": "Issue rejected", "issue_id": issue_id}
    
    raise HTTPException(404, "Issue not found")


@router.post("/jobs/{task_id}/issues/approve-all")
async def approve_all_issues(task_id: str):
    """
    Approve all pending issues.
    """
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    if job.status != TaskStatus.WAITING_FOR_USER:
        raise HTTPException(400, "Job is not in approval phase")
    
    ledger_data = job.result.get("issue_ledger")
    if not ledger_data:
        raise HTTPException(400, "No issue ledger found")
    
    count = 0
    for issue in ledger_data.get("issues", []):
        if issue["status"] == "pending":
            issue["status"] = "approved"
            count += 1
    
    ledger_data["summary"] = _recalc_summary(ledger_data["issues"])
    # Persist to database
    title_task_manager.update_result(task_id, job.result)
    return {"message": f"Approved {count} issues", "count": count}


@router.post("/jobs/{task_id}/issues/reject-all")
async def reject_all_issues(task_id: str):
    """
    Reject all pending issues.
    """
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    if job.status != TaskStatus.WAITING_FOR_USER:
        raise HTTPException(400, "Job is not in approval phase")
    
    ledger_data = job.result.get("issue_ledger")
    if not ledger_data:
        raise HTTPException(400, "No issue ledger found")
    
    count = 0
    for issue in ledger_data.get("issues", []):
        if issue["status"] == "pending":
            issue["status"] = "rejected"
            count += 1
    
    ledger_data["summary"] = _recalc_summary(ledger_data["issues"])
    # Persist to database
    title_task_manager.update_result(task_id, job.result)
    return {"message": f"Rejected {count} issues", "count": count}


@router.post("/jobs/{task_id}/issues/lock")
async def lock_issues(task_id: str):
    """
    Lock the issue ledger - no more changes allowed.
    """
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    if job.status != TaskStatus.WAITING_FOR_USER:
        raise HTTPException(400, "Job is not in approval phase")
    
    ledger_data = job.result.get("issue_ledger")
    if not ledger_data:
        raise HTTPException(400, "No issue ledger found")
    
    # Check if there are still pending issues
    pending = sum(1 for i in ledger_data.get("issues", []) if i["status"] == "pending")
    if pending > 0:
        raise HTTPException(400, f"Cannot lock: {pending} issues still pending. Approve or reject all first.")
    
    from datetime import datetime
    ledger_data["locked"] = True
    ledger_data["locked_at"] = datetime.now().isoformat()
    
    # Persist to database
    title_task_manager.update_result(task_id, job.result)
    return {"message": "Issue ledger locked", "summary": ledger_data["summary"]}


def _recalc_summary(issues: list) -> dict:
    """Recalculate issue summary counts."""
    summary = {"pending": 0, "approved": 0, "rejected": 0, "modified": 0, "total": len(issues)}
    for issue in issues:
        status = issue.get("status", "pending")
        if status in summary:
            summary[status] += 1
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3: Transformation DAG Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/jobs/{task_id}/dag")
async def get_transformation_dag(task_id: str):
    """
    Get the complete transformation DAG for a job.
    
    Returns the full audit trail of all data transformations applied.
    """
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    dag_data = job.result.get("transformation_dag")
    if not dag_data:
        raise HTTPException(400, "No transformation DAG available (job may still be processing)")
    
    return dag_data


@router.get("/jobs/{task_id}/dag/export")
async def export_transformation_dag(task_id: str, format: str = "json"):
    """
    Export the transformation DAG as an audit log.
    
    Args:
        format: "json" for ISO-compliant JSON, "csv" for spreadsheet export
        
    Returns:
        Audit log in the specified format
    """
    from app.services.transformation_dag import from_dict
    from fastapi.responses import PlainTextResponse
    
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    dag_data = job.result.get("transformation_dag")
    if not dag_data:
        raise HTTPException(400, "No transformation DAG available")
    
    # Reconstruct DAG from dict
    dag = from_dict(dag_data)
    
    if format.lower() == "csv":
        csv_content = dag.to_audit_csv()
        return PlainTextResponse(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=audit_log_{task_id}.csv"}
        )
    else:
        # Default to JSON audit log
        return dag.to_audit_log()


@router.get("/jobs/{task_id}/dag/summary")
async def get_dag_summary(task_id: str):
    """
    Get a high-level summary of the transformation DAG.
    """
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    dag_data = job.result.get("transformation_dag")
    if not dag_data:
        raise HTTPException(400, "No transformation DAG available")
    
    return dag_data.get("summary", {})


@router.get("/jobs/{task_id}/dag/{node_id}")
async def get_dag_node(task_id: str, node_id: str):
    """
    Get details of a single transformation node.
    """
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    dag_data = job.result.get("transformation_dag")
    if not dag_data:
        raise HTTPException(400, "No transformation DAG available")
    
    nodes = dag_data.get("nodes", {})
    node = nodes.get(node_id)
    
    if not node:
        raise HTTPException(404, f"Node {node_id} not found in DAG")
    
    return node


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 4: Comparison Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/jobs/{task_id}/comparison")
async def get_comparison_report(task_id: str):
    """
    Get the Data Quality Comparison Report (Before vs After).
    
    Returns metrics on how the data quality improved after cleaning.
    """
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    report = job.result.get("comparison_report")
    if not report:
        raise HTTPException(400, "Comparison report not available (job may be old or still processing)")
    
    return report


@router.get("/jobs/{task_id}/report")
async def download_report_pdf(task_id: str):
    """
    Generate and download the comprehensive PDF report (Tier 1-4).
    Includes:
    - Executive Summary
    - Data Quality Grades
    - Cleaning Actions
    - Transformation Audit Trail
    - Comparison Report (Before vs After)
    - Visualizations
    """
    from fastapi.responses import Response
    from app.services.report_generator import generate_pdf_report
    
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found or not completed")
    
    # Extract data from job result
    analysis = job.result
    # Ensure all keys are present for the generator
    if "analysis" in analysis:
        # Flatten structure if needed or pass as designed?
        # generate_pdf_report expects:
        # analysis_results: dict (the whole result bundle usually, or specific keys?)
        # Let's check generate_pdf_report signature:
        # args: analysis_results, charts, filename
        # And inside it looks for: confidence_scores, semantic_analysis, cleaning_report, etc.
        # Our final_result has: info, cleaning_report, analysis (contains decisions, summary, semantic, confidence...), charts...
        
        # We need to restructure slightly to match what report_generator expects
        # report_generator expects a flat dict with keys: 'confidence_scores', 'semantic_analysis', etc.
        # But 'final_result' puts them inside 'analysis'.
        
        # Let's flatten for the generator
        params = {}
        params.update(analysis.get("analysis", {})) # summary, semantic, confidence, advanced...
        params["cleaning_report"] = analysis.get("cleaning_report")
        params["metadata"] = analysis.get("info") # map info -> metadata
        params["comparison_report"] = analysis.get("comparison_report") # Tier 4
        
        charts = analysis.get("charts", {})
        filename = analysis.get("filename", "report.pdf")
        
        try:
            pdf_buffer, meta = await run_in_threadpool(
                generate_pdf_report, 
                params, 
                charts, 
                filename
            )
            
            return Response(
                content=pdf_buffer.getvalue(),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=Report_{filename}.pdf"}
            )
        except Exception as e:
            logger.error(f"PDF Generation failed: {e}")
            raise HTTPException(500, f"Failed to generate PDF: {str(e)}")
            
    else:
        raise HTTPException(400, "Invalid job result format")
