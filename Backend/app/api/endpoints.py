from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
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

logger = logging.getLogger(__name__)
router = APIRouter()

# Global Process Pool for CPU-bound tasks (PDF Generation)
process_pool = ProcessPoolExecutor(max_workers=3)

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

class IssueActionRequest(BaseModel):
    note: str = ""

class IssueModifyRequest(BaseModel):
    fix_code: str
    note: str = ""

# ─── Background Processor ────────────────────────────────────────────────────
async def run_inspection_task(task_id: str, file_path: str, filename: str):
    """
    Phase 1: Load Request -> Inspect Data -> Detect Issues -> Pause for User Input.
    """
    try:
        title_task_manager.update_progress(task_id, 10, "Loading file...")
        from app.services.data_processing import load_dataframe, inspect_dataset
        
        # 1. Load
        try:
            df = load_dataframe(file_path)
        except Exception as e:
            raise ParseError(f"Load failed: {e}")

        # 2. Inspect quality
        title_task_manager.update_progress(task_id, 25, "Inspecting data quality...")
        quality_report = await run_in_threadpool(inspect_dataset, df)
        
        # 3. Detect issues for Issue Ledger
        title_task_manager.update_progress(task_id, 35, "Detecting data issues...")
        issue_ledger = await run_in_threadpool(detect_issues, df)
        issue_count = len(issue_ledger.issues)
        logger.info(f"Task {task_id}: Detected {issue_count} data issues")
        
        # 4. Pause and Persist State
        # We store temp_path in result so Phase 2 can find it.
        partial_result = {
            "filename": filename,
            "quality_report": quality_report,
            "issue_ledger": issue_ledger.to_dict(),  # NEW: Issue Ledger
            "_temp_path": file_path,
            "stage": "INSPECTION"  # Required for frontend to recognize this phase
        }
        
        # Update status to WAITING_FOR_USER
        logger.info(f"Task {task_id}: Inspection Complete. {issue_count} issues detected. Setting WAITING_FOR_USER.")
        task_manager_response = title_task_manager.update_status(task_id, TaskStatus.WAITING_FOR_USER, partial_result)
        logger.info(f"Task {task_id}: Status Update Triggered.")
        
        title_task_manager.update_progress(task_id, 40, f"Review {issue_count} detected issues")
        
    except Exception as e:
        logger.error(f"Inspection failed: {e}")
        title_task_manager.fail_job(task_id, str(e))
        # Cleanup if failed
        if os.path.exists(file_path):
            try: os.remove(file_path)
            except: pass


async def resume_analysis_task(task_id: str, rules: Dict[str, Any]):
    """
    Phase 2: User Rules -> Clean -> Analyze -> Report.
    """
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        logger.error(f"Task {task_id} invalid for resumption.")
        return
        
    file_path = job.result.get("_temp_path")
    filename = job.result.get("filename", "unknown")
    
    if not file_path or not os.path.exists(file_path):
        title_task_manager.fail_job(task_id, "Source file expired or missing. Please upload again.")
        return

    try:
        title_task_manager.update_progress(task_id, 45, "Applying cleaning rules...")
        title_task_manager.update_progress(task_id, 45, "Applying cleaning rules...")
        from app.services.data_processing import load_dataframe, clean_data, get_dataset_info
        from app.services.analysis import analyze_dataset  # Import from analysis.py (has top_categories param)
        
        # Reload (Polars is fast)
        df = load_dataframe(file_path)
        
        # Clean with Rules
        cleaned_df, cleaning_report = await run_in_threadpool(clean_data, df, rules)
        
        # Analyze
        title_task_manager.update_progress(task_id, 60, "Running statistical analysis...")
        dataset_info = await run_in_threadpool(get_dataset_info, cleaned_df)
        cleaning_report_dict = cleaning_report.to_dict()
        
        # Extract optional config
        top_cats = rules.get("top_categories", 10)
        analysis_result = await run_in_threadpool(analyze_dataset, cleaned_df, top_cats)
        
        # Parallel Execution: Charts (CPU) & Insights (IO)
        title_task_manager.update_progress(task_id, 75, "Generating charts & insights...")
        
        # Wrap chart generation to be awaitable
        charts_task = run_in_threadpool(generate_charts, cleaned_df)
        insights_task = generate_insights(analysis_result)
        
        # Run both concurrently
        results = await asyncio.gather(charts_task, insights_task)
        charts_data, _ = results[0]
        insights_result = results[1]
        
        charts = charts_data
        
        # ─── RAG Ingestion ───
        # Build a text representation of the findings for the chatbot
        rag_text = f"""
        Analysis Report for {filename}
        
        --- Metadata ---
        Rows: {dataset_info.get('rows')}
        Columns: {dataset_info.get('columns')}
        
        --- Summary Statistics ---
        {str(analysis_result.get('summary', {}))}
        
        --- Insights ---
        {insights_result.to_dict().get('response', '')}
        
        --- Cleaning Actions ---
        {cleaning_report.to_dict()}
        """
        # Fire-and-forget ingestion so we don't block the report generation
        # The Chat feature will become available a few seconds after the report is ready
        asyncio.create_task(rag_service.ingest_report(task_id, rag_text))
        
        title_task_manager.update_progress(task_id, 95, "Rendering PDF...")
        
        final_result = {
            "filename": filename,
            "info": dataset_info,
            "cleaning_report": cleaning_report_dict,
            "analysis": analysis_result,
            "charts": charts,
            "insights": insights_result.to_dict()
        }
        
        title_task_manager.complete_job(task_id, final_result)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        title_task_manager.fail_job(task_id, str(e))
        
    finally:
        # Now we can delete the temp file
        if os.path.exists(file_path):
            try: os.remove(file_path)
            except: pass

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
        
        # Stream file to disk using tempfile
        # We use a temp directory that allows persistence during the background task
        # mkstemp creates a file that we must manage (delete manually)
        fd, temp_path = tempfile.mkstemp(suffix=f"_{safe_filename}")
        
        try:
            with os.fdopen(fd, 'wb') as tmp:
                shutil.copyfileobj(file.file, tmp)
        except Exception as write_err:
            os.remove(temp_path)
            raise write_err
        
        # Start Inspection Task (Phase 1)
        background_tasks.add_task(
            run_inspection_task, 
            task_id, 
            temp_path, 
            safe_filename
        )
        
        # Also schedule strict hygiene cleanup for old reports
        from app.services.cleanup import cleanup_old_files
        output_dir = os.path.join(os.getcwd(), "outputs")
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
        
    background_tasks.add_task(resume_analysis_task, task_id, request.rules)
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
        
        # Generate PDF (CPU Bound - Offload to Process)
        # We use a ProcessPool to avoid GIL blocking the API during PDF generation
        
        loop = asyncio.get_running_loop()
        pdf_buffer, metadata = await loop.run_in_executor(
            process_pool,
            generate_pdf_report,
            analysis_data,
            result.get("charts", {}),
            filename
        )
        
        # Save to disk
        with open(pdf_path, "wb") as f:
            f.write(pdf_buffer.getbuffer())
        
        # Update Job
        title_task_manager.complete_job(task_id, job.result, report_path=pdf_path)
        
        return {"message": "Report generated", "path": pdf_path}
    
    except Exception as e:
        logger.error(f"Persistent report generation failed: {str(e)}")
        raise HTTPException(500, str(e))

@router.get("/jobs/{task_id}/report")
async def download_report(task_id: str):
    """
    Downloads the persisted PDF report.
    """
    job = title_task_manager.get_job(task_id)
    if not job or not job.report_path:
        raise HTTPException(404, "Report not found. Generate it first.")
    
    if not os.path.exists(job.report_path):
        raise HTTPException(404, "Report file missing from disk.")
        
    return FileResponse(
        job.report_path, 
        media_type="application/pdf", 
        filename=os.path.basename(job.report_path)
    )

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
