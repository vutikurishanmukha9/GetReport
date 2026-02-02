from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import os
import asyncio

from app.services.data_processing import (
    load_dataframe, clean_data, get_dataset_info, 
    UnsupportedFileTypeError, FileTooLargeError, EmptyFileError, ParseError
)
from app.services.analysis import (
    analyze_dataset, EmptyDatasetError, InsufficientDataError, AnalysisError
)
from app.services.visualization import generate_charts
from app.services.report_renderer import generate_pdf_report  # NEW: HTML-to-PDF engine
from app.services.llm_insight import generate_insights
from app.services.task_manager import title_task_manager, TaskStatus

logger = logging.getLogger(__name__)
router = APIRouter()

# ─── Data Models ─────────────────────────────────────────────────────────────
class ReportRequest(BaseModel):
    filename: str
    analysis: Dict[str, Any]
    charts: Dict[str, Any]

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

# ─── Background Processor ────────────────────────────────────────────────────
async def process_file_in_background(task_id: str, file_path: str, filename: str):
    """
    Orchestrates the full analysis pipeline using Polars (High Performance).
    """
    try:
        # Step 1: Loading
        title_task_manager.update_progress(task_id, 10, "Loading file (Polars)...")
        from app.services.data_processing import load_dataframe
        
        try:
            # load_dataframe now returns a Polars LazyFrame or DataFrame
            df = load_dataframe(file_path)
            # if using LazyFrame, we might collect here or pass lazy.
            # Current implementation returns DataFrame (eager) for simplicity with existing logic.
        except Exception as load_err:
            raise ParseError(f"Failed to load: {str(load_err)}")

        # Step 2: Cleaning
        title_task_manager.update_progress(task_id, 30, "Cleaning data...")
        cleaned_df, cleaning_report = await run_in_threadpool(clean_data, df)
        
        # Step 3: Analysis
        title_task_manager.update_progress(task_id, 50, "Analyzing statistics...")
        dataset_info = await run_in_threadpool(get_dataset_info, cleaned_df)
        cleaning_report_dict = cleaning_report.to_dict()
        analysis_result = await run_in_threadpool(analyze_dataset, cleaned_df)
        
        # Step 4: Charts
        title_task_manager.update_progress(task_id, 70, "Generating visualizations...")
        charts, _ = await run_in_threadpool(generate_charts, cleaned_df)
        
        # Step 5: AI Insights
        title_task_manager.update_progress(task_id, 85, "Generating AI insights...")
        insights_result = await generate_insights(analysis_result)
        
        # Step 6: Finalize
        title_task_manager.update_progress(task_id, 95, "Finalizing report...")
        
        final_result = {
            "filename": filename,
            "info": dataset_info,
            "cleaning_report": cleaning_report_dict,
            "analysis": analysis_result,
            "charts": charts,
            "insights": insights_result.to_dict()
        }
        
        title_task_manager.complete_job(task_id, final_result)
        logger.info(f"Task {task_id} completed successfully (Polars Engine).")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        title_task_manager.fail_job(task_id, str(e))
    
    finally:
        # cleanup input file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
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
    
    try:
        # Pre-validate extension
        if not file.filename.lower().endswith(('.csv', '.xls', '.xlsx')):
             raise HTTPException(400, "Invalid file type. Only CSV and Excel supported.")
             
        # Create Task
        task_id = title_task_manager.create_job(file.filename)
        
        # Stream file to disk using tempfile
        # We use a temp directory that allows persistence during the background task
        # mkstemp creates a file that we must manage (delete manually)
        fd, temp_path = tempfile.mkstemp(suffix=f"_{file.filename}")
        
        try:
            with os.fdopen(fd, 'wb') as tmp:
                shutil.copyfileobj(file.file, tmp)
        except Exception as write_err:
            os.remove(temp_path)
            raise write_err
        
        # Start Background Task (processing + cleanup)
        background_tasks.add_task(
            process_file_in_background, 
            task_id, 
            temp_path, 
            file.filename
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
    rules: Dict[str, Any], 
    background_tasks: BackgroundTasks
):
    """
    Stage 2: User approves cleaning rules and starts full analysis.
    """
    job = title_task_manager.get_job(task_id)
    if not job:
        raise HTTPException(404, "Job not found")
        
    # Check if job is in correct state (WAITING_FOR_USER)
    # if job.status != "WAITING_FOR_USER":
    #    return JSONResponse(status_code=400, content={"message": "Job is not waiting for input"})
        
    background_tasks.add_task(resume_analysis_task, task_id, rules)
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
        
        # Generate PDF
        pdf_buffer, metadata = await run_in_threadpool(
            generate_pdf_report,
            result.get("analysis", {}),
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

@router.post("/generate-report")
async def generate_report_endpoint(request: ReportRequest):
    """
    Legacy/On-the-fly endpoint. 
    """
    try:
        # Wrap PDF generation in threadpool too just in case it's heavy
        pdf_buffer, metadata = await run_in_threadpool(
            generate_pdf_report,
            request.analysis,
            request.charts,
            request.filename
        )
        
        logger.info(f"Report generated successfully: {metadata}")
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=Report_{request.filename}.pdf",
                "X-Report-Metadata": str(metadata)
            }
        )

    except Exception as e:
        logger.error(f"Report generation endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
