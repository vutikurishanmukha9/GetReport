from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import asyncio

from app.services.data_processing import (
    load_dataframe, clean_data, get_dataset_info, 
    UnsupportedFileTypeError, FileTooLargeError, EmptyFileError, ParseError
)
from app.services.analysis import (
    analyze_dataset, EmptyDatasetError, InsufficientDataError, AnalysisError
)
from app.services.visualization import generate_charts
from app.services.report_generator import generate_pdf_report
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
    status: TaskStatus
    progress: int
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ─── Background Processor ────────────────────────────────────────────────────
async def process_file_in_background(task_id: str, file_content: bytes, filename: str):
    """
    Orchestrates the full analysis pipeline while updating the task status.
    Running in background allows the API to return immediately.
    """
    try:
        # Step 1: Loading
        title_task_manager.update_progress(task_id, 10, "Loading file...")
        from io import BytesIO
        import pandas as pd
        
        # We need to manually recreate the 'UploadFile' behavior or just modify load_dataframe 
        # to accept bytes, but for minimal refactor, we can wrap bytes in BytesIO 
        # and mock the UploadFile structure if load_dataframe is strict, 
        # OR better: refactor load_dataframe to take bytes.
        # Actually, let's just make a simple ad-hoc adapter since we already read the bytes.
        
        # Determine extension
        ext = ""
        if filename.endswith(".csv"): ext = ".csv"
        elif filename.endswith(".xlsx"): ext = ".xlsx"
        elif filename.endswith(".xls"): ext = ".xls"
        
        buffer = BytesIO(file_content)
        
        if ext == ".csv":
            df = pd.read_csv(buffer)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(buffer)
        else:
            raise UnsupportedFileTypeError(f"Unsupported extension: {ext}")

        # Basic post-load validation (reusing logic from data_processing roughly)
        if df.empty: raise EmptyFileError("File is empty")

        # Step 2: Cleaning
        title_task_manager.update_progress(task_id, 30, "Cleaning data...")
        # Clean data (CPU bound)
        cleaned_df, cleaning_report = await run_in_threadpool(clean_data, df)
        
        # Step 3: Analysis
        title_task_manager.update_progress(task_id, 50, "Analyzing statistics...")
        dataset_info = await run_in_threadpool(get_dataset_info, cleaned_df)
        cleaning_report_dict = cleaning_report.to_dict()
        analysis_result = await run_in_threadpool(analyze_dataset, cleaned_df)
        
        # Step 4: Charts
        title_task_manager.update_progress(task_id, 70, "Generating visualizations...")
        charts, _ = await run_in_threadpool(generate_charts, cleaned_df)
        
        # Step 5: AI Insights (Network bound)
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
        logger.info(f"Task {task_id} completed successfully.")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        title_task_manager.fail_job(task_id, str(e))

# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/upload", response_model=TaskResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Initiates file processing. 
    Returns a Task ID immediately. Client should poll /status/{task_id}.
    """
    try:
        # Pre-validate extension before accepting
        if not file.filename.lower().endswith(('.csv', '.xls', '.xlsx')):
             raise HTTPException(400, "Invalid file type")
             
        # Create Task
        task_id = title_task_manager.create_job()
        
        # Read file into memory immediately (fast for <50MB) 
        # so we can close the connection and pass data to background
        content = await file.read()
        
        # Start Background Task
        background_tasks.add_task(
            process_file_in_background, 
            task_id, 
            content, 
            file.filename
        )
        
        return TaskResponse(
            task_id=task_id, 
            message="File uploaded. Processing started."
        )

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        error=job.error
    )

@router.post("/generate-report")
async def generate_report_endpoint(request: ReportRequest):
    """
    Generates a PDF report from the provided analysis data and charts.
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
