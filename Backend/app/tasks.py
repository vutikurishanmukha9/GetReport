import os
import asyncio
import logging
from typing import Dict, Any, Optional
import threading

def run_async_wrapper(coro):
    """
    Run an async coroutine synchronously, handling existing event loops.
    If a loop is already running (e.g. in Celery eager mode/API thread), run in a separate thread.
    Otherwise, use asyncio.run().
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Loop is running, run in a separate thread to avoid conflict
        logger.info("Event loop detected. Running async task in separate thread.")
        result = None
        exception = None
        
        def run_in_thread():
            nonlocal result, exception
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result = new_loop.run_until_complete(coro)
                new_loop.close()
            except Exception as e:
                exception = e

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        
        if exception:
            raise exception
        return result
    else:
        # No loop running, use standard asyncio.run
        return asyncio.run(coro)

from app.core.celery_app import celery_app
from app.services.task_manager import title_task_manager, TaskStatus
from app.services.storage import get_storage_provider
from app.services.rag_service import rag_service

# Services
from app.services.data_processing import load_dataframe, inspect_dataset, clean_data, get_dataset_info, ParseError
from app.services.issue_ledger import detect_issues
from app.services.analysis import analyze_dataset
from app.services.analysis_config import AnalysisConfig
from app.services.comparison import comparison_service
from app.services.visualization import generate_charts
from app.services.llm_insight import generate_insights
from app.services.report_renderer import generate_pdf_report
from app.services.cleanup import cleanup_old_files

logger = logging.getLogger(__name__)
storage = get_storage_provider()

@celery_app.task(bind=True, name="app.tasks.inspect_file")
def inspect_file_task(self, task_id: str, file_ref: str, filename: str):
    """
    Phase 1: Load Request -> Inspect Data -> Detect Issues -> Pause for User Input.
    """
    try:
        title_task_manager.update_progress(task_id, 10, "Loading file...")
        
        # Resolve path
        file_path = storage.get_absolute_path(file_ref)
        
        # 1. Load
        try:
            df = load_dataframe(file_path)
        except Exception as e:
            raise ParseError(f"Load failed: {e}")

        # 2. Inspect quality
        title_task_manager.update_progress(task_id, 25, "Inspecting data quality...")
        quality_report = inspect_dataset(df)
        
        # 3. Detect issues for Issue Ledger
        title_task_manager.update_progress(task_id, 35, "Detecting data issues...")
        issue_ledger = detect_issues(df)
        issue_count = len(issue_ledger.issues)
        logger.info(f"Task {task_id}: Detected {issue_count} data issues")
        
        # 4. Pause and Persist State
        partial_result = {
            "filename": filename,
            "quality_report": quality_report,
            "issue_ledger": issue_ledger.to_dict(),
            "_file_ref": file_ref,
            "stage": "INSPECTION"
        }
        
        # Update status to WAITING_FOR_USER
        title_task_manager.update_status(task_id, TaskStatus.WAITING_FOR_USER, partial_result)
        title_task_manager.update_progress(task_id, 40, f"Review {issue_count} detected issues")
        
    except Exception as e:
        logger.error(f"Inspection failed: {e}")
        title_task_manager.fail_job(task_id, str(e))
        # Cleanup storage on failure
        if file_ref:
            try: storage.delete(file_ref)
            except: pass

@celery_app.task(bind=True, name="app.tasks.resume_analysis")
def resume_analysis_task(self, task_id: str, rules: Dict[str, Any], analysis_config_dict: Optional[Dict[str, Any]] = None):
    """
    Phase 2: User Rules -> Clean -> Analyze -> Report.
    """
    # 1. Load Job
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        logger.error(f"Task {task_id} invalid for resumption.")
        return
        
    file_ref = job.result.get("_file_ref") or job.result.get("_temp_path")
    filename = job.result.get("filename", "unknown")
    
    if not file_ref:
        title_task_manager.fail_job(task_id, "Source file reference missing.")
        return

    # Resolve path
    file_path = storage.get_absolute_path(file_ref)
    
    if not os.path.exists(file_path):
        title_task_manager.fail_job(task_id, "Source file expired or missing. Please upload again.")
        return

    try:
        title_task_manager.update_progress(task_id, 45, "Applying cleaning rules...")
        
        # Reload
        df = load_dataframe(file_path)
        
        # Clean with Rules (returns DAG for audit trail)
        cleaned_df, cleaning_report, transformation_dag = clean_data(df, rules, None, filename)
        
        # Analyze
        title_task_manager.update_progress(task_id, 60, "Running statistical analysis...")
        dataset_info = get_dataset_info(cleaned_df)
        cleaning_report_dict = cleaning_report.to_dict()
        
        # Extract optional config
        top_cats = rules.get("top_categories", 10)
        
        # Tier 5: Analysis Config
        analysis_config = None
        if analysis_config_dict:
            try:
                analysis_config = AnalysisConfig(**analysis_config_dict)
            except Exception as e:
                logger.warning(f"Invalid analysis config provided: {e}")
                
        analysis_result = analyze_dataset(cleaned_df, top_cats, analysis_config)
        
        # Charts & Insights
        title_task_manager.update_progress(task_id, 75, "Generating charts & insights...")
        
        # Comparison (Before vs After)
        comparison_report = comparison_service.compare(df, cleaned_df)
        
        # Charts (Synchronous)
        charts, _ = generate_charts(cleaned_df)
        
        # Insights (Async - handled safely)
        # Since we are in a synchronous Celery task, we use our robust wrapper
        logger.info(f"Generating insights for task {task_id}...")
        insights_result = run_async_wrapper(generate_insights(analysis_result))
        
        # RAG Ingestion (Fire and Forget via another task)
        # Prepare text
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
        rag_ingest_task.delay(task_id, rag_text) # Dispatch generic task
        
        title_task_manager.update_progress(task_id, 95, "Rendering PDF...")
        
        final_result = {
            "filename": filename,
            "info": dataset_info,
            "cleaning_report": cleaning_report_dict,
            "analysis": analysis_result,
            "charts": charts,
            "insights": insights_result.to_dict(),
            "transformation_dag": transformation_dag.to_dict(),
            "comparison_report": comparison_report.to_dict(),
        }
        
        # Complete
        title_task_manager.complete_job(task_id, final_result)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        title_task_manager.fail_job(task_id, str(e))
        
    finally:
        # Cleanup storage
        if file_ref:
            try: storage.delete(file_ref)
            except: pass

@celery_app.task(name="app.tasks.rag_ingest")
def rag_ingest_task(task_id: str, text: str):
    """
    Ingest text into vector store (RAG).
    Async wrapper for sync Celery.
    """
    try:
        run_async_wrapper(rag_service.ingest_report(task_id, text))
    except Exception as e:
        logger.error(f"RAG Ingestion Task failed: {e}")

@celery_app.task(name="app.tasks.generate_pdf")
def generate_pdf_task(task_id: str):
    """
    Generate PDF for download.
    """
    # 1. Get Job
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        logger.error(f"Job {task_id} not ready for PDF gen.")
        return
        
    try:
        result = job.result
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        filename = result.get("filename", "unknown")
        pdf_name = f"{task_id}_{filename}.pdf"
        pdf_path = os.path.join(output_dir, pdf_name)
        
        # Prepare Data
        analysis_data = result.get("analysis", {}).copy()
        insights_data = result.get("insights", {})
        if insights_data:
            analysis_data["insights"] = insights_data
            
        cleaning_data = result.get("cleaning_report", {})
        if cleaning_data:
            analysis_data["cleaning_report"] = cleaning_data
            
        # Render
        pdf_buffer, metadata = generate_pdf_report(
            analysis_data,
            result.get("charts", {}),
            filename
        )
        
        # Save
        with open(pdf_path, "wb") as f:
            f.write(pdf_buffer.getbuffer())
            
        # Update Job
        title_task_manager.complete_job(task_id, job.result, report_path=pdf_path)
        
    except Exception as e:
        logger.error(f"PDF Gen Task Failed: {e}")
        # We don't fail the whole job if PDF gen fails, but we log it
