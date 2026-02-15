import os
import logging
from typing import Dict, Any, Optional, List
from io import BytesIO

from celery import chain, group
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
from app.services.llm_insight import generate_insights, generate_insights_sync
from app.services.report_renderer import generate_pdf_report
from app.services.cleanup import cleanup_old_files

logger = logging.getLogger(__name__)
storage = get_storage_provider()

# ─── Phase 1: Inspection ─────────────────────────────────────────────────────

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

# ─── Phase 2: Analysis Workflow (Granular Tasks) ─────────────────────────────

@celery_app.task(bind=True, name="app.tasks.clean_data")
def clean_data_task(self, task_id: str, file_ref: str, rules: Dict[str, Any], filename: str):
    """Step 1: Clean Data"""
    try:
        title_task_manager.update_progress(task_id, 45, "Applying cleaning rules...")
        
        file_path = storage.get_absolute_path(file_ref)
        if not os.path.exists(file_path):
             raise ValueError("Source file missing")

        df = load_dataframe(file_path)
        cleaned_df, cleaning_report, transformation_dag = clean_data(df, rules, None, filename)
        
        # Save intermediate cleaned data (Parquet)
        buffer = BytesIO()
        cleaned_df.write_parquet(buffer)
        buffer.seek(0)
        cleaned_file_ref = storage.save_upload(buffer, f"cleaned_{task_id}.parquet")
        
        return {
            "task_id": task_id,
            "file_ref": file_ref, # Keep original
            "cleaned_file_ref": cleaned_file_ref,
            "filename": filename,
            "cleaning_report": cleaning_report.to_dict(),
            "transformation_dag": transformation_dag.to_dict(),
            "top_categories": rules.get("top_categories", 10)
        }
    except Exception as e:
        logger.error(f"Clean Data failed: {e}")
        title_task_manager.fail_job(task_id, f"Cleaning failed: {str(e)}")
        raise self.retry(exc=e, countdown=5, max_retries=3)

@celery_app.task(bind=True, name="app.tasks.analyze_data")
def analyze_data_task(self, context: Dict[str, Any], analysis_config_dict: Optional[Dict[str, Any]] = None):
    """Step 2: Analyze Data"""
    try:
        task_id = context["task_id"]
        title_task_manager.update_progress(task_id, 60, "Running statistical analysis...")
        
        cleaned_path = storage.get_absolute_path(context["cleaned_file_ref"])
        df = load_dataframe(cleaned_path) # Load Parquet (polars auto-detects)
        
        # Tier 5 Config
        analysis_config = None
        if analysis_config_dict:
             try: analysis_config = AnalysisConfig(**analysis_config_dict)
             except: pass

        analysis_result = analyze_dataset(df, context.get("top_categories", 10), analysis_config)
        dataset_info = get_dataset_info(df)
        
        # Compare with original (Optional optimization: load original again? Or assume stats enough?)
        # For comparison report, we need original DF.
        # This might be expensive. Let's load original.
        original_path = storage.get_absolute_path(context["file_ref"])
        original_df = load_dataframe(original_path)
        comparison_report = comparison_service.compare(original_df, df)
        
        context.update({
            "analysis_result": analysis_result,
            "dataset_info": dataset_info,
            "comparison_report": comparison_report.to_dict()
        })
        return context
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        title_task_manager.fail_job(context["task_id"], f"Analysis failed: {str(e)}")
        raise

@celery_app.task(bind=True, name="app.tasks.generate_charts")
def generate_charts_task(self, context: Dict[str, Any]):
    """Step 3a: Generate Charts"""
    try:
        cleaned_path = storage.get_absolute_path(context["cleaned_file_ref"])
        df = load_dataframe(cleaned_path)
        charts, _ = generate_charts(df)
        return {"charts": charts}
    except Exception as e:
        logger.error(f"Charts failed: {e}")
        return {"charts": {}} # Non-critical?

@celery_app.task(bind=True, name="app.tasks.generate_insights")
def generate_insights_task(self, context: Dict[str, Any]):
    """Step 3b: Generate Insights"""
    try:
        task_id = context["task_id"]
        logger.info(f"Generating insights for {task_id}...")
        insights_result = generate_insights_sync(context["analysis_result"])
        return {"insights": insights_result.to_dict()}
    except Exception as e:
         logger.error(f"Insights failed: {e}")
         return {"insights": {}}

@celery_app.task(bind=True, name="app.tasks.compile_report")
def compile_report_task(self, results: List[Dict[str, Any]], context: Dict[str, Any]):
    """Step 4: Compile PDF and Finish"""
    try:
        task_id = context["task_id"]
        title_task_manager.update_progress(task_id, 90, "Compiling Report...")
        
        # Merge results from group
        for res in results:
            context.update(res)
            
        filename = context["filename"]
        
        # RAG Ingest
        _trigger_rag_ingest(task_id, filename, context)
        
        title_task_manager.update_progress(task_id, 95, "Rendering PDF...")
        
        # Prepare Data for PDF
        analysis_data = context["analysis_result"].copy()
        if context.get("insights"):
            analysis_data["insights"] = context["insights"]
        if context.get("cleaning_report"):
            analysis_data["cleaning_report"] = context["cleaning_report"]
            
        pdf_buffer, _ = generate_pdf_report(
            analysis_data,
            context.get("charts", {}),
            filename
        )
        
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        pdf_name = f"{task_id}_{filename}.pdf"
        pdf_path = os.path.join(output_dir, pdf_name)
        
        with open(pdf_path, "wb") as f:
            f.write(pdf_buffer.getbuffer())
            
        final_result = {
            "filename": filename,
            "info": context["dataset_info"],
            "cleaning_report": context["cleaning_report"],
            "analysis": context["analysis_result"],
            "charts": context.get("charts", {}),
            "insights": context.get("insights", {}),
            "transformation_dag": context["transformation_dag"],
            "comparison_report": context["comparison_report"],
            "report_path": pdf_path
        }
        
        title_task_manager.complete_job(task_id, final_result, report_path=pdf_path)
        
        # Cleanup
        try:
             storage.delete(context["cleaned_file_ref"])
             storage.delete(context["file_ref"])
        except: pass
        
        return final_result
        
    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        title_task_manager.fail_job(context["task_id"], f"Report Generation failed: {str(e)}")
        raise

def _trigger_rag_ingest(task_id, filename, context):
    try:
        # Prepare text
        analysis = context["analysis_result"]
        insights = context.get("insights", {}).get("response", "")
        cleaning = context["cleaning_report"]
        info = context["dataset_info"]
        
        rag_text = f"""
        Analysis Report for {filename}
        --- Metadata ---
        Rows: {info.get('rows')}
        Columns: {info.get('columns')}
        --- Summary Statistics ---
        {str(analysis.get('summary', {}))}
        --- Insights ---
        {insights}
        --- Cleaning Actions ---
        {cleaning}
        """
        rag_ingest_task.delay(task_id, rag_text)
    except Exception as e:
        logger.warning(f"Failed to trigger RAG ingest: {e}")

@celery_app.task(bind=True, name="app.tasks.resume_analysis")
def resume_analysis_task(self, task_id: str, rules: Dict[str, Any], analysis_config_dict: Optional[Dict[str, Any]] = None):
    """
    Phase 2 Entry Point: Launches the Workflow.
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
        
    # 2. Build and Launch Chain
    # Clean -> Analyze -> [Charts, Insights] -> Compile
    # Note: 'compile_report_task' needs 'context' which is the output of 'analyze_data_task', 
    # BUT 'group' results are passed as the first arg. 
    # So compile_report signature is (results, context).
    # To pass 'context' to compile_report, analyze_data_task must return it.
    
    workflow = chain(
        clean_data_task.s(task_id, file_ref, rules, filename),
        analyze_data_task.s(analysis_config_dict=analysis_config_dict),
        # Context is now passed implicitly as the result of analyze_data_task
        # We need to broadcast context to group, and then gather results + context
        # Celery 'chord' is perfect: header=group, body=callback
        # But we need to pass the context to EACH task in the group.
        # This requires a bit of dynamic chain construction or explicit parameter passing.
        #
        # Simpler: 
        # clean -> analyze (returns context) -> 
        #   compile takes context, runs charts/insights internally? NO, defeats parallelism.
        #
        # Correct Celery Pattern:
        # Task A returns Context.
        # Task B (Chord):
        #   Header: [Charts(Context), Insights(Context)]
        #   Body: Compile(Results of Group, Context??) -> Wait, Body only gets Group Results.
        #
        # Solution: Pass Context in the Group Results or use a wrapper.
        # OR: compile_report_task(results, context) -- but how to get context into the args of compile_report_task if it was output of Analyze?
        #
        # We can implement a "Dispatcher" task that creates the chord.
        create_parallel_steps.s()
    )
    
    # We use self.replace to swap this task with the chain
    raise self.replace(workflow)

@celery_app.task(bind=True, name="app.tasks.create_parallel_steps")
def create_parallel_steps(self, context: Dict[str, Any]):
    """
    Dynamic dispatch to create the parallel group + final callback.
    Receives 'context' from analyze_data_task.
    """
    # Create the group
    parallel_tasks = group(
        generate_charts_task.s(context),
        generate_insights_task.s(context)
    )
    
    # Create the callback (Body)
    # Note: explicit 'context' passed as a partial argument to the body
    callback = compile_report_task.s(context)
    
    # Return chord
    return (parallel_tasks | callback).delay()

@celery_app.task(name="app.tasks.rag_ingest")
def rag_ingest_task(task_id: str, text: str):
    """
    Ingest text into vector store (RAG).
    Uses the blocking sync method to avoid asyncio.run() issues.
    """
    try:
        rag_service.ingest_report_blocking(task_id, text)
    except Exception as e:
        logger.error(f"RAG Ingestion Task failed: {e}")

@celery_app.task(name="app.tasks.generate_pdf")
def generate_pdf_task(task_id: str):
    """
    Generate PDF for download (Standalone / Re-generation).
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
