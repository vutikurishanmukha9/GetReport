import os
import logging
from typing import Dict, Any, Optional, List
from io import BytesIO
import json

try:
    from celery import chain, group
    from app.core.celery_app import celery_app
except ImportError:
    chain = group = None
    class DummyCelery:
        def task(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    celery_app = DummyCelery()
from app.services.task_manager import title_task_manager, TaskStatus
from app.services.storage import get_storage_provider
from app.services.rag_service import rag_service

# Services
from app.services.data_processing import load_dataframe, inspect_dataset, clean_data, get_dataset_info, ParseError
from app.services.issue_ledger import detect_issues
from app.services.analysis import analyze_dataset
from app.services.analysis_config import AnalysisConfig
from app.services.dataset_versioning import build_schema_profile, build_historical_comparison
from app.services.comparison import comparison_service
from app.services.visualization import generate_charts
from app.services.llm_insight import generate_insights, generate_insights_sync
from app.services.report_generator import generate_pdf_report
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
            try:
                storage.delete(file_ref)
            except Exception:
                pass

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
        issue_ledger = detect_issues(df)
        schema_profile = build_schema_profile(df)
        config_snapshot = analysis_config.snapshot() if analysis_config else AnalysisConfig.default().snapshot()
        previous_job = title_task_manager.find_previous_completed_job(task_id, context["filename"])
        historical_comparison = build_historical_comparison(
            previous_job.id if previous_job else None,
            previous_job.result if previous_job else None,
            schema_profile,
        )
        
        # Compare with original (Optional optimization: load original again? Or assume stats enough?)
        # For comparison report, we need original DF.
        # This might be expensive. Let's load original.
        original_path = storage.get_absolute_path(context["file_ref"])
        original_df = load_dataframe(original_path)
        comparison_report = comparison_service.compare(original_df, df)
        
        # Pass references, not data, to prevent Redis "fat payload" issue
        analysis_bytes = json.dumps(analysis_result).encode("utf-8")
        analysis_ref = storage.save_upload(BytesIO(analysis_bytes), f"analysis_{task_id}.json")
        
        comparison_bytes = json.dumps(comparison_report.to_dict()).encode("utf-8")
        comparison_ref = storage.save_upload(BytesIO(comparison_bytes), f"comparison_{task_id}.json")
        
        issue_bytes = json.dumps(issue_ledger.to_dict()).encode("utf-8")
        issue_ref = storage.save_upload(BytesIO(issue_bytes), f"issue_ledger_{task_id}.json")
        
        context.update({
            "analysis_result_ref": analysis_ref,
            "dataset_info": dataset_info,
            "comparison_report_ref": comparison_ref,
            "issue_ledger_ref": issue_ref,
            "analysis_config": config_snapshot,
            "schema_profile": schema_profile,
            "historical_comparison": historical_comparison,
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
        task_id = context["task_id"]
        title_task_manager.update_progress(task_id, 70, "Generating visualizations & charts...")
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
        title_task_manager.update_progress(task_id, 80, "Generating AI insights & narratives...")
        logger.info(f"Generating insights for {task_id}...")
        # Load analysis_result from storage reference
        analysis_ref = context["analysis_result_ref"]
        analysis_path = storage.get_absolute_path(analysis_ref)
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis_result = json.load(f)
            
        insights_result = generate_insights_sync(analysis_result)
        return {"insights": insights_result.to_dict()}
    except Exception as e:
         logger.error(f"Insights failed: {e}")
         return {"insights": {}}

@celery_app.task(bind=True, name="app.tasks.compile_report")
def compile_report_task(self, results: List[Dict[str, Any]], context: Dict[str, Any]):
    """Step 4: Compile PDF and Finish"""
    try:
        task_id = context["task_id"]
        title_task_manager.update_progress(task_id, 90, "Compiling PDF Report...")
        
        # Merge results from group
        for res in results:
            context.update(res)
            
        filename = context["filename"]
        
        # Load large data objects from references
        analysis_ref = context["analysis_result_ref"]
        analysis_path = storage.get_absolute_path(analysis_ref)
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis_result = json.load(f)

        comparison_ref = context["comparison_report_ref"]
        comparison_path = storage.get_absolute_path(comparison_ref)
        with open(comparison_path, "r", encoding="utf-8") as f:
            comparison_report = json.load(f)

        issue_ref = context["issue_ledger_ref"]
        issue_path = storage.get_absolute_path(issue_ref)
        with open(issue_path, "r", encoding="utf-8") as f:
            issue_ledger = json.load(f)
        
        # RAG Ingest
        _trigger_rag_ingest(task_id, filename, context, analysis_result)
        
        title_task_manager.update_progress(task_id, 95, "Rendering PDF...")
        
        # Prepare Data for PDF
        analysis_data = analysis_result.copy()
        if context.get("insights"):
            analysis_data["insights"] = context["insights"]
        if context.get("cleaning_report"):
            analysis_data["cleaning_report"] = context["cleaning_report"]
        if comparison_report:
            analysis_data["comparison_report"] = comparison_report
        if issue_ledger:
            analysis_data["issue_ledger"] = issue_ledger
            
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
            "analysis": analysis_result,
            "charts": context.get("charts", {}),
            "insights": context.get("insights", {}),
            "transformation_dag": context["transformation_dag"],
            "comparison_report": comparison_report,
            "issue_ledger": issue_ledger,
            "analysis_config": context["analysis_config"],
            "schema_profile": context["schema_profile"],
            "historical_comparison": context["historical_comparison"],
            # Keep the cleaned artifact available for user-requested exports.
            "cleaned_file_ref": context["cleaned_file_ref"],
            "report_path": pdf_path
        }
        
        title_task_manager.complete_job(task_id, final_result, report_path=pdf_path)
        
        # Cleanup
        try:
             storage.delete(context["file_ref"])
             storage.delete(context["analysis_result_ref"])
             storage.delete(context["comparison_report_ref"])
             storage.delete(context["issue_ledger_ref"])
        except: pass
        
        return final_result
        
    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        title_task_manager.fail_job(context["task_id"], f"Report Generation failed: {str(e)}")
        raise

def _build_rag_narrative(filename, analysis, cleaning, insights_text, info):
    """
    Convert structured analysis results into natural-language prose for RAG embedding.
    Natural language produces far better cosine similarity scores against user questions
    than raw JSON blobs.
    """
    sections = []
    
    # 1. Dataset overview
    rows = info.get('rows', 'N/A')
    columns = info.get('columns', [])
    col_count = len(columns) if isinstance(columns, list) else columns
    sections.append(
        f"This dataset '{filename}' contains {rows} rows and {col_count} columns. "
        f"The columns include: {', '.join(columns[:20]) if isinstance(columns, list) else 'N/A'}."
    )
    
    # 2. Summary statistics as prose
    summary = analysis.get('summary', {})
    if summary and isinstance(summary, dict):
        stat_lines = []
        for col, stats in list(summary.items())[:15]:
            if isinstance(stats, dict):
                parts = []
                if 'mean' in stats: parts.append(f"average of {stats['mean']}")
                if 'min' in stats: parts.append(f"minimum of {stats['min']}")
                if 'max' in stats: parts.append(f"maximum of {stats['max']}")
                if 'std' in stats: parts.append(f"standard deviation of {stats['std']}")
                if 'null_count' in stats: parts.append(f"{stats['null_count']} missing values")
                if parts:
                    stat_lines.append(f"The column '{col}' has {', '.join(parts)}.")
        if stat_lines:
            sections.append("Summary statistics for key columns:\n" + "\n".join(stat_lines))
    
    # 3. Correlations as prose
    strong_corrs = analysis.get('strong_correlations', [])
    if strong_corrs and isinstance(strong_corrs, list):
        corr_lines = []
        for c in strong_corrs[:10]:
            if isinstance(c, dict):
                col_a = c.get('column_a', c.get('col1', ''))
                col_b = c.get('column_b', c.get('col2', ''))
                r = c.get('r_value', c.get('correlation', 0))
                direction = c.get('direction', 'positive' if r > 0 else 'negative')
                strength = c.get('strength', 'strong')
                corr_lines.append(
                    f"There is a {strength} {direction} correlation between "
                    f"'{col_a}' and '{col_b}' with correlation coefficient {r:.4f}. "
                    f"When {col_a.replace('_', ' ')} increases, "
                    f"{col_b.replace('_', ' ')} {'also increases' if direction == 'positive' else 'tends to decrease'}."
                )
        if corr_lines:
            sections.append("Key correlations and feature relationships:\n" + "\n".join(corr_lines))
    
    # 4. Outliers as prose
    outliers = analysis.get('outliers', {})
    if outliers and isinstance(outliers, dict):
        outlier_lines = []
        for col, details in list(outliers.items())[:10]:
            if isinstance(details, dict):
                cnt = details.get('count', 0)
                pct = details.get('percentage', 0)
                outlier_lines.append(
                    f"Column '{col}' has {cnt} outlier values ({pct:.1f}% of records), "
                    f"representing unusually high or low data points."
                )
        if outlier_lines:
            sections.append("Outlier and anomaly detection results:\n" + "\n".join(outlier_lines))
    
    # 5. Data quality and confidence
    confidence = analysis.get('confidence_scores', {})
    if confidence and isinstance(confidence, dict):
        quality_score = confidence.get('dataset_confidence', confidence.get('overall_score', ''))
        grade = confidence.get('dataset_grade', confidence.get('grade', ''))
        if quality_score:
            sections.append(
                f"The overall data quality confidence score is {quality_score}% "
                f"(grade: {grade}). "
                f"This reflects schema consistency, null rate, and data integrity checks."
            )
    
    # 6. AI-generated insights
    if insights_text:
        clean_text = insights_text.replace('**', '').replace('<b>', '').replace('</b>', '')
        clean_text = clean_text.replace('<i>', '').replace('</i>', '')
        sections.append(f"AI-generated strategic insights:\n{clean_text}")
    
    # 7. Data cleaning actions
    if cleaning and isinstance(cleaning, dict):
        cleaning_parts = []
        dup = cleaning.get('duplicate_rows_removed', 0)
        empty = cleaning.get('empty_rows_dropped', 0)
        num_fill = cleaning.get('numeric_nans_filled', 0)
        cat_fill = cleaning.get('categorical_nans_filled', 0)
        if dup: cleaning_parts.append(f"removed {dup} duplicate rows")
        if empty: cleaning_parts.append(f"dropped {empty} empty rows")
        if num_fill: cleaning_parts.append(f"imputed {num_fill} missing numeric values")
        if cat_fill: cleaning_parts.append(f"imputed {cat_fill} missing categorical values")
        if cleaning_parts:
            sections.append(f"Data cleaning actions performed: {', '.join(cleaning_parts)}.")
        else:
            sections.append("Data cleaning: No significant cleaning actions were required. The dataset was already in good shape.")
    
    return "\n\n".join(sections)


def _trigger_rag_ingest(task_id, filename, context, analysis_result):
    try:
        analysis = analysis_result or {}
        raw_insights = context.get("insights", {})
        if isinstance(raw_insights, dict):
            insights_text = raw_insights.get("insights_text", "")
        else:
            insights_text = str(raw_insights)
        
        cleaning = context.get("cleaning_report", {})
        info = context.get("dataset_info", {})
        
        rag_text = _build_rag_narrative(filename, analysis, cleaning, insights_text, info)
        rag_ingest_task.delay(task_id, rag_text)
        logger.info(f"Triggered enhanced RAG ingestion for task {task_id}")
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
        if result.get("comparison_report"):
            analysis_data["comparison_report"] = result["comparison_report"]
        if result.get("issue_ledger"):
            analysis_data["issue_ledger"] = result["issue_ledger"]
            
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
        title_task_manager.set_report_status(task_id, "failed")
