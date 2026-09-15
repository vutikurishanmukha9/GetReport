"""
Ultra-lean, async-native ARQ background task definitions for GetReport.
Consumes ~25MB RAM under load (compared to 120MB+ for Celery).
Can be executed:
  1. Via ARQ Worker in production (`arq app.tasks_arq.WorkerSettings`) with Redis.
  2. Directly in-memory via task_dispatcher on local dev (zero Redis requirement).
"""
import os
import gc
import json
import logging
import asyncio
from io import BytesIO
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.core.config import settings
from app.core.task_dispatcher import register_task
from app.services.task_manager import title_task_manager, TaskStatus
from app.services.storage import get_storage_provider
from app.services.rag_service import rag_service

# Domain Services
from app.services.data_processing import (
    load_dataframe,
    inspect_dataset,
    clean_data,
    get_dataset_info,
    ParseError,
)
from app.services.issue_ledger import detect_issues
from app.services.analysis import analyze_dataset
from app.services.analysis_config import AnalysisConfig
from app.services.dataset_versioning import build_schema_profile, build_historical_comparison
from app.services.comparison import comparison_service
from app.services.visualization import generate_charts
from app.services.llm_insight import generate_insights_sync
from app.services.report_generator import generate_pdf_report

logger = logging.getLogger(__name__)
storage = get_storage_provider()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: RAG Narrative Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_narrative(filename: str, analysis: dict, cleaning: dict, insights_text: str, info: dict) -> str:
    """
    Convert structured analysis results into natural-language prose for RAG embedding.
    """
    sections: List[str] = []

    # 1. Dataset overview
    rows = info.get("rows", "N/A")
    columns = info.get("columns", [])
    col_count = len(columns) if isinstance(columns, list) else columns
    sections.append(
        f"This dataset '{filename}' contains {rows} rows and {col_count} columns. "
        f"The columns include: {', '.join(columns[:20]) if isinstance(columns, list) else 'N/A'}."
    )

    # 2. Summary statistics as prose
    summary = analysis.get("summary", {})
    if summary and isinstance(summary, dict):
        stat_lines = []
        for col, stats in list(summary.items())[:15]:
            if isinstance(stats, dict):
                parts = []
                if "mean" in stats:
                    parts.append(f"average of {stats['mean']}")
                if "min" in stats:
                    parts.append(f"minimum of {stats['min']}")
                if "max" in stats:
                    parts.append(f"maximum of {stats['max']}")
                if "std" in stats:
                    parts.append(f"standard deviation of {stats['std']}")
                if "null_count" in stats:
                    parts.append(f"{stats['null_count']} missing values")
                if parts:
                    stat_lines.append(f"The column '{col}' has {', '.join(parts)}.")
        if stat_lines:
            sections.append("Summary statistics for key columns:\n" + "\n".join(stat_lines))

    # 3. Correlations as prose
    strong_corrs = analysis.get("strong_correlations", [])
    if strong_corrs and isinstance(strong_corrs, list):
        corr_lines = []
        for c in strong_corrs[:10]:
            if isinstance(c, dict):
                col_a = c.get("column_a", c.get("col1", ""))
                col_b = c.get("column_b", c.get("col2", ""))
                r = c.get("r_value", c.get("correlation", 0))
                direction = c.get("direction", "positive" if r > 0 else "negative")
                strength = c.get("strength", "strong")
                corr_lines.append(
                    f"There is a {strength} {direction} correlation between "
                    f"'{col_a}' and '{col_b}' with correlation coefficient {r:.4f}. "
                    f"When {col_a.replace('_', ' ')} increases, "
                    f"{col_b.replace('_', ' ')} {'also increases' if direction == 'positive' else 'tends to decrease'}."
                )
        if corr_lines:
            sections.append("Key correlations and feature relationships:\n" + "\n".join(corr_lines))

    # 4. Outliers as prose
    outliers = analysis.get("outliers", {})
    if outliers and isinstance(outliers, dict):
        outlier_lines = []
        for col, details in list(outliers.items())[:10]:
            if isinstance(details, dict):
                cnt = details.get("count", 0)
                pct = details.get("percentage", 0)
                outlier_lines.append(
                    f"Column '{col}' has {cnt} outlier values ({pct:.1f}% of records), "
                    f"representing unusually high or low data points."
                )
        if outlier_lines:
            sections.append("Outlier and anomaly detection results:\n" + "\n".join(outlier_lines))

    # 5. Data quality and confidence
    confidence = analysis.get("confidence_scores", {})
    if confidence and isinstance(confidence, dict):
        quality_score = confidence.get("dataset_confidence", confidence.get("overall_score", ""))
        grade = confidence.get("dataset_grade", confidence.get("grade", ""))
        if quality_score:
            sections.append(
                f"The overall data quality confidence score is {quality_score}% (grade: {grade}). "
                f"This reflects schema consistency, null rate, and data integrity checks."
            )

    # 6. AI-generated insights
    if insights_text:
        clean_text = (
            insights_text.replace("**", "")
            .replace("<b>", "")
            .replace("</b>", "")
            .replace("<i>", "")
            .replace("</i>", "")
        )
        sections.append(f"AI-generated strategic insights:\n{clean_text}")

    # 7. Data cleaning actions
    if cleaning and isinstance(cleaning, dict):
        cleaning_parts = []
        dup = cleaning.get("duplicate_rows_removed", 0)
        empty = cleaning.get("empty_rows_dropped", 0)
        num_fill = cleaning.get("numeric_nans_filled", 0)
        cat_fill = cleaning.get("categorical_nans_filled", 0)
        if dup:
            cleaning_parts.append(f"removed {dup} duplicate rows")
        if empty:
            cleaning_parts.append(f"dropped {empty} empty rows")
        if num_fill:
            cleaning_parts.append(f"imputed {num_fill} missing numeric values")
        if cat_fill:
            cleaning_parts.append(f"imputed {cat_fill} missing categorical values")
        if cleaning_parts:
            sections.append(f"Data cleaning actions performed: {', '.join(cleaning_parts)}.")
        else:
            sections.append(
                "Data cleaning: No significant cleaning actions were required. The dataset was already in good shape."
            )

    return "\n\n".join(sections)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Inspect File (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

@register_task("app.tasks.inspect_file")
@register_task("inspect_file_task")
async def inspect_file_task(ctx: Optional[dict], task_id: str, file_ref: str, filename: str):
    """
    Phase 1: Ingest Data -> Inspect Quality -> Detect Issues -> Wait for User Confirmation.
    """
    logger.info("Starting inspect_file_task for task_id=%s, filename=%s", task_id, filename)
    try:
        title_task_manager.update_progress(task_id, 10, "Loading file with Polars streaming engine...")

        file_path = storage.get_absolute_path(file_ref)
        if not os.path.exists(file_path):
            raise ParseError(f"Uploaded file not found on disk: {file_ref}")

        # 1. Load via Polars streaming engine
        try:
            df = await asyncio.to_thread(load_dataframe, file_path)
        except Exception as e:
            raise ParseError(f"Load failed: {e}")

        # 2. Inspect data quality
        title_task_manager.update_progress(task_id, 25, "Inspecting data quality & schema...")
        quality_report = await asyncio.to_thread(inspect_dataset, df)

        # 3. Detect issues for Issue Ledger
        title_task_manager.update_progress(task_id, 35, "Detecting data quality issues...")
        issue_ledger = await asyncio.to_thread(detect_issues, df)
        issue_count = len(issue_ledger.issues)
        logger.info("Task %s: Detected %d data issues.", task_id, issue_count)

        # 4. Build partial result and pause for user input
        partial_result = {
            "filename": filename,
            "quality_report": quality_report,
            "issue_ledger": issue_ledger.to_dict(),
            "_file_ref": file_ref,
            "stage": "INSPECTION",
        }

        title_task_manager.update_status(task_id, TaskStatus.WAITING_FOR_USER, partial_result)
        title_task_manager.update_progress(task_id, 40, f"Review {issue_count} detected issues")

        del df
        gc.collect()

    except Exception as e:
        logger.error("Inspection failed for task %s: %s", task_id, e, exc_info=True)
        title_task_manager.fail_job(task_id, str(e))
        if file_ref:
            try:
                storage.delete(file_ref)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Resume Analysis Pipeline (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

@register_task("app.tasks.resume_analysis")
@register_task("resume_analysis_task")
async def resume_analysis_pipeline_task(
    ctx: Optional[dict],
    task_id: str,
    rules: Dict[str, Any],
    analysis_config_dict: Optional[Dict[str, Any]] = None,
):
    """
    Phase 2: Comprehensive pipeline executing:
      1. Data Cleaning
      2. Statistical Analysis
      3. Parallel Charts & AI Narrative Generation (via asyncio.gather)
      4. Report Compilation (Typst engine)
      5. Background RAG Vector Store Ingestion
    """
    logger.info("Starting resume_analysis_pipeline_task for task_id=%s", task_id)

    # 1. Load Job
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        logger.error("Task %s invalid for resumption: job or job.result missing.", task_id)
        title_task_manager.fail_job(task_id, "Job state invalid for resumption.")
        return

    file_ref = job.result.get("_file_ref") or job.result.get("_temp_path")
    filename = job.result.get("filename", "unknown")

    if not file_ref:
        title_task_manager.fail_job(task_id, "Source file reference missing.")
        return

    try:
        # Step 1: Clean Data
        title_task_manager.update_progress(task_id, 45, "Applying cleaning rules & transformations...")
        file_path = storage.get_absolute_path(file_ref)
        if not os.path.exists(file_path):
            raise ValueError(f"Source file missing from disk: {file_ref}")

        df = await asyncio.to_thread(load_dataframe, file_path)

        def _clean_worker():
            return clean_data(df, rules, None, filename)

        cleaned_df, cleaning_report, transformation_dag = await asyncio.to_thread(_clean_worker)

        # Persist intermediate cleaned data (Parquet)
        buffer = BytesIO()
        cleaned_df.write_parquet(buffer)
        buffer.seek(0)
        cleaned_file_ref = storage.save_upload(buffer, f"cleaned_{task_id}.parquet")

        # Step 2: Statistical Analysis
        title_task_manager.update_progress(task_id, 60, "Running deep statistical analysis...")
        top_cats = rules.get("top_categories", 10)

        analysis_config = None
        if analysis_config_dict:
            try:
                analysis_config = AnalysisConfig(**analysis_config_dict)
            except Exception as conf_err:
                logger.warning("Could not parse analysis_config (%s). Using defaults.", conf_err)
                analysis_config = AnalysisConfig.default()
        else:
            analysis_config = AnalysisConfig.default()

        def _analysis_worker():
            analysis_res = analyze_dataset(cleaned_df, top_cats, analysis_config)
            ds_info = get_dataset_info(cleaned_df)
            iss_ledger = detect_issues(cleaned_df)
            sch_profile = build_schema_profile(cleaned_df)
            prev_job = title_task_manager.find_previous_completed_job(task_id, filename)
            hist_comp = build_historical_comparison(
                prev_job.id if prev_job else None,
                prev_job.result if prev_job else None,
                sch_profile,
            )
            # Compare with original dataset
            orig_df = load_dataframe(file_path)
            comp_report = comparison_service.compare(orig_df, cleaned_df)
            del orig_df
            return analysis_res, ds_info, iss_ledger, sch_profile, hist_comp, comp_report

        (
            analysis_result,
            dataset_info,
            issue_ledger,
            schema_profile,
            historical_comparison,
            comparison_report,
        ) = await asyncio.to_thread(_analysis_worker)

        config_snapshot = analysis_config.snapshot()

        # Step 3: Parallel Execution of Visualizations & LLM Insights
        title_task_manager.update_progress(task_id, 75, "Generating charts and AI insights concurrently...")

        async def _run_charts():
            try:
                charts, _ = await asyncio.to_thread(generate_charts, cleaned_df)
                return charts
            except Exception as c_err:
                logger.error("Charts generation error: %s", c_err, exc_info=True)
                return {}

        async def _run_insights():
            try:
                insights = await asyncio.to_thread(generate_insights_sync, analysis_result)
                return insights.to_dict() if hasattr(insights, "to_dict") else dict(insights)
            except Exception as i_err:
                logger.error("Insights generation error: %s", i_err, exc_info=True)
                return {}

        charts_res, insights_res = await asyncio.gather(_run_charts(), _run_insights())

        # Step 4: Compile Report (Typst Native Vector Engine)
        title_task_manager.update_progress(task_id, 90, "Compiling PDF Report via Typst engine...")

        analysis_data = analysis_result.copy()
        if insights_res:
            analysis_data["insights"] = insights_res
        if cleaning_report:
            analysis_data["cleaning_report"] = cleaning_report.to_dict()
        if comparison_report:
            analysis_data["comparison_report"] = comparison_report.to_dict()
        if issue_ledger:
            analysis_data["issue_ledger"] = issue_ledger.to_dict()

        title_task_manager.update_progress(task_id, 95, "Rendering PDF...")

        def _compile_pdf():
            pdf_buf, _ = generate_pdf_report(
                analysis_data,
                charts_res,
                filename,
            )
            out_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"{task_id}_{filename}.pdf"
            out_path = os.path.join(out_dir, out_name)
            with open(out_path, "wb") as f_out:
                f_out.write(pdf_buf.getbuffer())
            return out_path

        pdf_path = await asyncio.to_thread(_compile_pdf)

        # Assemble final result
        final_result = {
            "filename": filename,
            "info": dataset_info,
            "cleaning_report": cleaning_report.to_dict() if hasattr(cleaning_report, "to_dict") else cleaning_report,
            "analysis": analysis_result,
            "charts": charts_res,
            "insights": insights_res,
            "transformation_dag": (
                transformation_dag.to_dict() if hasattr(transformation_dag, "to_dict") else transformation_dag
            ),
            "comparison_report": (
                comparison_report.to_dict() if hasattr(comparison_report, "to_dict") else comparison_report
            ),
            "issue_ledger": issue_ledger.to_dict() if hasattr(issue_ledger, "to_dict") else issue_ledger,
            "analysis_config": config_snapshot,
            "schema_profile": schema_profile,
            "historical_comparison": historical_comparison,
            "cleaned_file_ref": cleaned_file_ref,
            "report_path": pdf_path,
        }

        # Step 5: Trigger RAG Ingestion asynchronously in background
        try:
            raw_insights_text = ""
            if isinstance(insights_res, dict):
                raw_insights_text = insights_res.get("insights_text", "")
            rag_narrative = build_rag_narrative(
                filename,
                analysis_result,
                final_result["cleaning_report"],
                raw_insights_text,
                dataset_info,
            )
            # Schedule non-blocking RAG ingestion
            asyncio.create_task(rag_ingest_task(ctx, task_id, rag_narrative))
        except Exception as rag_err:
            logger.warning("Could not initiate RAG narrative generation: %s", rag_err)

        # Mark job complete
        title_task_manager.complete_job(task_id, final_result, report_path=pdf_path)
        logger.info("Successfully completed analysis pipeline for task %s (PDF: %s)", task_id, pdf_path)

        # Cleanup source file
        try:
            storage.delete(file_ref)
        except Exception:
            pass

        del df, cleaned_df, analysis_data
        gc.collect()
        return final_result

    except Exception as e:
        logger.error("Analysis pipeline failed for task %s: %s", task_id, e, exc_info=True)
        title_task_manager.fail_job(task_id, f"Analysis pipeline failed: {str(e)}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Standalone / Re-generation PDF Task
# ─────────────────────────────────────────────────────────────────────────────

@register_task("app.tasks.generate_pdf")
@register_task("generate_pdf_task")
async def generate_pdf_task(ctx: Optional[dict], task_id: str):
    """
    Generate PDF for download (Standalone / Re-generation).
    """
    logger.info("Starting generate_pdf_task for task_id=%s", task_id)
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        logger.error("Job %s not ready for PDF generation.", task_id)
        return

    try:
        result = job.result
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)

        filename = result.get("filename", "unknown")
        pdf_name = f"{task_id}_{filename}.pdf"
        pdf_path = os.path.join(output_dir, pdf_name)

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

        def _render():
            pdf_buffer, _ = generate_pdf_report(
                analysis_data,
                result.get("charts", {}),
                filename,
            )
            with open(pdf_path, "wb") as f:
                f.write(pdf_buffer.getbuffer())

        await asyncio.to_thread(_render)
        title_task_manager.complete_job(task_id, job.result, report_path=pdf_path)
        logger.info("generate_pdf_task successfully finished for %s at %s", task_id, pdf_path)

    except Exception as e:
        logger.error("PDF Gen Task Failed for task %s: %s", task_id, e, exc_info=True)
        title_task_manager.set_report_status(task_id, "failed")


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: RAG Ingest Task
# ─────────────────────────────────────────────────────────────────────────────

@register_task("app.tasks.rag_ingest")
@register_task("rag_ingest_task")
async def rag_ingest_task(ctx: Optional[dict], task_id: str, text: str):
    """
    Ingest text into vector store (RAG) and construct Dataset Knowledge Graph.
    """
    logger.info("Starting rag_ingest_task for task_id=%s (text length: %d chars)", task_id, len(text))
    try:
        await asyncio.to_thread(rag_service.ingest_report_blocking, task_id, text)

        # Build and persist Dataset Knowledge Graph
        def _graph_worker():
            job = title_task_manager.get_job(task_id)
            if job and job.result:
                from app.services.dataset_graph_builder import build_dataset_graph
                base_dir = Path(__file__).resolve().parent.parent
                cache_dir = os.path.join(base_dir, "temp_cache")
                os.makedirs(cache_dir, exist_ok=True)
                graph_path = os.path.join(cache_dir, f"{task_id}_graph.json")
                ledger_issues = job.result.get("ledger_issues", job.result.get("issues", []))
                graph_store = build_dataset_graph(task_id, job.result, ledger_issues)
                graph_store.save_to_file(graph_path)
                logger.info("Dataset Knowledge Graph constructed and saved to disk: %s", graph_path)

        await asyncio.to_thread(_graph_worker)

    except Exception as e:
        logger.error("RAG Ingestion Task failed for task %s: %s", task_id, e, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# ARQ Worker Settings (For Production Deployment: arq app.tasks_arq.WorkerSettings)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from arq.connections import RedisSettings
    from arq.worker import func

    class WorkerSettings:
        functions = [
            func(inspect_file_task, name="app.tasks.inspect_file"),
            func(inspect_file_task, name="inspect_file_task"),
            func(resume_analysis_pipeline_task, name="app.tasks.resume_analysis"),
            func(resume_analysis_pipeline_task, name="resume_analysis_task"),
            func(generate_pdf_task, name="app.tasks.generate_pdf"),
            func(generate_pdf_task, name="generate_pdf_task"),
            func(rag_ingest_task, name="app.tasks.rag_ingest"),
            func(rag_ingest_task, name="rag_ingest_task"),
        ]
        redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
        max_jobs = 8
        job_timeout = 600
        keep_result = 3600

except ImportError:
    WorkerSettings = None  # type: ignore
