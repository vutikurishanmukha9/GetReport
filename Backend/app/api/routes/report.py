"""
Report Routes — PDF generation, status, download, and DAG endpoints.
"""
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging
import os
import re
import html

from app.core.limiter import limiter, REPORT_LIMIT
from app.core.auth import verify_api_key, validate_task_id, check_task_ownership, get_authenticated_owner_id
from app.services.task_manager import title_task_manager, TaskStatus
from app.core.task_dispatcher import dispatch_task

logger = logging.getLogger(__name__)
router = APIRouter()

class AnalysisRulesRequest(BaseModel):
    rules: Dict[str, Any]
    analysis_config: Optional[Dict[str, Any]] = None

class SQLQueryRequest(BaseModel):
    sql: str
    limit: Optional[int] = Field(default=500, ge=1, le=1000)

class GoldenQueryCreateRequest(BaseModel):
    question: str
    sql_query: str
    description: Optional[str] = ""

class SandboxExecRequest(BaseModel):
    code: str = Field(..., max_length=50000, description="Python/Polars code to execute in sandbox")
    timeout_seconds: Optional[int] = Field(default=10, ge=1, le=30, description="Execution timeout cap in seconds")

class DeriveConceptRequest(BaseModel):
    concept_name: str
    formula_or_intent: str
    description: Optional[str] = ""

# ─── Tenant Boundary & Path Guards ──────────────────────────────────────────

ALLOWED_OUTPUT_DIR = os.path.abspath("outputs")

def _validate_report_path(report_path: str) -> str:
    """Ensure report_path is within the allowed outputs directory."""
    real_path = os.path.realpath(report_path)
    try:
        is_allowed = os.path.commonpath([ALLOWED_OUTPUT_DIR, real_path]) == ALLOWED_OUTPUT_DIR
    except ValueError:
        is_allowed = False
    if not is_allowed:
        logger.warning(f"Path traversal attempt blocked: {report_path}")
        raise HTTPException(403, "Access denied")
    return real_path

def _sanitize_download_filename(filename: str) -> str:
    """Sanitize filename for Content-Disposition header."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)


async def _get_authorized_job(request: Optional[Request], task_id: str, require_result: bool = False):
    """
    Validates task_id format, retrieves job, and enforces strict tenant isolation (BOLA / CWE-285).
    """
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job:
        raise HTTPException(404, "Job not found")
    caller_owner_id = get_authenticated_owner_id(request)
    check_task_ownership(job.owner_id, caller_owner_id)
    if require_result and not job.result:
        raise HTTPException(400, "Job analysis result not found or not ready yet")
    return job


# ─── PDF Generation & Download ──────────────────────────────────────────────

@router.post("/jobs/{task_id}/report")
@limiter.limit(REPORT_LIMIT)
async def generate_persistent_report(
    request: Request, task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Triggers PDF generation via Celery/ARQ and returns immediately.
    Frontend should poll /jobs/{task_id}/report/status until ready.
    """
    job = await _get_authorized_job(request, task_id, require_result=True)
    
    try:
        await title_task_manager.set_report_status_async(task_id, "generating")
        await dispatch_task("app.tasks.generate_pdf", task_id)
        return {"message": "Report generation started. Poll /report/status for progress.", "path": None}
        
    except Exception as e:
        await title_task_manager.set_report_status_async(task_id, "failed")
        logger.error(f"Persistent report generation failed: {str(e)}")
        raise HTTPException(500, "Failed to start report generation.")


@router.get("/jobs/{task_id}/report/status")
async def get_report_status(
    task_id: str,
    request: Request = None,
    _auth: None = Depends(verify_api_key),
):
    """
    Check if the PDF report has been generated and is ready for download.
    Returns: { status: 'generating' | 'ready' | 'not_started', path?: string }
    """
    job = await _get_authorized_job(request, task_id)
    
    if job.report_path and os.path.exists(job.report_path):
        return {"status": "ready", "download_url": f"/api/jobs/{task_id}/report"}
    elif job.report_status == "generating":
        return {"status": "generating"}
    elif job.report_status == "failed":
        return {"status": "failed"}
    else:
        return {"status": "not_started"}


@router.get("/jobs/{task_id}/report")
async def download_report(
    task_id: str,
    request: Request = None,
    _auth: None = Depends(verify_api_key),
):
    """
    Downloads the persisted PDF report.
    Returns 202 if still generating, 200 + file if ready.
    """
    job = await _get_authorized_job(request, task_id)
    if not job.report_path:
        raise HTTPException(404, "Report not found. Generate it first.")
    
    # Path traversal guard (VULN-05)
    safe_path = _validate_report_path(job.report_path)
    
    if not os.path.exists(safe_path):
        return JSONResponse(status_code=202, content={"message": "Report is still generating. Try again shortly."})
        
    return FileResponse(
        safe_path, 
        media_type="application/pdf", 
        filename=_sanitize_download_filename(os.path.basename(safe_path))
    )


# ─── Comprehensive PDF (Full Report Generator) ──────────────────────────────

@router.get("/jobs/{task_id}/report/full")
@limiter.limit(REPORT_LIMIT)
async def download_full_report_pdf(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Generate and download the comprehensive PDF report (Tier 1-4).
    Uses the full report_generator (not the simple renderer).
    """
    from app.services.report_generator import generate_pdf_report
    
    job = await _get_authorized_job(request, task_id, require_result=True)
    
    analysis = job.result
    if "analysis" not in analysis:
        raise HTTPException(400, "Invalid job result format")
    
    raw_analysis = analysis.get("analysis", {})
    params = dict(raw_analysis)
    params["cleaning_report"] = analysis.get("cleaning_report")
    params["metadata"] = raw_analysis.get("metadata") or analysis.get("info")
    params["comparison_report"] = analysis.get("comparison_report")
    
    charts = analysis.get("charts", {})
    filename = analysis.get("filename", "report.pdf")
    safe_filename = _sanitize_download_filename(filename)
    
    try:
        # PDF Generator is sync/CPU heavy, keep in threadpool
        pdf_buffer, meta = await run_in_threadpool(
            generate_pdf_report, params, charts, filename
        )
        
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=Report_{safe_filename}.pdf"}
        )
    except Exception as e:
        logger.error(f"PDF Generation failed: {e}", exc_info=True)
        raise HTTPException(500, "Failed to generate PDF report due to an internal server error.")


# ─── Transformation DAG ─────────────────────────────────────────────────────

@router.get("/jobs/{task_id}/dag")
async def get_transformation_dag(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Get the complete transformation DAG for a job.
    """
    job = await _get_authorized_job(request, task_id, require_result=True)
    
    dag_data = job.result.get("transformation_dag")
    if not dag_data:
        raise HTTPException(400, "No transformation DAG available (job may still be processing)")
    
    return dag_data


@router.get("/jobs/{task_id}/dag/export")
@limiter.limit(REPORT_LIMIT)
async def export_transformation_dag(
    request: Request,
    task_id: str, format: str = "json",
    _auth: None = Depends(verify_api_key),
):
    """
    Export the transformation DAG as an audit log.
    """
    from app.services.transformation_dag import from_dict
    
    job = await _get_authorized_job(request, task_id, require_result=True)
    
    dag_data = job.result.get("transformation_dag")
    if not dag_data:
        raise HTTPException(400, "No transformation DAG available")
    
    dag = from_dict(dag_data)
    
    if format.lower() == "csv":
        csv_content = dag.to_audit_csv()
        return PlainTextResponse(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=audit_log_{task_id}.csv"}
        )
    else:
        return dag.to_audit_log()


@router.get("/jobs/{task_id}/dag/summary")
async def get_dag_summary(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """Get a high-level summary of the transformation DAG."""
    job = await _get_authorized_job(request, task_id, require_result=True)
    
    dag_data = job.result.get("transformation_dag")
    if not dag_data:
        raise HTTPException(400, "No transformation DAG available")
    
    return dag_data.get("summary", {})


@router.get("/jobs/{task_id}/dag/{node_id}")
async def get_dag_node(
    request: Request,
    task_id: str, node_id: str,
    _auth: None = Depends(verify_api_key),
):
    """Get details of a single transformation node."""
    job = await _get_authorized_job(request, task_id, require_result=True)
    
    dag_data = job.result.get("transformation_dag")
    if not dag_data:
        raise HTTPException(400, "No transformation DAG available")
    
    nodes = dag_data.get("nodes", {})
    node = nodes.get(node_id)
    
    if not node:
        raise HTTPException(404, f"Node not found in DAG")
    
    return node


# ─── Comparison ──────────────────────────────────────────────────────────────

@router.get("/jobs/{task_id}/comparison")
async def get_comparison_report(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Get the Data Quality Comparison Report (Before vs After).
    """
    job = await _get_authorized_job(request, task_id, require_result=True)
    
    report = job.result.get("comparison_report")
    if not report:
        raise HTTPException(400, "Comparison report not available")
    
    return report


@router.get("/jobs/{task_id}/history")
async def get_historical_comparison(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """Return additive schema-drift and trend data for this dataset run."""
    job = await _get_authorized_job(request, task_id, require_result=True)
    return job.result.get("historical_comparison", {"schema_drift": {"baseline_available": False, "status": "no_baseline"}})

class GenerateReportRequest(BaseModel):
    filename: str
    analysis: Dict[str, Any]
    charts: Dict[str, Any]

@router.post("/generate-report")
@limiter.limit(REPORT_LIMIT)
async def generate_report_direct(
    request: Request,
    body: GenerateReportRequest,
    _auth: None = Depends(verify_api_key),
):
    """
    Exposes a direct on-the-fly PDF generation endpoint (typically for tests/isolated verification).
    """
    from app.services.report_generator import generate_pdf_report
    try:
        pdf_buffer, meta = await run_in_threadpool(
            generate_pdf_report, body.analysis, body.charts, body.filename
        )
        safe_filename = _sanitize_download_filename(body.filename)
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={safe_filename}"}
        )
    except Exception as e:
        logger.error(f"Direct PDF Generation failed: {str(e)}", exc_info=True)
        raise HTTPException(500, "Failed to generate PDF report due to an internal server error.")


# ─── Multi-Format Export Route (CSV / Parquet / HTML) ─────────────────────────

@router.get("/jobs/{task_id}/export/{export_format}")
@limiter.limit(REPORT_LIMIT)
async def export_cleaned_data_or_report(
    request: Request,
    task_id: str,
    export_format: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Exports the audited/cleaned dataset or HTML report bundle.
    Supported formats: 'csv', 'parquet', 'html'.
    """
    job = await _get_authorized_job(request, task_id, require_result=True)

    clean_fmt = export_format.lower()
    raw_base = _sanitize_download_filename(job.filename or "dataset")
    filename_base = os.path.splitext(raw_base)[0]
    
    from app.services.storage import get_storage_provider
    from app.services.data_processing import load_dataframe
    import io

    storage = get_storage_provider()
    
    if clean_fmt in ("csv", "parquet", "excel", "xlsx"):
        try:
            cleaned_file_ref = job.result.get("cleaned_file_ref")
            if not cleaned_file_ref:
                raise HTTPException(404, "Cleaned dataset is not available for export")
            file_path = storage.get_absolute_path(cleaned_file_ref)
            if not os.path.exists(file_path):
                raise HTTPException(404, "Cleaned dataset source file not found")

            df = load_dataframe(file_path)

            from app.core.factories.exporters import DataExporterFactory

            format_key = "excel" if clean_fmt in ("excel", "xlsx") else clean_fmt
            exporter = DataExporterFactory.create_exporter(format_key)
            temp_path, media_type, ext = exporter.export(df, filename_base)
            try:
                content_bytes = temp_path.read_bytes()
            finally:
                if temp_path.exists():
                    temp_path.unlink()

            return Response(
                content=content_bytes,
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename=Cleaned_{filename_base}{ext}"}
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Data export failed: {e}", exc_info=True)
            raise HTTPException(500, "Failed to export cleaned dataset due to an internal server error.")

    elif clean_fmt == "html":
        info = job.result.get("info", {})
        summary = job.result.get("analysis", {}).get("executive_summary", "No executive summary available.")
        score = job.result.get("ml_readiness", {}).get("overall_score", 100)

        # Security: Escape all user-controlled data to prevent Stored XSS (SEC-01)
        safe_filename_display = html.escape(str(filename_base))
        safe_summary = html.escape(str(summary)).replace('\n', '<br/>')
        safe_rows = html.escape(f"{info.get('rows', 0):,}")
        safe_cols = html.escape(str(info.get("columns", 0)))
        safe_health = html.escape(str(info.get("data_health_score", 100)))
        safe_score = html.escape(str(score))
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GetReport: Executive Data Briefing: {safe_filename_display}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #FAF6F0; color: #2C2C2C; padding: 40px; margin: 0; }}
        .card {{ background: #FFFFFF; border: 1px solid #E5DCC3; border-radius: 16px; padding: 32px; max-width: 900px; margin: 0 auto; box-shadow: 0 10px 30px rgba(114,47,55,0.08); }}
        .header {{ border-bottom: 2px solid #722F37; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center; }}
        h1 {{ color: #722F37; margin: 0; font-size: 26px; }}
        .badge {{ background: #722F37; color: #FFFFFF; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 14px; }}
        .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 30px; }}
        .stat {{ background: #FAF6F0; padding: 16px; border-radius: 12px; text-align: center; border: 1px solid #E5DCC3; }}
        .stat-val {{ font-size: 22px; font-weight: bold; color: #722F37; }}
        .stat-lbl {{ font-size: 12px; color: #666; text-transform: uppercase; margin-top: 4px; }}
        .summary {{ background: #FFF9F5; border-left: 4px solid #722F37; padding: 20px; border-radius: 8px; line-height: 1.6; margin-bottom: 30px; }}
        .footer {{ text-align: center; font-size: 12px; color: #888; margin-top: 40px; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div>
                <h1>Executive Data Analysis Briefing</h1>
                <p style="margin: 4px 0 0 0; color: #666;">Dataset: {safe_filename_display}</p>
            </div>
            <span class="badge">ML Readiness: {safe_score}/100</span>
        </div>
        <div class="grid">
            <div class="stat"><div class="stat-val">{safe_rows}</div><div class="stat-lbl">Total Rows</div></div>
            <div class="stat"><div class="stat-val">{safe_cols}</div><div class="stat-lbl">Total Columns</div></div>
            <div class="stat"><div class="stat-val">{safe_health}%</div><div class="stat-lbl">Data Health</div></div>
        </div>
        <h2>Executive Summary & Insights</h2>
        <div class="summary">
            {safe_summary}
        </div>
        <div class="footer">
            Generated by GetReport Platform • Privacy Guaranteed & Isolated Processing
        </div>
    </div>
</body>
</html>"""
        return Response(
            content=html_content,
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename=Report_{filename_base}.html"}
        )
    else:
        raise HTTPException(400, "Invalid export format. Supported: 'csv', 'parquet', 'excel', 'html'")


# ─── In-Process DuckDB Analytical Query Endpoint ────────────────────────────

@router.post("/jobs/{task_id}/query-sql")
@limiter.limit(REPORT_LIMIT)
async def query_dataset_sql(
    request: Request,
    task_id: str,
    body: SQLQueryRequest,
    _auth: None = Depends(verify_api_key),
):
    """
    Executes an analytical read-only SQL query against the cleaned dataset
    using in-process DuckDB.
    """
    job = await _get_authorized_job(request, task_id, require_result=True)

    from app.services.storage import get_storage_provider
    from app.services.duckdb_engine import DuckDBAnalyticalSession, DuckDBSecurityError
    from app.services.data_processing import load_dataframe

    storage = get_storage_provider()
    cleaned_file_ref = job.result.get("cleaned_file_ref") or job.result.get("_file_ref")
    if not cleaned_file_ref:
        raise HTTPException(404, "Dataset artifact not found for this job")

    file_path = storage.get_absolute_path(cleaned_file_ref)
    if not os.path.exists(file_path):
        raise HTTPException(404, "Dataset file does not exist on storage")

    session = DuckDBAnalyticalSession()
    try:
        if file_path.endswith(".parquet"):
            session.register_parquet("dataset", file_path)
        else:
            df = load_dataframe(file_path)
            session.register_polars("dataset", df)

        import time
        t_start = time.perf_counter()
        query_limit = min(body.limit or 500, 1000)
        records = session.execute_read_query(body.sql, max_rows=query_limit)
        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
        columns = list(records[0].keys()) if records else []
        capped_records = records[:query_limit]

        return {
            "task_id": task_id,
            "sql": body.sql,
            "columns": columns,
            "records": capped_records,
            "total_returned": len(records),
            "capped": len(records) > len(capped_records),
            "execution_ms": duration_ms,
        }
    except DuckDBSecurityError as se:
        raise HTTPException(403, str(se))
    except Exception as e:
        logger.error(f"SQL execution error for task {task_id}: {e}", exc_info=True)
        raw_err = str(e).strip()
        sanitized_err = re.sub(r"[A-Za-z]:\\[^ \n\r\t:]+", "<path>", raw_err)
        sanitized_err = re.sub(r"/(?:Users|home|tmp|var|opt)/[^ \n\r\t:]+", "<path>", sanitized_err)
        first_line = sanitized_err.split("\n")[0].strip()
        detail_msg = first_line if first_line else "Query execution failed. Please verify query syntax and column references."
        raise HTTPException(400, f"DuckDB Query Error: {detail_msg}")
    finally:
        session.close()


@router.get("/jobs/{task_id}/sql-schema")
@limiter.limit(REPORT_LIMIT)
async def get_job_sql_schema(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Returns DuckDB schema introspection for the dataset registered in the job.
    Includes tables, column names, column data types, nullability, and sample preview values.
    """
    job = await _get_authorized_job(request, task_id, require_result=True)

    from app.services.storage import get_storage_provider
    from app.services.duckdb_engine import DuckDBAnalyticalSession
    from app.services.data_processing import load_dataframe

    cleaned_file_ref = job.result.get("cleaned_file_ref") or job.result.get("_file_ref")
    if not cleaned_file_ref:
        raise HTTPException(404, "Dataset artifact not found for this job")

    storage = get_storage_provider()
    file_path = storage.get_absolute_path(cleaned_file_ref)
    if not os.path.exists(file_path):
        raise HTTPException(404, "Dataset file does not exist on storage")

    session = DuckDBAnalyticalSession()
    try:
        if file_path.endswith(".parquet"):
            session.register_parquet("dataset", file_path)
        else:
            df = load_dataframe(file_path)
            session.register_polars("dataset", df)

        schema_meta = session.get_schema_metadata()
        return {
            "task_id": task_id,
            "success": True,
            **schema_meta,
        }
    except Exception as e:
        logger.error(f"Error fetching SQL schema for task {task_id}: {e}", exc_info=True)
        raise HTTPException(500, "Failed to inspect SQL schema due to an internal server error.")
    finally:
        session.close()


# ─── Great Expectations Suite Export Endpoint ──────────────────────────────

@router.get("/jobs/{task_id}/export-gx")
@limiter.limit(REPORT_LIMIT)
async def export_great_expectations_suite(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Generates and exports an industry-standard Great Expectations Suite JSON
    based on the cleaned dataset, quality scoring, and approved issue ledger actions.
    """
    job = await _get_authorized_job(request, task_id, require_result=True)

    from app.services.storage import get_storage_provider
    from app.services.data_processing import load_dataframe
    from app.services.gx_exporter import GreatExpectationsSuiteExporter

    storage = get_storage_provider()
    cleaned_file_ref = job.result.get("cleaned_file_ref") or job.result.get("_file_ref")
    if not cleaned_file_ref:
        raise HTTPException(404, "Dataset artifact not found for this job")

    file_path = storage.get_absolute_path(cleaned_file_ref)
    if not os.path.exists(file_path):
        raise HTTPException(404, "Dataset file does not exist on storage")

    df = load_dataframe(file_path)
    filename_base = _sanitize_download_filename(job.filename or "dataset")
    suite_name = f"{filename_base}_suite"

    quality_info = job.result.get("info")
    issue_ledger_data = job.result.get("issue_ledger", {})
    approved_issues = [
        issue
        for issue in issue_ledger_data.get("issues", [])
        if issue.get("status") == "approved"
    ]

    exporter = GreatExpectationsSuiteExporter(suite_name=suite_name)
    suite = exporter.generate_suite_from_polars(
        df, quality_report=quality_info, approved_issues=approved_issues
    )
    suite_json = exporter.export_json(suite)

    return Response(
        content=suite_json,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename={filename_base}_expectation_suite.json"
        },
    )


# ─── Golden Query Repository Endpoints (Dataherald-inspired) ───────────────

@router.post("/jobs/{task_id}/golden-queries")
@limiter.limit(REPORT_LIMIT)
async def create_golden_query(
    request: Request,
    task_id: str,
    body: GoldenQueryCreateRequest,
    _auth: None = Depends(verify_api_key),
):
    """
    Saves a verified analytical SQL query as a Golden KPI for zero-latency retrieval.
    """
    job = await _get_authorized_job(request, task_id)

    from app.services.golden_query_service import GoldenQueryService
    try:
        query = await GoldenQueryService.save_golden_query_async(
            task_id=task_id,
            question=body.question,
            sql_query=body.sql_query,
            description=body.description or "",
        )
        return query.to_dict()
    except ValueError as ve:
        raise HTTPException(400, str(ve))
    except Exception as e:
        logger.error(f"Failed to save golden query for task {task_id}: {e}", exc_info=True)
        raise HTTPException(500, "Could not save golden query due to an internal server error.")


@router.get("/jobs/{task_id}/golden-queries")
@limiter.limit(REPORT_LIMIT)
async def list_golden_queries(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Lists all verified Golden KPI queries saved for a job dataset.
    """
    job = await _get_authorized_job(request, task_id)

    from app.services.golden_query_service import GoldenQueryService
    queries = await GoldenQueryService.get_golden_queries_async(task_id)
    return [q.to_dict() for q in queries]


@router.delete("/jobs/{task_id}/golden-queries/{query_id}")
@limiter.limit(REPORT_LIMIT)
async def delete_golden_query(
    request: Request,
    task_id: str,
    query_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Deletes a golden query by ID.
    """
    job = await _get_authorized_job(request, task_id)

    from app.services.golden_query_service import GoldenQueryService
    await GoldenQueryService.delete_golden_query_async(task_id=task_id, query_id=query_id)
    return {"status": "success", "deleted": True, "query_id": query_id}


# ─── Conversational Sandboxing & Virtual Concepts Endpoints (Phase 3) ───────

@router.post("/jobs/{task_id}/sandbox-exec")
@limiter.limit(REPORT_LIMIT)
async def execute_sandboxed_python(
    request: Request,
    task_id: str,
    body: SandboxExecRequest,
    _auth: None = Depends(verify_api_key),
):
    """
    Executes Python/Polars analysis code in a secure AST-sandboxed runtime.
    Captures stdout/stderr, generated plots (base64 PNG), and execution metrics.
    """
    job = await _get_authorized_job(request, task_id, require_result=True)

    cleaned_file_ref = job.result.get("cleaned_file_ref") or job.result.get("_file_ref")
    if not cleaned_file_ref:
        raise HTTPException(404, "Dataset artifact not found for this job")

    from app.services.storage import get_storage_provider
    from app.services.data_processing import load_dataframe
    from app.services.sandboxed_analyst import SandboxedAnalystAgent

    storage = get_storage_provider()
    file_path = storage.get_absolute_path(cleaned_file_ref)
    if not os.path.exists(file_path):
        raise HTTPException(404, "Dataset file does not exist on storage")

    df = await run_in_threadpool(load_dataframe, file_path)
    agent = SandboxedAnalystAgent(timeout_seconds=body.timeout_seconds or 10)
    result = await run_in_threadpool(agent.execute_polars, body.code, df)

    return result.to_dict()


@router.post("/jobs/{task_id}/concepts/derive")
@limiter.limit(REPORT_LIMIT)
async def derive_virtual_concept(
    request: Request,
    task_id: str,
    body: DeriveConceptRequest,
    _auth: None = Depends(verify_api_key),
):
    """
    Synthesizes and derives a virtual analytical metric/concept into the dataset,
    audited through the TransformationDAG, persisting the augmented dataset.
    """
    job = await _get_authorized_job(request, task_id, require_result=True)

    cleaned_file_ref = job.result.get("cleaned_file_ref") or job.result.get("_file_ref")
    if not cleaned_file_ref:
        raise HTTPException(404, "Dataset artifact not found for this job")

    from app.services.storage import get_storage_provider
    from app.services.data_processing import load_dataframe
    from app.services.transformation_dag import from_dict, create_dag
    from app.services.concept_synthesizer import VirtualConceptSynthesizer, ConceptDeclaration
    import io

    storage = get_storage_provider()
    file_path = storage.get_absolute_path(cleaned_file_ref)
    if not os.path.exists(file_path):
        raise HTTPException(404, "Dataset file does not exist on storage")

    df = await run_in_threadpool(load_dataframe, file_path)

    # Load or initialize DAG
    dag_data = job.result.get("transformation_dag")
    if dag_data:
        try:
            dag = from_dict(dag_data)
        except Exception:
            dag = create_dag(df, dataset_name=job.filename or "dataset")
    else:
        dag = create_dag(df, dataset_name=job.filename or "dataset")

    synthesizer = VirtualConceptSynthesizer()
    declaration = ConceptDeclaration(
        concept_name=body.concept_name,
        formula_or_intent=body.formula_or_intent,
        description=body.description,
    )

    try:
        new_df, node = await run_in_threadpool(
            synthesizer.apply_and_record_concept, df, declaration, dag
        )
    except (ValueError, RuntimeError) as err:
        raise HTTPException(400, str(err))
    except Exception as err:
        logger.error(f"Concept derivation failed for task {task_id}: {err}", exc_info=True)
        raise HTTPException(500, "Concept derivation failed due to an internal server error.")

    # Write augmented DataFrame back to storage
    buffer = io.BytesIO()
    new_df.write_parquet(buffer)
    buffer.seek(0)
    updated_file_ref = storage.save_upload(buffer, f"cleaned_{task_id}.parquet")

    # Update job result
    job.result["cleaned_file_ref"] = updated_file_ref
    job.result["transformation_dag"] = dag.to_dict()

    # Track concept list in job result for convenience
    concepts = job.result.setdefault("derived_concepts", [])
    concepts.append({
        "concept_name": body.concept_name,
        "formula_or_intent": body.formula_or_intent,
        "description": body.description or "",
        "node_id": node.id,
        "timestamp": node.timestamp,
        "duration_ms": node.duration_ms,
        "expression": node.parameters.get("expression"),
    })

    await title_task_manager.update_result_async(task_id, job.result)

    return {
        "status": "success",
        "concept_name": body.concept_name,
        "node": node.to_dict(),
        "total_columns": new_df.width,
        "total_rows": new_df.height,
    }


@router.get("/jobs/{task_id}/concepts")
@limiter.limit(REPORT_LIMIT)
async def list_derived_concepts(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Lists all virtual concepts derived for this job dataset.
    """
    job = await _get_authorized_job(request, task_id, require_result=True)

    concepts = job.result.get("derived_concepts")
    if concepts is not None:
        return concepts

    # Fallback: inspect DAG for concept_derivation nodes
    dag_data = job.result.get("transformation_dag", {})
    nodes = dag_data.get("nodes", {})
    concept_list = []
    for node_id, node in nodes.items():
        if isinstance(node, dict) and node.get("operation") == "concept_derivation":
            params = node.get("parameters", {})
            concept_list.append({
                "concept_name": node.get("target_column"),
                "formula_or_intent": params.get("formula_or_intent", ""),
                "description": params.get("description", ""),
                "node_id": node_id,
                "timestamp": node.get("timestamp"),
                "duration_ms": node.get("duration_ms"),
                "expression": params.get("expression"),
            })
    return concept_list



