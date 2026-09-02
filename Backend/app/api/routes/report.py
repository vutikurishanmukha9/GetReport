"""
Report Routes — PDF generation, status, download, and DAG endpoints.
"""
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import os
import re
import html

from app.core.limiter import limiter, REPORT_LIMIT
from app.core.auth import verify_api_key, validate_task_id
from app.services.task_manager import title_task_manager, TaskStatus
from app.tasks import generate_pdf_task

logger = logging.getLogger(__name__)
router = APIRouter()

class AnalysisRulesRequest(BaseModel):
    rules: Dict[str, Any]
    analysis_config: Optional[Dict[str, Any]] = None

# ─── Path Traversal Guard (VULN-05) ─────────────────────────────────────────

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


# ─── PDF Generation & Download ──────────────────────────────────────────────

@router.post("/jobs/{task_id}/report")
@limiter.limit(REPORT_LIMIT)
async def generate_persistent_report(
    request: Request, task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Triggers PDF generation via Celery and returns immediately.
    Frontend should poll /jobs/{task_id}/report/status until ready.
    """
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.result:
        raise HTTPException(400, "Job analysis not ready yet")
    
    try:
        await title_task_manager.set_report_status_async(task_id, "generating")
        generate_pdf_task.delay(task_id)
        return {"message": "Report generation started. Poll /report/status for progress.", "path": None}
        
    except Exception as e:
        await title_task_manager.set_report_status_async(task_id, "failed")
        logger.error(f"Persistent report generation failed: {str(e)}")
        raise HTTPException(500, "Failed to start report generation.")


@router.get("/jobs/{task_id}/report/status")
async def get_report_status(
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Check if the PDF report has been generated and is ready for download.
    Returns: { status: 'generating' | 'ready' | 'not_started', path?: string }
    """
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
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
    _auth: None = Depends(verify_api_key),
):
    """
    Downloads the persisted PDF report.
    Returns 202 if still generating, 200 + file if ready.
    """
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job or not job.report_path:
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
    
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found or not completed")
    
    analysis = job.result
    if "analysis" not in analysis:
        raise HTTPException(400, "Invalid job result format")
    
    params = {}
    params.update(analysis.get("analysis", {}))
    params["cleaning_report"] = analysis.get("cleaning_report")
    params["metadata"] = analysis.get("info")
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
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Get the complete transformation DAG for a job.
    """
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
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
    
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
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
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """Get a high-level summary of the transformation DAG."""
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    dag_data = job.result.get("transformation_dag")
    if not dag_data:
        raise HTTPException(400, "No transformation DAG available")
    
    return dag_data.get("summary", {})


@router.get("/jobs/{task_id}/dag/{node_id}")
async def get_dag_node(
    task_id: str, node_id: str,
    _auth: None = Depends(verify_api_key),
):
    """Get details of a single transformation node."""
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
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
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Get the Data Quality Comparison Report (Before vs After).
    """
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    report = job.result.get("comparison_report")
    if not report:
        raise HTTPException(400, "Comparison report not available")
    
    return report


@router.get("/jobs/{task_id}/history")
async def get_historical_comparison(
    task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """Return additive schema-drift and trend data for this dataset run."""
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found or analysis not complete")
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
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={body.filename}"}
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
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found or analysis not complete")

    clean_fmt = export_format.lower()
    filename_base = _sanitize_download_filename(job.filename or "dataset")
    
    from app.services.storage import get_storage_provider
    from app.services.data_processing import load_dataframe
    import io

    storage = get_storage_provider()
    
    if clean_fmt in ("csv", "parquet"):
        try:
            cleaned_file_ref = job.result.get("cleaned_file_ref")
            if not cleaned_file_ref:
                raise HTTPException(404, "Cleaned dataset is not available for export")
            file_path = storage.get_absolute_path(cleaned_file_ref)
            if not os.path.exists(file_path):
                raise HTTPException(404, "Cleaned dataset source file not found")

            df = load_dataframe(file_path)
            
            if clean_fmt == "csv":
                buffer = io.BytesIO()
                df.write_csv(buffer)
                buffer.seek(0)
                return Response(
                    content=buffer.getvalue(),
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=Cleaned_{filename_base}.csv"}
                )
            else: # parquet
                buffer = io.BytesIO()
                df.write_parquet(buffer)
                buffer.seek(0)
                return Response(
                    content=buffer.getvalue(),
                    media_type="application/octet-stream",
                    headers={"Content-Disposition": f"attachment; filename=Cleaned_{filename_base}.parquet"}
                )
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
    <title>GetReport — Executive Data Briefing: {safe_filename_display}</title>
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
            Generated by GetReport Platform — Privacy Guaranteed & Isolated Processing
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
        raise HTTPException(400, "Invalid export format. Supported: 'csv', 'parquet', 'html'")

