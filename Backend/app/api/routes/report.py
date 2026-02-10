"""
Report Routes — PDF generation, status, download, and DAG endpoints.
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import os

from app.core.limiter import limiter, REPORT_LIMIT
from app.services.task_manager import title_task_manager, TaskStatus
from app.tasks import generate_pdf_task

logger = logging.getLogger(__name__)
router = APIRouter()

class AnalysisRulesRequest(BaseModel):
    rules: Dict[str, Any]
    analysis_config: Optional[Dict[str, Any]] = None


# ─── PDF Generation & Download ──────────────────────────────────────────────

@router.post("/jobs/{task_id}/report")
@limiter.limit(REPORT_LIMIT)
async def generate_persistent_report(request: Request, task_id: str):
    """
    Triggers PDF generation via Celery and returns immediately.
    Frontend should poll /jobs/{task_id}/report/status until ready.
    """
    job = title_task_manager.get_job(task_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.result:
        raise HTTPException(400, "Job analysis not ready yet")
    
    try:
        generate_pdf_task.delay(task_id)
        return {"message": "Report generation started. Poll /report/status for progress.", "path": None}
        
    except Exception as e:
        logger.error(f"Persistent report generation failed: {str(e)}")
        raise HTTPException(500, str(e))


@router.get("/jobs/{task_id}/report/status")
async def get_report_status(task_id: str):
    """
    Check if the PDF report has been generated and is ready for download.
    Returns: { status: 'generating' | 'ready' | 'not_started', path?: string }
    """
    job = title_task_manager.get_job(task_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    if job.report_path and os.path.exists(job.report_path):
        return {"status": "ready", "path": job.report_path}
    elif job.report_path:
        return {"status": "generating"}
    else:
        return {"status": "not_started"}


@router.get("/jobs/{task_id}/report")
async def download_report(task_id: str):
    """
    Downloads the persisted PDF report.
    Returns 202 if still generating, 200 + file if ready.
    """
    job = title_task_manager.get_job(task_id)
    if not job or not job.report_path:
        raise HTTPException(404, "Report not found. Generate it first.")
    
    if not os.path.exists(job.report_path):
        return JSONResponse(status_code=202, content={"message": "Report is still generating. Try again shortly."})
        
    return FileResponse(
        job.report_path, 
        media_type="application/pdf", 
        filename=os.path.basename(job.report_path)
    )


# ─── Comprehensive PDF (Full Report Generator) ──────────────────────────────

@router.get("/jobs/{task_id}/report/full")
async def download_full_report_pdf(task_id: str):
    """
    Generate and download the comprehensive PDF report (Tier 1-4).
    Uses the full report_generator (not the simple renderer).
    """
    from app.services.report_generator import generate_pdf_report
    
    job = title_task_manager.get_job(task_id)
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
    
    try:
        pdf_buffer, meta = await run_in_threadpool(
            generate_pdf_report, params, charts, filename
        )
        
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=Report_{filename}.pdf"}
        )
    except Exception as e:
        logger.error(f"PDF Generation failed: {e}")
        raise HTTPException(500, f"Failed to generate PDF: {str(e)}")


# ─── Transformation DAG ─────────────────────────────────────────────────────

@router.get("/jobs/{task_id}/dag")
async def get_transformation_dag(task_id: str):
    """
    Get the complete transformation DAG for a job.
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
    """
    from app.services.transformation_dag import from_dict
    
    job = title_task_manager.get_job(task_id)
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
async def get_dag_summary(task_id: str):
    """Get a high-level summary of the transformation DAG."""
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    dag_data = job.result.get("transformation_dag")
    if not dag_data:
        raise HTTPException(400, "No transformation DAG available")
    
    return dag_data.get("summary", {})


@router.get("/jobs/{task_id}/dag/{node_id}")
async def get_dag_node(task_id: str, node_id: str):
    """Get details of a single transformation node."""
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


# ─── Comparison ──────────────────────────────────────────────────────────────

@router.get("/jobs/{task_id}/comparison")
async def get_comparison_report(task_id: str):
    """
    Get the Data Quality Comparison Report (Before vs After).
    """
    job = title_task_manager.get_job(task_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found")
    
    report = job.result.get("comparison_report")
    if not report:
        raise HTTPException(400, "Comparison report not available")
    
    return report
