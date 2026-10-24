"""
Knowledge Graph & Lineage API Endpoints for GetReport.
Exposes Graphify-powered community clusters, God Nodes, causal paths,
interactive Vis.js HTML, and blast radius impact analysis.
"""
from __future__ import annotations
import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, Depends
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel

from app.core.auth import verify_api_key, validate_task_id, check_task_ownership, get_authenticated_owner_id
from app.services.task_manager import title_task_manager
from app.core.graph_store import DatasetGraphStore
from app.services.dataset_graph_builder import build_dataset_graph
from app.services.graphify.export_bridge import export_graph_to_format

logger = logging.getLogger(__name__)

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


def _get_cache_dir() -> str:
    cache_dir = os.path.join(BASE_DIR, "temp_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


async def _resolve_graph_store(task_id: str, request: Request) -> DatasetGraphStore:
    """Helper to validate ownership, check cache, or build the graph dynamically."""
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")

    caller_owner_id = get_authenticated_owner_id(request)
    check_task_ownership(job.owner_id, caller_owner_id)

    cache_dir = _get_cache_dir()
    graph_path = os.path.join(cache_dir, f"{task_id}_graph.json")

    # 1. Try loading from cache
    store = DatasetGraphStore.load_from_file(task_id, graph_path)
    if store and store.graph and store.graph.number_of_nodes() > 0:
        return store

    # 2. Build dynamically from job result
    if job.result and isinstance(job.result, dict):
        try:
            ledger_issues = job.result.get("ledger_issues", job.result.get("issues", []))
            store = build_dataset_graph(task_id, job.result, ledger_issues)
            store.save_to_file(graph_path)
            # Generate HTML view
            html_path = os.path.join(cache_dir, f"{task_id}_graph.html")
            store.generate_html(html_path)
            return store
        except Exception as e:
            logger.error("Failed to build dataset knowledge graph: %s", e)
            raise HTTPException(status_code=500, detail=f"Failed to generate knowledge graph: {str(e)}")

    raise HTTPException(status_code=404, detail="No dataset analysis available to construct knowledge graph.")


@router.get("/jobs/{task_id}/graph")
async def get_dataset_knowledge_graph(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Returns full Graphify-clustered knowledge graph payload:
    - Nodes with community cluster IDs, hub labels, degrees, and profiling attributes.
    - Edges with relation types, confidence tags, and weights.
    - God Nodes (top architectural hubs) and Surprising Connections.
    - Community summaries with cohesion scores.
    """
    store = await _resolve_graph_store(task_id, request)

    # Ensure clustering has run
    analysis = store.get_analysis()
    if not analysis or "total_communities" not in analysis:
        analysis = store.cluster()

    nodes_data = []
    for n, d in store.graph.nodes(data=True):
        label = d.get("name") or d.get("column_name") or d.get("label") or str(n)
        nodes_data.append({
            "id": str(n),
            "label": label,
            "entity_type": d.get("entity_type", "Entity"),
            "community": d.get("community", 0),
            "community_label": d.get("community_label", "General"),
            "degree": d.get("degree", store.graph.degree(n)),
            "null_percentage": d.get("null_percentage"),
            "severity": d.get("severity"),
            "benford_status": d.get("benford_status"),
            "attributes": {k: v for k, v in d.items() if k not in ("name", "label", "entity_type", "community", "community_label", "degree")}
        })

    edges_data = []
    for u, v, k, d in store.graph.edges(keys=True, data=True):
        edges_data.append({
            "source": str(u),
            "target": str(v),
            "key": str(k),
            "relation": d.get("relation_type", d.get("relation", "CONNECTED")),
            "confidence": d.get("confidence", "EXTRACTED"),
            "weight": float(d.get("weight", 1.0)),
            "description": d.get("description", "")
        })

    return {
        "task_id": task_id,
        "node_count": len(nodes_data),
        "edge_count": len(edges_data),
        "nodes": nodes_data,
        "edges": edges_data,
        "communities": store.get_communities(),
        "community_labels": store.get_community_labels(),
        "analysis": analysis,
        "interactive_html_url": f"/api/jobs/{task_id}/graph/html"
    }


@router.get("/jobs/{task_id}/graph/html", response_class=HTMLResponse)
async def get_graph_interactive_html(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key)
):
    """
    Renders or serves the self-contained Vis.js interactive graph report.
    Can be viewed in browser or downloaded for offline analysis.
    """
    store = await _resolve_graph_store(task_id, request)
    cache_dir = _get_cache_dir()
    html_path = os.path.join(cache_dir, f"{task_id}_graph.html")

    if not os.path.exists(html_path):
        store.generate_html(html_path)

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        logger.error("Error reading graph HTML file: %s", e)
        raise HTTPException(status_code=500, detail="Could not read interactive HTML graph.")


@router.get("/jobs/{task_id}/graph/path")
async def get_graph_path(
    request: Request,
    task_id: str,
    source_id: str = Query(..., description="Source entity ID, e.g. col:Amount"),
    target_id: str = Query(..., description="Target entity ID, e.g. dataset:transactions.csv"),
    _auth: None = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Finds the shortest and causal path between two entities in the knowledge graph."""
    store = await _resolve_graph_store(task_id, request)
    return store.find_path(source_id, target_id)


@router.get("/jobs/{task_id}/graph/blast-radius")
async def get_node_blast_radius(
    request: Request,
    task_id: str,
    node_id: str = Query(..., description="Node ID to evaluate, e.g. col:user_id"),
    _auth: None = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Computes the upstream dependencies and downstream impact blast radius for a given node."""
    store = await _resolve_graph_store(task_id, request)
    return store.get_blast_radius(node_id)


@router.get("/jobs/{task_id}/graph/search")
async def search_graph(
    request: Request,
    task_id: str,
    query: str = Query(..., description="Search query string"),
    limit: int = Query(10, ge=1, le=50),
    _auth: None = Depends(verify_api_key)
) -> List[Dict[str, Any]]:
    """Sub-millisecond fuzzy search over graph entities via in-memory trigram index."""
    store = await _resolve_graph_store(task_id, request)
    return store.search_entities(query, limit=limit)


@router.get("/jobs/{task_id}/graph/export")
async def export_graph(
    request: Request,
    task_id: str,
    format: str = Query("json", pattern="^(json|svg|html|cypher|graphml)$"),
    _auth: None = Depends(verify_api_key)
):
    """
    Exports the dataset knowledge graph in various enterprise formats:
    - html: Standalone Vis.js interactive canvas
    - svg: High-resolution vector graphic for Typst PDF or presentations
    - cypher: Neo4j import statements
    - graphml: Cytoscape / Gephi XML format
    - json: Standard node-link JSON
    """
    store = await _resolve_graph_store(task_id, request)
    cache_dir = _get_cache_dir()

    if format == "html":
        html_path = os.path.join(cache_dir, f"{task_id}_graph.html")
        if not os.path.exists(html_path):
            store.generate_html(html_path)
        return FileResponse(html_path, media_type="text/html", filename=f"{task_id}_graph.html")

    if format == "svg":
        svg_path = os.path.join(cache_dir, f"{task_id}_graph.svg")
        store.generate_svg(svg_path)
        return FileResponse(svg_path, media_type="image/svg+xml", filename=f"{task_id}_graph.svg")

    ext_map = {"cypher": "cyp", "graphml": "graphml", "json": "json"}
    ext = ext_map.get(format, "json")
    out_path = os.path.join(cache_dir, f"{task_id}_graph.{ext}")
    export_graph_to_format(store.graph, format, out_path)

    media_types = {
        "cypher": "text/plain",
        "graphml": "application/xml",
        "json": "application/json"
    }
    return FileResponse(out_path, media_type=media_types.get(format, "text/plain"), filename=f"{task_id}_graph.{ext}")


class SQLLineageRequest(BaseModel):
    sql: str
    dialect: Optional[str] = "snowflake"


@router.post("/jobs/{task_id}/graph/sql-lineage")
async def ingest_sql_lineage(
    request: Request,
    task_id: str,
    payload: SQLLineageRequest,
    _auth: None = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Parses a SQL statement and attaches its table/view/CTE lineage
    directly to the task's dataset knowledge graph.
    """
    store = await _resolve_graph_store(task_id, request)
    lineage = store.attach_sql(payload.sql, dialect=payload.dialect or "snowflake")
    # Re-cluster to incorporate new SQL nodes in topological domains
    store.cluster()
    cache_dir = _get_cache_dir()
    store.save_to_file(os.path.join(cache_dir, f"{task_id}_graph.json"))
    return {
        "success": True,
        "tables_found": len(lineage["nodes"]),
        "edges_added": len(lineage["edges"]),
        "lineage": lineage
    }


# ─── Understand-Anything Backend Integration Endpoints ───────────────────────

@router.get("/jobs/{task_id}/graph/layers")
async def get_graph_layers(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key)
) -> List[Dict[str, Any]]:
    """
    Returns the Understand-Anything canonical architectural layers
    (Ingestion, Features, Profiling, Quality, Security, Lineage) partitioning the graph.
    """
    store = await _resolve_graph_store(task_id, request)
    return store.get_layers()


@router.get("/jobs/{task_id}/graph/tour")
async def get_graph_audit_tour(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key)
) -> List[Dict[str, Any]]:
    """
    Generates an ordered, pedagogical Guided Audit Tour (TourStep[])
    for automated triage, auditor onboarding, or AI prompt injection.
    """
    store = await _resolve_graph_store(task_id, request)
    job = title_task_manager.get_job(task_id)
    analysis = job.result.get("analysis") if job and job.result else None
    return store.get_audit_tour(analysis_results=analysis)


@router.get("/jobs/{task_id}/graph/fingerprint")
async def get_graph_fingerprint(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Returns a deterministic SHA-256 structural fingerprint of the dataset graph.
    """
    store = await _resolve_graph_store(task_id, request)
    return store.get_fingerprint()


@router.get("/jobs/{task_id}/graph/ua-export")
async def export_ua_graph(
    request: Request,
    task_id: str,
    _auth: None = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Exports the graph in full Understand-Anything KnowledgeGraph JSON schema
    (version, kind, project, nodes, edges, layers, tour).
    """
    store = await _resolve_graph_store(task_id, request)
    job = title_task_manager.get_job(task_id)
    filename = job.filename if job else "dataset"
    analysis = job.result.get("analysis") if job and job.result else None
    return store.to_understand_anything_json(
        project_name=f"GetReport Audit: {filename}",
        description=f"Automated forensic data audit graph for {filename}",
    )


class DriftRequest(BaseModel):
    baseline_task_id: str


@router.post("/jobs/{task_id}/graph/drift")
async def compare_graph_drift(
    request: Request,
    task_id: str,
    payload: DriftRequest,
    _auth: None = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Compares the structural fingerprint of current dataset against a baseline task
    to detect schema drift, dropped features, new constraints, and topological shifts.
    """
    from app.services.understand_anything.fingerprint import compare_dataset_fingerprints

    current_store = await _resolve_graph_store(task_id, request)
    baseline_store = await _resolve_graph_store(payload.baseline_task_id, request)

    curr_fp = current_store.get_fingerprint()
    base_fp = baseline_store.get_fingerprint()

    comparison = compare_dataset_fingerprints(base_fp, curr_fp)
    return {
        "current_task_id": task_id,
        "baseline_task_id": payload.baseline_task_id,
        "comparison": comparison
    }

