"""
Understand-Anything Guided Tour Generator for GetReport Backend.
Generates ordered, pedagogical Data Audit Investigation Tours (TourStep[])
guiding auditors and AI agents through the dataset's critical architecture:
1. Ingestion Overview & Scale
2. Critical Quality & Missing Value Bottlenecks
3. Forensic Benford & Statistical Outliers
4. Governance & PII Exposures
5. Cross-Feature Correlations & Structural Hubs
6. Downstream Blast Radius & SQL Lineage
"""
from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional
import networkx as nx

logger = logging.getLogger(__name__)


def generate_dataset_audit_tour(
    G: nx.MultiDiGraph,
    analysis_results: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Constructs an ordered list of TourStep objects prioritizing the most urgent
    and architecturally significant nodes for data auditors and LLM agents.
    
    Returns:
        List of TourStep dicts compatible with Understand-Anything schema:
        [
            {
                "order": 1,
                "title": "Dataset Ingestion & Scale",
                "description": "...",
                "nodeIds": ["dataset:sales.csv"],
                "auditInsight": "..."
            },
            ...
        ]
    """
    if not G or G.number_of_nodes() == 0:
        return []

    tour_steps: List[Dict[str, Any]] = []
    order = 1

    # ─── STEP 1: Ingestion Overview & Scale ──────────────────────────────────
    dataset_nodes = [n for n, d in G.nodes(data=True) if d.get("entity_type") in ("Dataset", "RawSource") or n.startswith("dataset:")]
    if dataset_nodes:
        d_node = dataset_nodes[0]
        d_attrs = G.nodes[d_node]
        row_count = d_attrs.get("total_rows", analysis_results.get("summary", {}).get("total_rows", "N/A") if analysis_results else "N/A")
        col_count = len([n for n in G.nodes if n.startswith("col:")])

        tour_steps.append({
            "order": order,
            "title": "Dataset Ingestion & Structural Scale",
            "description": f"Entry point showing original dataset source '{d_attrs.get('name', d_node)}'. "
                           f"Contains {row_count} rows across {col_count} feature columns.",
            "nodeIds": [d_node],
            "auditInsight": f"Baseline dataset dimension initialized with {col_count} mapped columns."
        })
        order += 1

    # ─── STEP 2: Critical Data Quality & Integrity Anomalies ──────────────────
    quality_issue_nodes = [
        (n, d) for n, d in G.nodes(data=True)
        if d.get("entity_type") in ("DataQualityIssue", "IntegrityViolation") or n.startswith("issue:")
    ]
    if quality_issue_nodes:
        # Sort by severity: critical > high > medium > low
        severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        quality_issue_nodes.sort(key=lambda x: severity_rank.get(str(x[1].get("severity", "medium")).lower(), 1), reverse=True)
        top_quality_ids = [n for n, _ in quality_issue_nodes[:4]]

        tour_steps.append({
            "order": order,
            "title": "Data Quality & Integrity Bottlenecks",
            "description": f"Discovered {len(quality_issue_nodes)} data quality violations impacting confidence scores. "
                           f"Highlighted nodes show the highest-severity constraint failures.",
            "nodeIds": top_quality_ids,
            "auditInsight": "Immediate remediation or filtering required to prevent downstream model bias."
        })
        order += 1

    # ─── STEP 3: Security & PII Privacy Exposures ─────────────────────────────
    security_nodes = [
        (n, d) for n, d in G.nodes(data=True)
        if d.get("entity_type") in ("SecurityRisk", "PIIExposure") or n.startswith("pii:") or n.startswith("sec:")
    ]
    if security_nodes:
        sec_ids = [n for n, _ in security_nodes[:5]]
        tour_steps.append({
            "order": order,
            "title": "Security, PII & Governance Exposures",
            "description": f"Identified {len(security_nodes)} sensitive PII elements (such as SSNs, credentials, or private contact details) "
                           f"requiring automated masking or role-based access control.",
            "nodeIds": sec_ids,
            "auditInsight": "High regulatory compliance risk. Apply column-level hashing or pseudonymization."
        })
        order += 1

    # ─── STEP 4: Statistical Anomalies & Benford Violations ────────────────────
    benford_nodes = [
        (n, d) for n, d in G.nodes(data=True)
        if d.get("benford_status") in ("Flagged", "Anomalous") or "benford" in str(d.get("issue_type", "")).lower()
    ]
    if benford_nodes:
        benford_ids = [n for n, _ in benford_nodes[:4]]
        tour_steps.append({
            "order": order,
            "title": "Forensic Benford & Statistical Deviations",
            "description": f"Detected first-digit Benford distribution anomalies across {len(benford_nodes)} financial or numeric fields, "
                           f"which may indicate synthetic data generation, transaction fraud, or artificial capping.",
            "nodeIds": benford_ids,
            "auditInsight": "Conduct forensic transaction audits on flagged financial features."
        })
        order += 1

    # ─── STEP 5: Structural Hubs (God Nodes) & High Correlations ──────────────
    # Identify high-degree columns
    feature_nodes = [
        (n, G.degree(n)) for n, d in G.nodes(data=True)
        if n.startswith("col:")
    ]
    feature_nodes.sort(key=lambda x: x[1], reverse=True)
    if feature_nodes:
        top_hubs = [n for n, deg in feature_nodes[:4]]
        tour_steps.append({
            "order": order,
            "title": "Central Structural Hubs & Feature Drivers",
            "description": f"These central columns have the highest graph connectivity and cross-domain correlation. "
                           f"Changes or errors in these features propagate widely across the dataset.",
            "nodeIds": top_hubs,
            "auditInsight": "Core dependency anchors. Must be protected from schema drift."
        })
        order += 1

    # ─── STEP 6: SQL Lineage & Downstream Transformation ──────────────────────
    lineage_nodes = [
        n for n in G.nodes
        if n.startswith("table:") or n.startswith("view:") or n.startswith("cte:")
    ]
    if lineage_nodes:
        tour_steps.append({
            "order": order,
            "title": "Downstream SQL Lineage & Analytical Feeds",
            "description": f"Trace of SQL tables and derived views feeding downstream reporting pipelines.",
            "nodeIds": lineage_nodes[:5],
            "auditInsight": "Review views to ensure no deprecated columns are queried."
        })
        order += 1

    return tour_steps
