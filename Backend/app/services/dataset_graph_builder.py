import logging
from typing import Dict, Any, List, Optional
from app.core.graph_store import DatasetGraphStore

logger = logging.getLogger(__name__)

def build_dataset_graph(
    task_id: str,
    job_result: Dict[str, Any],
    ledger_issues: Optional[List[Dict[str, Any]]] = None
) -> DatasetGraphStore:
    """
    LightRAG-inspired Automated Dataset Knowledge Graph Builder.
    Transforms deterministic Polars profiling statistics, forensic Benford audits,
    correlation matrices, and Issue Ledger records into an explicit directed multi-graph.
    
    Zero external LLM token expenditure is needed for factual topology construction.
    """
    store = DatasetGraphStore(task_id)
    if not job_result:
        return store

    filename = job_result.get("filename", "Dataset")
    analysis = job_result.get("analysis", {})
    metadata = analysis.get("metadata", {})
    summary = analysis.get("summary", {})
    columns_info = analysis.get("columns", {})
    confidence = job_result.get("confidence", {})
    cleaning_report = job_result.get("cleaning_report", {})
    correlations = analysis.get("correlation", {}).get("strong_correlations", [])
    if not correlations:
        correlations = analysis.get("strong_correlations", [])

    # 1. Add Dataset Root Node
    dataset_node_id = f"dataset:{filename}"
    store.add_entity(
        entity_id=dataset_node_id,
        entity_type="Dataset",
        attributes={
            "name": filename,
            "filename": filename,
            "total_rows": metadata.get("total_rows"),
            "total_columns": metadata.get("total_columns"),
            "completeness_pct": round(100 - metadata.get("missing_value_pct", 0), 2),
            "confidence_score": confidence.get("dataset_confidence"),
            "confidence_grade": confidence.get("dataset_grade"),
            "domain": analysis.get("domain", "General"),
            "cleaning_operations": cleaning_report.get("total_changes", 0),
        }
    )

    # 2. Add Feature Column Nodes & CONTAINS_COLUMN Edges
    for col_name, stats in summary.items():
        col_node_id = f"col:{col_name}"
        info = columns_info.get(col_name, {})
        
        col_attrs = {
            "name": col_name,
            "column_name": col_name,
            "data_type": info.get("data_type", stats.get("data_type", "Unknown")),
            "semantic_type": info.get("semantic_type", "General"),
            "null_count": stats.get("null_count", 0),
            "null_percentage": round(stats.get("null_percentage", 0.0), 2),
            "unique_count": stats.get("unique_count"),
        }
        if "mean" in stats and stats["mean"] is not None:
            col_attrs["mean"] = round(float(stats["mean"]), 4) if isinstance(stats["mean"], (int, float)) else stats["mean"]
        if "min" in stats and stats["min"] is not None:
            col_attrs["min"] = stats["min"]
        if "max" in stats and stats["max"] is not None:
            col_attrs["max"] = stats["max"]
        if "std" in stats and stats["std"] is not None:
            col_attrs["std"] = round(float(stats["std"]), 4) if isinstance(stats["std"], (int, float)) else stats["std"]
        if "benford_status" in stats:
            col_attrs["benford_status"] = stats["benford_status"]

        store.add_entity(
            entity_id=col_node_id,
            entity_type="FeatureColumn",
            attributes=col_attrs
        )
        store.add_relation(
            src_id=dataset_node_id,
            tgt_id=col_node_id,
            relation_type="CONTAINS_COLUMN",
            attributes={
                "description": f"Dataset {filename} contains feature column '{col_name}'"
            }
        )

    # 3. Add Issue Ledger Nodes & HAS_QUALITY_ISSUE Edges
    active_issues = list(ledger_issues or [])
    # Also incorporate critical issues from confidence scoring if present
    crit_issues = confidence.get("critical_issues", [])
    if isinstance(crit_issues, list):
        for ci in crit_issues:
            if isinstance(ci, dict):
                active_issues.append(ci)

    for idx, iss in enumerate(active_issues):
        issue_id = iss.get("id", f"issue_{idx+1}")
        issue_node_id = f"issue:{issue_id}"
        col = iss.get("column", "General")
        issue_type = iss.get("issue_type", iss.get("issue", "DataQualityIssue"))
        severity = iss.get("severity", "medium")
        desc = iss.get("description", str(iss))
        action = iss.get("suggested_action", iss.get("remediation", "Review"))
        status = iss.get("status", "identified")

        store.add_entity(
            entity_id=issue_node_id,
            entity_type="DataQualityIssue",
            attributes={
                "issue_type": issue_type,
                "column": col,
                "severity": severity,
                "description": desc,
                "suggested_action": action,
                "status": status
            }
        )

        # Connect to target column or dataset
        col_node_id = f"col:{col}"
        target_src = col_node_id if store.graph.has_node(col_node_id) else dataset_node_id
        store.add_relation(
            src_id=target_src,
            tgt_id=issue_node_id,
            relation_type="HAS_QUALITY_ISSUE",
            attributes={
                "severity": severity,
                "description": f"Quality alert on {col}: {issue_type} - {desc}"
            }
        )

        # Connect issue to dataset confidence impact
        store.add_relation(
            src_id=issue_node_id,
            tgt_id=dataset_node_id,
            relation_type="AFFECTS_CONFIDENCE",
            attributes={
                "severity": severity,
                "description": f"Issue '{issue_type}' impacts overall dataset integrity grade {confidence.get('dataset_grade', 'N/A')}"
            }
        )

    # 4. Add Strong Correlation Edges between FeatureColumns
    for c in correlations:
        if isinstance(c, dict):
            c1 = c.get("column_a", c.get("col1"))
            c2 = c.get("column_b", c.get("col2"))
            r_val = c.get("r_value", c.get("correlation", 0.0))
        elif isinstance(c, (list, tuple)) and len(c) >= 3:
            c1, c2, r_val = c[0], c[1], float(c[2])
        else:
            continue

        if not c1 or not c2:
            continue

        node1 = f"col:{c1}"
        node2 = f"col:{c2}"
        if store.graph.has_node(node1) and store.graph.has_node(node2):
            direction = "positive" if r_val > 0 else "inverse"
            desc = f"Strong {direction} linear correlation between '{c1}' and '{c2}' (Pearson r = {r_val:.2f})"
            # Bidirectional correlation in graph
            store.add_relation(
                src_id=node1,
                tgt_id=node2,
                relation_type="CORRELATED_WITH",
                attributes={"r_value": r_val, "direction": direction, "description": desc},
                key=f"corr:{c1}<->{c2}"
            )
            store.add_relation(
                src_id=node2,
                tgt_id=node1,
                relation_type="CORRELATED_WITH",
                attributes={"r_value": r_val, "direction": direction, "description": desc},
                key=f"corr:{c2}<->{c1}"
            )

    # 5. Add Cleaning Action Nodes from Cleaning Report
    renamed = cleaning_report.get("columns_renamed", {})
    if renamed and isinstance(renamed, dict):
        clean_node_id = f"action:standardize_headers"
        store.add_entity(
            entity_id=clean_node_id,
            entity_type="CleaningTransformation",
            attributes={"action": "header_standardization", "count": len(renamed)}
        )
        for orig, new_name in list(renamed.items())[:8]:
            if store.graph.has_node(f"col:{new_name}"):
                store.add_relation(
                    src_id=f"col:{new_name}",
                    tgt_id=clean_node_id,
                    relation_type="TRANSFORMED_BY",
                    attributes={"description": f"Renamed header from '{orig}' to standardized '{new_name}'"}
                )

    # 6. Add Kats Time Series Changepoints (Structural Shift Topology)
    ts_analysis = analysis.get("time_series", {})
    changepoint_dict = ts_analysis.get("changepoints", {})
    for col_name, cps in changepoint_dict.items():
        col_node = f"col:{col_name}"
        if store.graph.has_node(col_node) and isinstance(cps, list):
            for cp in cps:
                tau = cp.get("index")
                shift = cp.get("mean_shift")
                ts_str = cp.get("timestamp") or f"index_{tau}"
                cp_node_id = f"changepoint:{col_name}:{tau}"
                store.add_entity(
                    entity_id=cp_node_id,
                    entity_type="StructuralChangepoint",
                    attributes={
                        "column": col_name,
                        "timestamp": ts_str,
                        "mean_shift": shift,
                        "pre_mean": cp.get("pre_mean"),
                        "post_mean": cp.get("post_mean"),
                        "llr_score": cp.get("llr_score")
                    }
                )
                store.add_relation(
                    src_id=col_node,
                    tgt_id=cp_node_id,
                    relation_type="STRUCTURAL_CHANGE_AT",
                    attributes={
                        "description": f"Column '{col_name}' underwent a structural shift at {ts_str} (mean shifted by {shift})"
                    }
                )

    # 7. Add OpenMetadata PII Findings (Security & Compliance Topology)
    pii_findings = job_result.get("pii", {}).get("findings", {})
    for col_name, pii_info in pii_findings.items():
        col_node = f"col:{col_name}"
        if store.graph.has_node(col_node):
            pii_type = pii_info.get("pii_type", "PII")
            pii_node_id = f"pii:{col_name}"
            store.add_entity(
                entity_id=pii_node_id,
                entity_type="SecurityRisk",
                attributes={
                    "pii_type": pii_type,
                    "confidence": pii_info.get("confidence"),
                    "sample_masked": pii_info.get("sample_masked")
                }
            )
            store.add_relation(
                src_id=col_node,
                tgt_id=pii_node_id,
                relation_type="HAS_PII_RISK",
                attributes={
                    "description": f"Column '{col_name}' exposes {pii_type} data requiring masking or encryption"
                }
            )

    # 8. Add Data Quality Test Failures (Governance Topology)
    dq_results = job_result.get("data_quality", {}).get("results", [])
    for dq in dq_results:
        if not dq.get("success") and dq.get("column"):
            col_name = dq["column"]
            col_node = f"col:{col_name}"
            if store.graph.has_node(col_node):
                t_type = dq.get("test_type", "Assertion")
                dq_node_id = f"dq_failure:{col_name}:{t_type}"
                store.add_entity(
                    entity_id=dq_node_id,
                    entity_type="DataQualityAssertion",
                    attributes={
                        "test_type": t_type,
                        "description": dq.get("description"),
                        "violations": dq.get("violation_count", 0)
                    }
                )
                store.add_relation(
                    src_id=col_node,
                    tgt_id=dq_node_id,
                    relation_type="FAILS_QUALITY_TEST",
                    attributes={
                        "description": f"Column '{col_name}' failed quality assertion '{t_type}': {dq.get('description')}"
                    }
                )

    logger.info(
        f"Built dataset knowledge graph for task {task_id}: "
        f"{store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges"
    )
    return store
