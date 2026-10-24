"""
Understand-Anything Layer Detector for GetReport Backend.
Classifies dataset graph entities into 6 canonical architectural data layers:
1. Ingestion & Raw Sources (layer:ingestion)
2. Schema & Feature Architecture (layer:features)
3. Statistical & Anomaly Profiling (layer:profiling)
4. Data Quality & Constraints (layer:quality)
5. Security & Governance (layer:security)
6. Downstream Lineage & Queries (layer:lineage)
"""
from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional
import networkx as nx

logger = logging.getLogger(__name__)

CANONICAL_DATA_LAYERS = [
    {
        "id": "layer:ingestion",
        "name": "Ingestion & Raw Sources",
        "description": "Original raw dataset files, Snowflake tables, and external database extraction sources.",
        "types": {"Dataset", "SQLSourceTable", "RawSource"}
    },
    {
        "id": "layer:features",
        "name": "Feature Architecture",
        "description": "Dataset columns, data types, primary keys, and categorical/numerical feature dimensions.",
        "types": {"FeatureColumn", "Column", "Dimension", "Metric"}
    },
    {
        "id": "layer:profiling",
        "name": "Statistical & Anomaly Profiling",
        "description": "Statistical distributions, Benford law compliance, correlation clusters, and structural changepoints.",
        "types": {"StatisticalProfile", "BenfordAnomaly", "StructuralChangepoint", "Correlation"}
    },
    {
        "id": "layer:quality",
        "name": "Data Quality & Integrity Constraints",
        "description": "Null rates, out-of-range violations, Great Expectations data contracts, and cleaning transformations.",
        "types": {"DataQualityIssue", "CleaningTransformation", "IntegrityViolation", "Expectation"}
    },
    {
        "id": "layer:security",
        "name": "Security & Governance Risks",
        "description": "Personally Identifiable Information (PII), SSNs, emails, credit card tokens, and compliance flags.",
        "types": {"SecurityRisk", "PIIExposure", "SensitiveData"}
    },
    {
        "id": "layer:lineage",
        "name": "Downstream Lineage & Transformations",
        "description": "Derived views, SQL CTEs, analytical models, and target export artifacts.",
        "types": {"SQLTargetView", "SQLCTE", "ReportArtifact", "ExportView"}
    }
]


def detect_dataset_layers(G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    """
    Partitions the nodes of a DatasetGraphStore into structured architectural layers.
    
    Returns:
        List of Layer dictionaries compatible with Understand-Anything schema:
        [
            {
                "id": "layer:ingestion",
                "name": "Ingestion & Raw Sources",
                "description": "...",
                "nodeIds": ["dataset:sales.csv", ...]
            },
            ...
        ]
    """
    if not G or G.number_of_nodes() == 0:
        return []

    layer_node_map: Dict[str, List[str]] = {layer["id"]: [] for layer in CANONICAL_DATA_LAYERS}

    for nid, attrs in G.nodes(data=True):
        entity_type = attrs.get("entity_type", attrs.get("type", "Unknown"))
        assigned = False

        # Match by explicit entity_type
        for layer in CANONICAL_DATA_LAYERS:
            if entity_type in layer["types"]:
                layer_node_map[layer["id"]].append(nid)
                assigned = True
                break

        # Fallback heuristic matching based on node ID prefix
        if not assigned:
            if nid.startswith("dataset:") or nid.startswith("raw:"):
                layer_node_map["layer:ingestion"].append(nid)
            elif nid.startswith("col:") or nid.startswith("feature:"):
                layer_node_map["layer:features"].append(nid)
            elif nid.startswith("issue:") or nid.startswith("clean:"):
                layer_node_map["layer:quality"].append(nid)
            elif nid.startswith("pii:") or nid.startswith("sec:"):
                layer_node_map["layer:security"].append(nid)
            elif nid.startswith("table:") or nid.startswith("view:") or nid.startswith("cte:"):
                layer_node_map["layer:lineage"].append(nid)
            else:
                layer_node_map["layer:features"].append(nid)

    # Assemble non-empty layers
    layers: List[Dict[str, Any]] = []
    for layer_meta in CANONICAL_DATA_LAYERS:
        matched_nodes = layer_node_map[layer_meta["id"]]
        if matched_nodes:
            layers.append({
                "id": layer_meta["id"],
                "name": layer_meta["name"],
                "description": layer_meta["description"],
                "nodeIds": matched_nodes,
                "nodeCount": len(matched_nodes)
            })

    return layers
