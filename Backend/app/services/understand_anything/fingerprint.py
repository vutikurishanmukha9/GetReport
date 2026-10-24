"""
Understand-Anything Semantic Fingerprinting & Drift Detection for GetReport Backend.
Computes deterministic SHA-256 structural fingerprints of dataset schemas,
quality constraints, and statistical profiles to detect drift across successive runs.
"""
from __future__ import annotations
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional
import networkx as nx

logger = logging.getLogger(__name__)


def compute_dataset_fingerprint(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """
    Computes a deterministic fingerprint of a DatasetGraphStore's topology,
    schema types, and quality metrics.
    
    Returns:
        Dict with overall content_hash and per-entity signature hashes.
    """
    if not G:
        return {"content_hash": "", "entity_count": 0, "entities": {}}

    entity_fingerprints: Dict[str, Dict[str, Any]] = {}
    normalized_tokens: List[str] = []

    for nid in sorted(G.nodes()):
        attrs = G.nodes[nid]
        entity_type = str(attrs.get("entity_type", "Unknown"))
        name = str(attrs.get("name", nid))
        data_type = str(attrs.get("data_type", ""))
        null_pct = str(round(float(attrs.get("null_percentage", 0.0)), 2))
        degree = str(G.degree(nid))

        # Compact representation
        sig = f"{nid}|{entity_type}|{name}|{data_type}|{null_pct}|{degree}"
        sig_hash = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]

        entity_fingerprints[nid] = {
            "name": name,
            "entity_type": entity_type,
            "signature_hash": sig_hash,
            "degree": G.degree(nid)
        }
        normalized_tokens.append(f"{nid}:{sig_hash}")

    # Edge signatures
    edge_tokens: List[str] = []
    for u, v, k, d in sorted(G.edges(keys=True, data=True)):
        rel = str(d.get("relation_type", d.get("relation", "connected_to")))
        edge_tokens.append(f"{u}->{v}:{rel}")

    composite_raw = "\n".join(normalized_tokens) + "\n---\n" + "\n".join(edge_tokens)
    overall_hash = hashlib.sha256(composite_raw.encode("utf-8")).hexdigest()

    return {
        "content_hash": overall_hash,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "entities": entity_fingerprints
    }


def compare_dataset_fingerprints(
    baseline: Dict[str, Any],
    current: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compares two dataset fingerprints (e.g. from a past audit job vs. current audit job)
    to classify drift into: added entities, removed entities, and modified attributes.
    """
    base_entities = baseline.get("entities", {})
    curr_entities = current.get("entities", {})

    base_keys = set(base_entities.keys())
    curr_keys = set(curr_entities.keys())

    added_nodes = list(curr_keys - base_keys)
    removed_nodes = list(base_keys - curr_keys)
    common_nodes = base_keys & curr_keys

    modified_nodes: List[Dict[str, Any]] = []
    for k in common_nodes:
        if base_entities[k]["signature_hash"] != curr_entities[k]["signature_hash"]:
            modified_nodes.append({
                "id": k,
                "name": curr_entities[k]["name"],
                "old_degree": base_entities[k]["degree"],
                "new_degree": curr_entities[k]["degree"]
            })

    is_identical = baseline.get("content_hash") == current.get("content_hash")
    drift_score = 0.0
    total_tracked = max(len(base_keys) + len(curr_keys), 1)
    drift_score = round(float(len(added_nodes) + len(removed_nodes) + len(modified_nodes)) / float(total_tracked), 4)

    return {
        "identical": is_identical,
        "drift_score": drift_score,
        "added_count": len(added_nodes),
        "removed_count": len(removed_nodes),
        "modified_count": len(modified_nodes),
        "added_nodes": added_nodes[:20],
        "removed_nodes": removed_nodes[:20],
        "modified_nodes": modified_nodes[:20]
    }
