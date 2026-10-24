"""
Graphify Query & Blast Radius Bridge for GetReport.
Provides trigram entity search, causal shortest-path traversal, and automated impact blast radius analysis.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import deque
import networkx as nx

logger = logging.getLogger(__name__)


def _trigrams(text: str) -> Set[str]:
    """Generates trigrams for fuzzy string candidate discovery."""
    s = text.lower().strip()
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i+3] for i in range(len(s) - 2)}


def search_graph_entities(G: nx.MultiDiGraph, query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Sub-millisecond in-memory fuzzy entity lookup using trigram overlap and token scoring.
    """
    if not G or not query:
        return []

    clean_query = query.strip().lower()
    q_tokens = [t for t in clean_query.replace("_", " ").replace(":", " ").split() if len(t) > 1]
    q_tri = _trigrams(clean_query)

    scored: List[Tuple[float, str]] = []

    for nid, data in G.nodes(data=True):
        label = str(data.get("name") or data.get("label") or nid).lower()
        node_id_lower = str(nid).lower()
        e_type = str(data.get("entity_type", "")).lower()

        score = 0.0

        # Exact match bonus
        if clean_query == label or clean_query == node_id_lower:
            score += 100.0
        elif clean_query in label or clean_query in node_id_lower:
            score += 50.0

        # Token match bonus
        for tok in q_tokens:
            if tok in label:
                score += 15.0
            if tok in node_id_lower:
                score += 10.0
            if tok in e_type:
                score += 5.0

        # Trigram Jaccard similarity bonus
        n_tri = _trigrams(label)
        if q_tri and n_tri:
            intersection = len(q_tri & n_tri)
            union = len(q_tri | n_tri)
            score += (intersection / union) * 20.0

        if score > 0:
            scored.append((score, nid))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sc, nid in scored[:limit]:
        d = G.nodes[nid]
        results.append({
            "id": nid,
            "label": d.get("name") or d.get("label") or nid,
            "entity_type": d.get("entity_type", "Entity"),
            "community_label": d.get("community_label", "Unknown"),
            "score": round(sc, 2),
            "attributes": {k: v for k, v in d.items() if k not in ("name", "label", "entity_type")}
        })
    return results


def find_causal_path(
    G: nx.MultiDiGraph,
    source_id: str,
    target_id: str,
    max_hops: int = 4
) -> Dict[str, Any]:
    """
    Finds the shortest directed and undirected causal path between two entities,
    returning structured node sequences, edge descriptions, and markdown summary.
    """
    if not G.has_node(source_id) or not G.has_node(target_id):
        return {
            "found": False,
            "source_id": source_id,
            "target_id": target_id,
            "message": f"One or both nodes do not exist in graph."
        }

    # 1. Try directed shortest path first
    path_nodes = None
    is_directed = False
    try:
        path_nodes = nx.shortest_path(G, source=source_id, target=target_id)
        is_directed = True
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # 2. Fall back to undirected shortest path
        try:
            undirected_view = G.to_undirected(as_view=True)
            path_nodes = nx.shortest_path(undirected_view, source=source_id, target=target_id)
            is_directed = False
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_nodes = None

    if not path_nodes:
        return {
            "found": False,
            "source_id": source_id,
            "target_id": target_id,
            "message": "No causal connection found between the specified entities."
        }

    # Extract path nodes and connecting edge metadata
    nodes_info = []
    for nid in path_nodes:
        d = G.nodes[nid]
        nodes_info.append({
            "id": nid,
            "label": d.get("name") or d.get("label") or nid,
            "entity_type": d.get("entity_type", "Entity"),
            "community_label": d.get("community_label", "Unknown")
        })

    edges_info = []
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        edge_data = None
        # Check u -> v
        if G.has_edge(u, v):
            edge_keys = list(G[u][v].keys())
            edge_data = G[u][v][edge_keys[0]]
            direction = "forward"
        elif G.has_edge(v, u):
            edge_keys = list(G[v][u].keys())
            edge_data = G[v][u][edge_keys[0]]
            direction = "reverse"
        else:
            direction = "undirected"
            edge_data = {"relation_type": "CONNECTED_TO", "description": "Relational link"}

        edges_info.append({
            "source": u,
            "target": v,
            "direction": direction,
            "relation_type": edge_data.get("relation_type", edge_data.get("relation", "CONNECTED")),
            "description": edge_data.get("description", "")
        })

    # Generate grounded natural-language explanation
    steps = []
    for i, e in enumerate(edges_info):
        src_label = nodes_info[i]["label"]
        tgt_label = nodes_info[i + 1]["label"]
        rel = e["relation_type"]
        desc = f" ({e['description']})" if e["description"] else ""
        arrow = "-->" if e["direction"] == "forward" else "<--" if e["direction"] == "reverse" else "<-->"
        steps.append(f"`{src_label}` {arrow} [{rel}] {arrow} `{tgt_label}`{desc}")

    explanation_md = f"**Causal Path ({len(path_nodes)-1} hops):**\n" + "\n".join(f"- {s}" for s in steps)

    return {
        "found": True,
        "source_id": source_id,
        "target_id": target_id,
        "hop_count": len(path_nodes) - 1,
        "is_directed": is_directed,
        "path_nodes": nodes_info,
        "path_edges": edges_info,
        "explanation_markdown": explanation_md
    }


def compute_blast_radius(
    G: nx.MultiDiGraph,
    node_id: str,
    max_depth: int = 3
) -> Dict[str, Any]:
    """
    Computes the upstream dependencies and downstream blast radius of a node,
    identifying what other columns, quality issues, or tests will be impacted.
    """
    if not G.has_node(node_id):
        return {
            "node_id": node_id,
            "found": False,
            "message": f"Node '{node_id}' does not exist in graph."
        }

    root_data = G.nodes[node_id]

    # 1. Downstream Traversal (Successors - things that depend on this node)
    downstream: Set[str] = set()
    queue = deque([(node_id, 0)])
    visited = {node_id}

    while queue:
        curr, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for succ in G.successors(curr):
            if succ not in visited:
                visited.add(succ)
                downstream.add(succ)
                queue.append((succ, depth + 1))

    # 2. Upstream Traversal (Predecessors - things this node depends on)
    upstream: Set[str] = set()
    queue = deque([(node_id, 0)])
    visited = {node_id}

    while queue:
        curr, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for pred in G.predecessors(curr):
            if pred not in visited:
                visited.add(pred)
                upstream.add(pred)
                queue.append((pred, depth + 1))

    # 3. Categorize impacted entities
    def _package_nodes(node_set: Set[str]) -> List[Dict[str, Any]]:
        pkg = []
        for nid in node_set:
            d = G.nodes[nid]
            pkg.append({
                "id": nid,
                "label": d.get("name") or d.get("label") or nid,
                "entity_type": d.get("entity_type", "Entity"),
                "severity": d.get("severity", "none"),
                "community_label": d.get("community_label", "Unknown")
            })
        return pkg

    downstream_nodes = _package_nodes(downstream)
    upstream_nodes = _package_nodes(upstream)

    # Compute risk score (number of affected nodes weighted by severity)
    severity_weights = {"critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0, "none": 0.5}
    risk_score = 0.0
    for n in downstream_nodes:
        sev = str(n.get("severity", "none")).lower()
        risk_score += severity_weights.get(sev, 1.0)

    risk_level = "High" if risk_score > 10 else "Medium" if risk_score > 3 else "Low"

    return {
        "node_id": node_id,
        "label": root_data.get("name") or root_data.get("label") or node_id,
        "entity_type": root_data.get("entity_type", "Entity"),
        "found": True,
        "risk_level": risk_level,
        "risk_score": round(risk_score, 1),
        "total_impacted_count": len(downstream_nodes),
        "total_upstream_count": len(upstream_nodes),
        "downstream_impact": downstream_nodes,
        "upstream_dependencies": upstream_nodes
    }
