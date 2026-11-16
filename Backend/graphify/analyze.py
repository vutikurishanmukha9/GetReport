"""
Forensic Topological Analysis for Graphify.
Extracts structural God Nodes (central architectural hubs) and surprising cross-domain connections.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
import networkx as nx


def god_nodes(G: nx.Graph, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Identifies the top central hub nodes ranked by degree centrality.
    
    Returns:
        List of dicts with 'id', 'label', and 'degree'.
    """
    if not G or G.number_of_nodes() == 0:
        return []

    sorted_degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    results: List[Dict[str, Any]] = []

    for n, deg in sorted_degrees[:top_n]:
        if deg > 0:
            node_attrs = G.nodes.get(n, {})
            label = str(node_attrs.get("name") or node_attrs.get("label") or n)
            results.append({
                "id": n,
                "label": label,
                "degree": int(deg)
            })

    return results


def surprising_connections(
    G: nx.Graph,
    communities: Dict[int, List[str]],
    top_n: int = 6
) -> List[Dict[str, Any]]:
    """
    Identifies high-impact edges that bridge between distinct community clusters.
    These cross-domain connections often reveal unexpected data lineage or cross-table anomalies.
    
    Returns:
        List of dicts describing cross-community edges.
    """
    if not G or not communities:
        return []

    node_to_comm: Dict[str, int] = {}
    for cid, members in communities.items():
        for m in members:
            node_to_comm[m] = cid

    surprises: List[Dict[str, Any]] = []
    seen_pairs = set()

    for u, v, d in G.edges(data=True):
        cu = node_to_comm.get(u)
        cv = node_to_comm.get(v)

        if cu is not None and cv is not None and cu != cv:
            pair_key = tuple(sorted([str(u), str(v)]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            relation = d.get("relation") or d.get("relation_type") or "cross_domain"
            weight = float(d.get("weight", 1.0))

            surprises.append({
                "source": u,
                "target": v,
                "source_community": cu,
                "target_community": cv,
                "relation": str(relation),
                "weight": weight
            })

            if len(surprises) >= top_n:
                break

    return surprises


def suggest_questions(G: nx.Graph, communities: Dict[int, List[str]]) -> List[str]:
    """
    Generates intelligent forensic prompts based on graph structure and bridges.
    """
    hubs = god_nodes(G, top_n=3)
    bridges = surprising_connections(G, communities, top_n=2)
    questions: List[str] = []

    for h in hubs:
        questions.append(f"What drives the high connectivity around '{h['label']}'?")

    for b in bridges:
        questions.append(f"Why is '{b['source']}' connected to '{b['target']}' across domains?")

    return questions
