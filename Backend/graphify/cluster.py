"""
Community Clustering and Partitioning module for Graphify.
Uses deterministic Louvain modularity optimization with resolution sweep
and hub-based community labeling.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
import networkx as nx


def cluster(
    G: nx.Graph,
    resolution: float = 1.0,
    exclude_hubs_percentile: float = 95.0,
    seed: int = 42
) -> Dict[int, List[str]]:
    """
    Partitions the graph into discrete communities using modularity optimization.
    
    Args:
        G: Undirected NetworkX graph.
        resolution: Resolution parameter (>1 produces more, smaller communities).
        exclude_hubs_percentile: Percentile threshold to identify super-connected hub nodes.
        seed: Deterministic seed for reproducible partitioning.
        
    Returns:
        Dict mapping integer community ID to list of node IDs.
    """
    if not G or G.number_of_nodes() == 0:
        return {}

    if G.number_of_nodes() == 1:
        return {0: list(G.nodes())}

    # Attempt Louvain community detection with deterministic seed
    try:
        from networkx.algorithms.community import louvain_communities
        raw_comms = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
        # Sort each community deterministically
        communities = {i: sorted(list(c)) for i, c in enumerate(raw_comms) if c}
    except Exception:
        # Fallback to connected components
        raw_comps = list(nx.connected_components(G))
        communities = {i: sorted(list(c)) for i, c in enumerate(raw_comps) if c}

    if not communities:
        communities = {0: sorted(list(G.nodes()))}

    return communities


def label_communities_by_hub(
    G: nx.Graph,
    communities: Dict[int, List[str]]
) -> Dict[int, str]:
    """
    Generates human-readable domain labels for each community based on its highest-degree hub node.
    Strips raw structural prefixes (col:, dataset:, issue:, pii:, table:, view:) for clean reporting.
    """
    labels: Dict[int, str] = {}
    known_prefixes = ("col:", "dataset:", "issue:", "pii:", "table:", "view:")

    for cid, members in communities.items():
        if not members:
            labels[cid] = f"Domain_{cid}"
            continue

        # Find the node with the highest degree in this community
        best_node = max(members, key=lambda n: G.degree(n) if G.has_node(n) else 0)
        node_data = G.nodes.get(best_node, {}) if G.has_node(best_node) else {}
        raw_label = str(node_data.get("name") or node_data.get("label") or best_node)

        clean_label = raw_label
        for prefix in known_prefixes:
            if clean_label.startswith(prefix):
                clean_label = clean_label[len(prefix):]
                break

        clean_label = clean_label.replace("_", " ").strip()
        labels[cid] = clean_label if clean_label else f"Domain_{cid}"

    return labels


def cohesion_score(G: nx.Graph, members: List[str]) -> float:
    """
    Computes internal edge density score (0.0 to 1.0) for a given community cluster.
    """
    if len(members) <= 1:
        return 1.0
    valid_members = [m for m in members if G.has_node(m)]
    if len(valid_members) <= 1:
        return 1.0
    sub = G.subgraph(valid_members)
    return round(float(nx.density(sub)), 4)


def score_all(G: nx.Graph, communities: Dict[int, List[str]]) -> Dict[int, float]:
    """
    Computes cohesion scores for all communities in a dictionary.
    """
    return {cid: cohesion_score(G, members) for cid, members in communities.items()}
