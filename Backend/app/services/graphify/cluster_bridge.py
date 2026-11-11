"""
Graphify Clustering Bridge for GetReport.
Wraps Graphify's Rust-native Leiden and deterministic Louvain algorithms
with cohesion scoring, automatic hub-based labeling, god nodes, and surprising connection discovery.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Any, Tuple, Optional
import networkx as nx

# Import directly from graphify package
try:
    from graphify.cluster import cluster, label_communities_by_hub, cohesion_score
    from graphify.analyze import god_nodes, surprising_connections
    HAS_GRAPHIFY_CLUSTER = True
except ImportError as err:
    logging.getLogger(__name__).warning("Failed to import graphify cluster/analyze: %s", err)
    HAS_GRAPHIFY_CLUSTER = False

logger = logging.getLogger(__name__)


def _builtin_god_nodes(G: nx.Graph, top_n: int = 10) -> List[Dict[str, Any]]:
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    results = []
    for n, deg in degrees[:top_n]:
        if deg > 0:
            results.append({
                "id": n,
                "label": str(G.nodes[n].get("name") or G.nodes[n].get("label") or n),
                "degree": deg
            })
    return results


def _builtin_communities(G: nx.Graph) -> Dict[int, List[str]]:
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, seed=42)
        return {i: list(c) for i, c in enumerate(comms)}
    except Exception:
        comps = list(nx.connected_components(G))
        return {i: list(c) for i, c in enumerate(comps)}


def _builtin_label_hubs(G: nx.Graph, communities: Dict[int, List[str]]) -> Dict[int, str]:
    labels = {}
    for cid, members in communities.items():
        if not members:
            labels[cid] = f"Domain_{cid}"
            continue
        best_node = max(members, key=lambda n: G.degree(n))
        best_label = G.nodes[best_node].get("name") or G.nodes[best_node].get("label") or best_node
        clean_label = str(best_label).replace("col:", "").replace("dataset:", "").replace("issue:", "").replace("pii:", "")
        labels[cid] = clean_label or f"Domain_{cid}"
    return labels


def _builtin_cohesion(G: nx.Graph, members: List[str]) -> float:
    if len(members) <= 1:
        return 1.0
    sub = G.subgraph(members)
    return round(float(nx.density(sub)), 4)


def _builtin_surprises(G: nx.Graph, communities: Dict[int, List[str]], top_n: int = 6) -> List[Dict[str, Any]]:
    node_to_comm = {n: cid for cid, members in communities.items() for n in members}
    surprises = []
    for u, v, d in G.edges(data=True):
        cu = node_to_comm.get(u)
        cv = node_to_comm.get(v)
        if cu is not None and cv is not None and cu != cv:
            surprises.append({
                "source": u,
                "target": v,
                "source_community": cu,
                "target_community": cv,
                "relation": d.get("relation", "cross_domain"),
                "weight": d.get("weight", 1.0)
            })
            if len(surprises) >= top_n:
                break
    return surprises


def cluster_and_analyze_dataset_graph(
    G: nx.MultiDiGraph,
    resolution: float = 1.0,
    top_god_nodes: int = 10,
    top_surprises: int = 6
) -> Tuple[Dict[int, List[str]], Dict[int, str], Dict[str, Any]]:
    """
    Executes Leiden / Louvain clustering, hub labeling, and forensic topological analysis
    over an ingested dataset knowledge graph.
    
    Args:
        G: The directed MultiDiGraph built by dataset_graph_builder.
        resolution: Leiden/Louvain resolution parameter (>1 produces more, smaller communities).
        top_god_nodes: Number of central structural hubs to return.
        top_surprises: Number of surprising cross-community bridging edges to return.
        
    Returns:
        communities: {community_id: [node_ids]}
        community_labels: {community_id: "HubName"}
        analysis_metadata: Dict with god_nodes, surprising_connections, and cohesion metrics.
    """
    if not G or G.number_of_nodes() == 0:
        return {}, {}, {
            "total_communities": 0,
            "cohesion_scores": {},
            "god_nodes": [],
            "surprising_connections": [],
            "community_summary": []
        }

    # 1. Convert MultiDiGraph to a simple undirected Graph for community detection
    # and metric calculations, preserving maximal edge weights.
    undirected_simple = nx.Graph()
    dataset_file = str(G.graph.get("filename") or "dataset.csv")
    if "." not in dataset_file:
        dataset_file = f"{dataset_file}.csv"

    for n, attrs in G.nodes(data=True):
        clean_node_attrs = dict(attrs)
        clean_node_attrs.setdefault("source_file", dataset_file)
        clean_node_attrs.setdefault("label", str(clean_node_attrs.get("name") or clean_node_attrs.get("column_name") or n))
        undirected_simple.add_node(n, **clean_node_attrs)

    for u, v, k, attrs in G.edges(keys=True, data=True):
        weight = float(attrs.get("weight", 1.0))
        # Ensure relation string is present
        rel = attrs.get("relation_type", attrs.get("relation", "connected_to"))
        clean_attrs = dict(attrs)
        clean_attrs["relation"] = rel
        clean_attrs["weight"] = weight

        if undirected_simple.has_edge(u, v):
            cur_weight = undirected_simple[u][v].get("weight", 1.0)
            if weight > cur_weight:
                undirected_simple[u][v].update(clean_attrs)
        else:
            undirected_simple.add_edge(u, v, **clean_attrs)

    # 2. Execute Leiden / Louvain clustering via Graphify or built-in NetworkX
    communities: Dict[int, List[str]] = {}
    community_labels: Dict[int, str] = {}
    cohesion_scores: Dict[int, float] = {}

    if HAS_GRAPHIFY_CLUSTER:
        try:
            communities = cluster(
                undirected_simple,
                resolution=resolution,
                exclude_hubs_percentile=95.0
            )
            community_labels = label_communities_by_hub(undirected_simple, communities)
            
            for cid, members in communities.items():
                try:
                    c_score = cohesion_score(undirected_simple, members)
                    cohesion_scores[cid] = round(float(c_score), 4)
                except Exception as c_err:
                    logger.debug("Error computing cohesion for cid %s: %s", cid, c_err)
                    cohesion_scores[cid] = 0.0
        except Exception as e:
            logger.warning("Graphify cluster execution error: %s. Using built-in partition.", e)
            communities = _builtin_communities(undirected_simple)
            community_labels = _builtin_label_hubs(undirected_simple, communities)
            for cid, members in communities.items():
                cohesion_scores[cid] = _builtin_cohesion(undirected_simple, members)
    else:
        # Robust built-in NetworkX clustering
        communities = _builtin_communities(undirected_simple)
        community_labels = _builtin_label_hubs(undirected_simple, communities)
        for cid, members in communities.items():
            cohesion_scores[cid] = _builtin_cohesion(undirected_simple, members)

    # 3. Annotate the original MultiDiGraph with community metadata and degree
    for cid, members in communities.items():
        c_label = community_labels.get(cid, f"Domain_{cid}")
        for nid in members:
            if G.has_node(nid):
                G.nodes[nid]["community"] = cid
                G.nodes[nid]["community_label"] = c_label
                G.nodes[nid]["degree"] = G.degree(nid)

    # 4. Extract God Nodes and Surprising Cross-Community Connections
    hubs: List[Dict[str, Any]] = []
    surprises: List[Dict[str, Any]] = []

    if HAS_GRAPHIFY_CLUSTER:
        try:
            raw_hubs = god_nodes(undirected_simple, top_n=top_god_nodes)
            for h in raw_hubs:
                nid = h.get("id")
                node_data = G.nodes.get(nid, {})
                hubs.append({
                    "id": nid,
                    "label": h.get("label", nid),
                    "degree": h.get("degree", 0),
                    "entity_type": node_data.get("entity_type", "Entity"),
                    "community": node_data.get("community", 0),
                    "community_label": node_data.get("community_label", "Unknown")
                })
        except Exception as e:
            logger.warning("Error calculating god nodes: %s", e)

        try:
            surprises = surprising_connections(undirected_simple, communities, top_n=top_surprises)
        except Exception as e:
            logger.warning("Error calculating surprising connections: %s", e)

    # Fallback to built-in if graphify was unavailable or returned empty
    if not hubs:
        raw_hubs = _builtin_god_nodes(undirected_simple, top_n=top_god_nodes)
        for h in raw_hubs:
            nid = h.get("id")
            node_data = G.nodes.get(nid, {})
            hubs.append({
                "id": nid,
                "label": h.get("label", nid),
                "degree": h.get("degree", 0),
                "entity_type": node_data.get("entity_type", "Entity"),
                "community": node_data.get("community", 0),
                "community_label": node_data.get("community_label", "Unknown")
            })

    if not surprises:
        surprises = _builtin_surprises(undirected_simple, communities, top_n=top_surprises)

    # 5. Build structured community summaries for API and Dashboard consumption
    community_summary = []
    for cid, members in communities.items():
        c_name = community_labels.get(cid, f"Domain_{cid}")
        entity_types = {}
        for m in members:
            e_type = G.nodes[m].get("entity_type", "Entity")
            entity_types[e_type] = entity_types.get(e_type, 0) + 1

        community_summary.append({
            "community_id": cid,
            "label": c_name,
            "member_count": len(members),
            "cohesion": cohesion_scores.get(cid, 0.0),
            "entity_breakdown": entity_types,
            "members": members[:20]  # sample of first 20 members
        })

    analysis_metadata = {
        "total_communities": len(communities),
        "cohesion_scores": cohesion_scores,
        "god_nodes": hubs,
        "surprising_connections": surprises,
        "community_summary": community_summary
    }

    logger.info(
        "Successfully clustered graph for task into %d communities, %d god nodes, %d surprising connections.",
        len(communities), len(hubs), len(surprises)
    )
    return communities, community_labels, analysis_metadata
