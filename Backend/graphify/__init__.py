"""
Graphify - Deterministic Knowledge Graph Analytics and Clustering Engine for GetReport.
Provides community detection (Louvain/Leiden), topological analysis (god nodes, surprising connections),
and multi-format graph exporters (Vis.js HTML, vector SVG, Neo4j Cypher, GraphML, JSON).
"""
from __future__ import annotations

__version__ = "0.2.0"

from graphify.cluster import cluster, label_communities_by_hub, cohesion_score
from graphify.analyze import god_nodes, surprising_connections
from graphify.export import to_svg, to_cypher, to_graphml, to_json
from graphify.exporters.html import to_html

__all__ = [
    "cluster",
    "label_communities_by_hub",
    "cohesion_score",
    "god_nodes",
    "surprising_connections",
    "to_svg",
    "to_cypher",
    "to_graphml",
    "to_json",
    "to_html",
]
