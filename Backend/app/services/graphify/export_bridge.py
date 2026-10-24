"""
Graphify Export Bridge for GetReport.
Wraps Graphify's standalone Vis.js HTML exporter, vector SVG generator for Typst PDF,
and enterprise graph database formats (Neo4j Cypher, GraphML).
"""
from __future__ import annotations
import os
import logging
from typing import Dict, List, Any, Optional
import networkx as nx

try:
    from graphify.exporters.html import to_html
    from graphify.export import to_svg, to_cypher, to_graphml, to_json
    HAS_GRAPHIFY_EXPORTERS = True
except ImportError as err:
    logging.getLogger(__name__).warning("Failed to import graphify exporters: %s", err)
    HAS_GRAPHIFY_EXPORTERS = False

logger = logging.getLogger(__name__)


def _to_simple_export_graph(G: nx.MultiDiGraph) -> nx.Graph:
    """Converts a MultiDiGraph to simple nx.Graph required by Graphify exporters."""
    export_graph = nx.Graph()
    for n, d in G.nodes(data=True):
        attrs = dict(d)
        attrs.setdefault("label", attrs.get("name", str(n)))
        attrs.setdefault("source_file", attrs.get("filename", "dataset"))
        export_graph.add_node(n, **attrs)

    for u, v, k, d in G.edges(keys=True, data=True):
        attrs = dict(d)
        attrs.setdefault("relation", attrs.get("relation_type", "connected_to"))
        attrs.setdefault("confidence", "EXTRACTED")
        attrs.setdefault("weight", float(attrs.get("weight", 1.0)))
        export_graph.add_edge(u, v, **attrs)

    return export_graph


def generate_interactive_html(
    G: nx.MultiDiGraph,
    communities: Dict[int, List[str]],
    output_html_path: str,
    title: str = "GetReport Knowledge & Lineage Graph"
) -> str:
    """
    Generates a standalone, self-contained interactive Vis.js HTML report with
    convex hull community cluster shading, search filtering, and node inspection.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    export_graph = _to_simple_export_graph(G)

    # If no communities, put all in community 0
    if not communities:
        communities = {0: list(export_graph.nodes())}

    if HAS_GRAPHIFY_EXPORTERS:
        try:
            to_html(export_graph, communities, output_html_path)
            logger.info("Generated standalone interactive graph HTML at %s", output_html_path)
            return output_html_path
        except Exception as e:
            logger.warning("Graphify to_html failed: %s. Generating fallback HTML canvas.", e)

    # Fallback HTML generator in case to_html encounters any edge case
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background: #0f0f1a; color: #e0e0e0; font-family: -apple-system, sans-serif; }}
        #network {{ width: 100vw; height: 100vh; }}
        #header {{ position: absolute; top: 16px; left: 16px; z-index: 10; background: rgba(26,26,46,0.85); padding: 12px 18px; border-radius: 12px; border: 1px solid #2a2a4e; }}
        h1 {{ margin: 0; font-size: 16px; color: #fff; }}
        p {{ margin: 4px 0 0; font-size: 12px; color: #aaa; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{title}</h1>
        <p>{export_graph.number_of_nodes()} Nodes | {export_graph.number_of_edges()} Edges</p>
    </div>
    <div id="network"></div>
    <script>
        const nodes = new vis.DataSet({[{"id": str(n), "label": str(d.get("label", n)), "shape": "dot"} for n, d in export_graph.nodes(data=True)]});
        const edges = new vis.DataSet({[{"from": str(u), "to": str(v), "arrows": "to"} for u, v in export_graph.edges()]});
        new vis.Network(document.getElementById('network'), {{ nodes, edges }}, {{ physics: {{ solver: "forceAtlas2Based" }} }});
    </script>
</body>
</html>""")
    return output_html_path


def generate_typst_vector_svg(
    G: nx.MultiDiGraph,
    output_svg_path: str,
    communities: Optional[Dict[int, List[str]]] = None,
    community_labels: Optional[Dict[int, str]] = None
) -> str:
    """
    Generates a clean vector SVG representation of the graph for direct embedding
    into Typst PDF executive reports.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_svg_path)), exist_ok=True)
    export_graph = _to_simple_export_graph(G)

    comms = communities or {0: list(export_graph.nodes())}
    labels = community_labels or {0: "General_Domain"}

    if HAS_GRAPHIFY_EXPORTERS:
        try:
            to_svg(export_graph, comms, output_svg_path, community_labels=labels)
            logger.info("Generated Typst vector graph SVG at %s", output_svg_path)
            return output_svg_path
        except Exception as e:
            logger.warning("Graphify to_svg failed: %s. Generating basic SVG fallback.", e)

    # Basic vector SVG fallback
    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(f'''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400" viewBox="0 0 800 400">
    <rect width="100%" height="100%" fill="#1a1a2e" rx="12"/>
    <text x="400" y="200" fill="#ffffff" font-family="sans-serif" font-size="16" text-anchor="middle">
        Knowledge Graph ({export_graph.number_of_nodes()} Entities, {export_graph.number_of_edges()} Relations)
    </text>
</svg>''')
    return output_svg_path


def export_graph_to_format(
    G: nx.MultiDiGraph,
    format_type: str,
    output_path: str
) -> str:
    """Exports the graph to enterprise formats: cypher, graphml, json."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    export_graph = _to_simple_export_graph(G)

    if format_type == "cypher" and HAS_GRAPHIFY_EXPORTERS:
        to_cypher(export_graph, output_path)
    elif format_type == "graphml" and HAS_GRAPHIFY_EXPORTERS:
        to_graphml(export_graph, output_path)
    else:
        # Default JSON
        from networkx.readwrite import json_graph
        import json
        data = json_graph.node_link_data(export_graph)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return output_path
