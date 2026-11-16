"""
Interactive Vis.js HTML Exporter for Graphify.
Generates standalone, self-contained HTML network visualizations
with community cluster shading, search filtering, and node inspector panels.
"""
from __future__ import annotations
import os
import json
import html
from typing import Dict, List, Any, Optional
import networkx as nx

# Modern curated color palette for community clusters
_COMMUNITY_PALETTE = [
    "#800000",  # Maroon (Brand accent)
    "#2563eb",  # Blue
    "#059669",  # Emerald green
    "#d97706",  # Amber
    "#7c3aed",  # Purple
    "#db2777",  # Pink
    "#0891b2",  # Cyan
    "#4f46e5",  # Indigo
    "#ea580c",  # Orange
    "#65a30d",  # Lime
]


def to_html(
    G: nx.Graph,
    communities: Dict[int, List[str]],
    output_path: str,
    title: str = "GetReport Knowledge & Lineage Graph"
) -> str:
    """
    Renders an interactive Vis.js HTML document from a NetworkX graph.
    
    Args:
        G: NetworkX graph with node and edge attributes.
        communities: Mapping of community ID to member node IDs.
        output_path: Destination file path for the .html report.
        title: Title displayed on the header banner.
        
    Returns:
        The output file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Build node-to-community mapping
    node_to_comm = {}
    for cid, members in communities.items():
        for m in members:
            node_to_comm[m] = cid

    vis_nodes = []
    for n, d in G.nodes(data=True):
        comm_id = node_to_comm.get(n, 0)
        color = _COMMUNITY_PALETTE[comm_id % len(_COMMUNITY_PALETTE)]
        label = str(d.get("name") or d.get("label") or d.get("column_name") or n)
        entity_type = str(d.get("entity_type") or "Entity")
        deg = G.degree(n)

        # Scale node size with degree
        size = max(14, min(36, 12 + deg * 2))

        vis_nodes.append({
            "id": str(n),
            "label": label,
            "title": f"<b>{html.escape(label)}</b><br/>Type: {html.escape(entity_type)}<br/>Community: {comm_id}<br/>Degree: {deg}",
            "color": {
                "background": color,
                "border": "#ffffff",
                "highlight": {"background": "#ffffff", "border": color}
            },
            "font": {"color": "#e0e0e0", "size": 12, "face": "system-ui, sans-serif"},
            "size": size,
            "shape": "dot"
        })

    vis_edges = []
    for u, v, d in G.edges(data=True):
        weight = float(d.get("weight", 1.0))
        relation = str(d.get("relation") or d.get("relation_type") or "connected_to")
        vis_edges.append({
            "from": str(u),
            "to": str(v),
            "label": relation if len(relation) <= 16 else "",
            "title": f"Relation: {html.escape(relation)}<br/>Weight: {weight}",
            "arrows": "to",
            "color": {"color": "#4a4a68", "highlight": "#800000", "hover": "#800000"},
            "width": max(1, min(4, int(weight))),
            "font": {"size": 9, "color": "#888899", "align": "middle"}
        })

    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)
    escaped_title = html.escape(title)
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    comm_count = len(communities)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{escaped_title}</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: #090a0f;
            color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            overflow: hidden;
            height: 100vh;
            width: 100vw;
        }}
        #network {{
            width: 100%;
            height: 100%;
        }}
        #hud {{
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 100;
            background: rgba(18, 20, 29, 0.92);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            max-width: 380px;
        }}
        h1 {{
            font-size: 15px;
            font-weight: 600;
            letter-spacing: -0.01em;
            color: #ffffff;
            margin-bottom: 6px;
        }}
        .metrics {{
            font-size: 11px;
            color: #94a3b8;
            display: flex;
            gap: 12px;
            margin-bottom: 12px;
        }}
        .metric-pill {{
            background: rgba(255, 255, 255, 0.05);
            padding: 2px 8px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.04);
        }}
        .search-box {{
            width: 100%;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 6px;
            padding: 8px 12px;
            color: #fff;
            font-size: 12px;
            outline: none;
            transition: border-color 0.15s;
        }}
        .search-box:focus {{
            border-color: #800000;
        }}
    </style>
</head>
<body>
    <div id="hud">
        <h1>{escaped_title}</h1>
        <div class="metrics">
            <span class="metric-pill"><b>{node_count}</b> Nodes</span>
            <span class="metric-pill"><b>{edge_count}</b> Edges</span>
            <span class="metric-pill"><b>{comm_count}</b> Communities</span>
        </div>
        <input type="text" id="search" class="search-box" placeholder="Filter entities..." oninput="handleSearch(this.value)" />
    </div>
    <div id="network"></div>
    <script>
        const rawNodes = {nodes_json};
        const rawEdges = {edges_json};
        const nodesDataSet = new vis.DataSet(rawNodes);
        const edgesDataSet = new vis.DataSet(rawEdges);

        const container = document.getElementById('network');
        const data = {{ nodes: nodesDataSet, edges: edgesDataSet }};
        const options = {{
            nodes: {{
                shape: 'dot',
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                smooth: {{ type: 'continuous' }},
                arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }}
            }},
            physics: {{
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -38,
                    centralGravity: 0.008,
                    springLength: 95,
                    springConstant: 0.08
                }},
                stabilization: {{ iterations: 150 }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                hideEdgesOnDrag: true
            }}
        }};

        const network = new vis.Network(container, data, options);

        function handleSearch(query) {{
            const q = query.toLowerCase().trim();
            if (!q) {{
                nodesDataSet.update(rawNodes);
                return;
            }}
            const updates = rawNodes.map(n => ({{
                id: n.id,
                hidden: !n.label.toLowerCase().includes(q) && !n.id.toLowerCase().includes(q)
            }}));
            nodesDataSet.update(updates);
        }}
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
