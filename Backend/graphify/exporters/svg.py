"""
Vector SVG Exporter for Graphify.
Generates publication-quality SVG diagrams of knowledge graphs
for embedding into Typst PDF reports and web previews.
"""
from __future__ import annotations
import os
import html
import math
from typing import Dict, List, Any, Optional
import networkx as nx

_PALETTE = [
    "#800000",  # Maroon
    "#2563eb",  # Blue
    "#059669",  # Green
    "#d97706",  # Amber
    "#7c3aed",  # Purple
    "#db2777",  # Pink
    "#0891b2",  # Cyan
    "#4f46e5",  # Indigo
]


def to_svg(
    G: nx.Graph,
    communities: Dict[int, List[str]],
    output_path: str,
    community_labels: Optional[Dict[int, str]] = None
) -> str:
    """
    Renders a standalone vector SVG diagram of the knowledge graph.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    width = 900
    height = 560
    margin = 60

    labels = community_labels or {}
    node_to_comm: Dict[str, int] = {}
    for cid, members in communities.items():
        for m in members:
            node_to_comm[m] = cid

    nodes = list(G.nodes())
    if not nodes:
        svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
    <rect width="100%" height="100%" fill="#0e1017" rx="12"/>
    <text x="{width//2}" y="{height//2}" fill="#64748b" font-family="sans-serif" font-size="14" text-anchor="middle">
        Empty Graph (0 Nodes)
    </text>
</svg>'''
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        return output_path

    # Layout computation
    if len(nodes) <= 2:
        pos = {nodes[0]: (0.3, 0.5)}
        if len(nodes) > 1:
            pos[nodes[1]] = (0.7, 0.5)
    else:
        try:
            pos = nx.spring_layout(G, seed=42, k=1.2 / math.sqrt(len(nodes)), iterations=50)
        except Exception:
            pos = nx.circular_layout(G)

    # Scale positions to canvas coordinates
    coords = {}
    for n, (x, y) in pos.items():
        cx = margin + (x + 1.0) / 2.0 * (width - 2 * margin)
        cy = margin + (y + 1.0) / 2.0 * (height - 2 * margin)
        coords[n] = (round(cx, 1), round(cy, 1))

    # Build SVG elements
    svg_edges = []
    for u, v, d in G.edges(data=True):
        if u in coords and v in coords:
            x1, y1 = coords[u]
            x2, y2 = coords[v]
            weight = float(d.get("weight", 1.0))
            stroke_w = max(1.0, min(3.0, weight))
            svg_edges.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#334155" stroke-width="{stroke_w}" stroke-opacity="0.6"/>'
            )

    svg_nodes = []
    for n in nodes:
        if n in coords:
            cx, cy = coords[n]
            cid = node_to_comm.get(n, 0)
            color = _PALETTE[cid % len(_PALETTE)]
            deg = G.degree(n)
            r = max(6, min(16, 6 + deg * 1.5))
            node_data = G.nodes.get(n, {})
            label = str(node_data.get("name") or node_data.get("label") or n)
            escaped_label = html.escape(label[:18])

            svg_nodes.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
            )
            svg_nodes.append(
                f'<text x="{cx}" y="{cy + r + 13}" fill="#cbd5e1" font-family="system-ui, sans-serif" font-size="10" text-anchor="middle">{escaped_label}</text>'
            )

    # Build Legend
    svg_legend = []
    legend_x = 24
    legend_y = 30
    svg_legend.append(f'<g transform="translate({legend_x}, {legend_y})">')
    for cid, c_nodes in list(communities.items())[:5]:
        color = _PALETTE[cid % len(_PALETTE)]
        c_label = html.escape(labels.get(cid, f"Cluster {cid}"))
        item_y = cid * 18
        svg_legend.append(
            f'<rect x="0" y="{item_y}" width="10" height="10" rx="3" fill="{color}"/>'
            f'<text x="16" y="{item_y + 9}" fill="#94a3b8" font-family="system-ui, sans-serif" font-size="11">{c_label} ({len(c_nodes)})</text>'
        )
    svg_legend.append('</g>')

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
    <rect width="100%" height="100%" fill="#0a0c14" rx="14"/>
    <g class="edges">
        {''.join(svg_edges)}
    </g>
    <g class="nodes">
        {''.join(svg_nodes)}
    </g>
    {''.join(svg_legend)}
</svg>'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    return output_path
