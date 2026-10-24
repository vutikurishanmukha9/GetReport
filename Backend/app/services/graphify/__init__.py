"""
Graphify Integration Service Package for GetReport.
Exposes deterministic graph clustering, topological analysis,
sub-millisecond trigram search, causal path traversal, and multi-format exporters.
"""
import sys
from pathlib import Path

# Ensure graphify package directory in workspace root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_GRAPHIFY_PATH = _REPO_ROOT / "graphify"
if _GRAPHIFY_PATH.exists() and str(_GRAPHIFY_PATH) not in sys.path:
    sys.path.insert(0, str(_GRAPHIFY_PATH))

from app.services.graphify.cluster_bridge import cluster_and_analyze_dataset_graph
from app.services.graphify.query_bridge import find_causal_path, compute_blast_radius, search_graph_entities
from app.services.graphify.export_bridge import generate_interactive_html, generate_typst_vector_svg
from app.services.graphify.sql_lineage_bridge import extract_sql_lineage, attach_sql_lineage_to_dataset_graph

__all__ = [
    "cluster_and_analyze_dataset_graph",
    "find_causal_path",
    "compute_blast_radius",
    "search_graph_entities",
    "generate_interactive_html",
    "generate_typst_vector_svg",
    "extract_sql_lineage",
    "attach_sql_lineage_to_dataset_graph",
]
