import os
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
try:
    import networkx as nx
    from networkx.readwrite import json_graph
    HAS_NETWORKX = True
except ImportError:
    nx = None
    json_graph = None
    HAS_NETWORKX = False

logger = logging.getLogger(__name__)

class DatasetGraphStore:
    """
    LightRAG-inspired In-Process Knowledge Graph Store.
    Encapsulates a directed multi-graph (MultiDiGraph) representing the semantic
    topology of an ingested dataset:
    - Nodes: Datasets, FeatureColumns, DataQualityIssues, MetricSummaries, CleaningRules.
    - Edges: CONTAINS_COLUMN, HAS_QUALITY_ISSUE, CORRELATED_WITH, TRANSFORMED_BY, AFFECTS_CONFIDENCE.
    
    Supports:
    1. Local Retrieval: 1-hop and 2-hop neighborhood expansion for specific entities (low-level keywords).
    2. Global Retrieval: Thematic high-level relationship aggregation (high-level keywords).
    3. Fast Disk Caching: Serialization to/from JSON in temp_cache.
    """
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.graph = nx.MultiDiGraph() if HAS_NETWORKX else None

    def add_entity(self, entity_id: str, entity_type: str, attributes: Optional[Dict[str, Any]] = None):
        """Add or update an entity node in the graph."""
        attrs = attributes or {}
        self.graph.add_node(
            str(entity_id),
            entity_type=entity_type,
            **attrs
        )

    def add_relation(
        self,
        src_id: str,
        tgt_id: str,
        relation_type: str,
        attributes: Optional[Dict[str, Any]] = None,
        key: Optional[str] = None
    ):
        """Add a directed relationship edge between two entities."""
        attrs = attributes or {}
        # Ensure endpoints exist
        if not self.graph.has_node(str(src_id)):
            self.add_entity(str(src_id), entity_type="Unknown")
        if not self.graph.has_node(str(tgt_id)):
            self.add_entity(str(tgt_id), entity_type="Unknown")

        edge_key = key or f"{relation_type}:{src_id}->{tgt_id}"
        self.graph.add_edge(
            str(src_id),
            str(tgt_id),
            key=edge_key,
            relation_type=relation_type,
            **attrs
        )

    def get_local_subgraph(self, entity_ids: List[str], max_hops: int = 1) -> Dict[str, Any]:
        """
        LightRAG Local Retrieval Algorithm:
        Extracts target entity nodes, their 1-hop or 2-hop neighbors (both incoming and outgoing),
        and all connecting edges.
        """
        selected_nodes: Set[str] = set()
        clean_ids = [str(eid) for eid in entity_ids if self.graph.has_node(str(eid))]
        
        if not clean_ids:
            return {"nodes": [], "edges": []}

        for root_id in clean_ids:
            selected_nodes.add(root_id)
            # Undirected view for bidirectional neighborhood discovery
            undirected_view = self.graph.to_undirected(as_view=True)
            try:
                neighbors = nx.single_source_shortest_path_length(undirected_view, root_id, cutoff=max_hops)
                selected_nodes.update(neighbors.keys())
            except Exception as e:
                logger.warning(f"Error computing local subgraph neighborhood for {root_id}: {e}")

        # Extract induced subgraph
        subgraph = self.graph.subgraph(selected_nodes)
        
        nodes_data = []
        for n, d in subgraph.nodes(data=True):
            nodes_data.append({"id": n, **d})
            
        edges_data = []
        for u, v, k, d in subgraph.edges(keys=True, data=True):
            edges_data.append({
                "source": u,
                "target": v,
                "key": k,
                **d
            })

        return {"nodes": nodes_data, "edges": edges_data}

    def get_global_relations(self, high_level_keywords: List[str]) -> List[Dict[str, Any]]:
        """
        LightRAG Global Retrieval Algorithm:
        Finds relationship edges matching high-level thematic keywords (e.g. correlations,
        critical quality bottlenecks, confidence score drivers).
        """
        if not high_level_keywords:
            return []

        clean_kws = [k.lower() for k in high_level_keywords if k and len(k) > 1]
        matched_edges = []
        seen_keys = set()

        for u, v, k, d in self.graph.edges(keys=True, data=True):
            rel_type = str(d.get("relation_type", "")).lower()
            desc = str(d.get("description", "")).lower()
            edge_blob = f"{rel_type} {desc}"

            # Check if any keyword matches the edge metadata
            if any(kw in edge_blob for kw in clean_kws):
                edge_id = f"{u}->{v}:{k}"
                if edge_id not in seen_keys:
                    seen_keys.add(edge_id)
                    src_data = self.graph.nodes.get(u, {})
                    tgt_data = self.graph.nodes.get(v, {})
                    matched_edges.append({
                        "source": u,
                        "source_name": src_data.get("name", u),
                        "source_type": src_data.get("entity_type", "Entity"),
                        "target": v,
                        "target_name": tgt_data.get("name", v),
                        "target_type": tgt_data.get("entity_type", "Entity"),
                        "relation_type": d.get("relation_type", "RELATED_TO"),
                        "description": d.get("description", ""),
                        "attributes": {k_attr: v_attr for k_attr, v_attr in d.items() if k_attr not in ("relation_type", "description")}
                    })

        return matched_edges

    def format_subgraph_markdown(self, subgraph: Dict[str, Any]) -> str:
        """Format local subgraph into concise, grounded markdown context for the LLM."""
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        if not nodes and not edges:
            return ""

        lines = ["#### Knowledge Graph Local Entity Topology"]
        lines.append("**Connected Entities & Attributes:**")
        for n in nodes[:15]:
            nid = n.get("id")
            ntype = n.get("entity_type", "Entity")
            name = n.get("name", nid)
            # Pick primary attributes
            stat_parts = []
            for k, val in n.items():
                if k not in ("id", "entity_type", "name") and val is not None:
                    stat_parts.append(f"{k}={val}")
            attr_str = f" ({', '.join(stat_parts[:4])})" if stat_parts else ""
            lines.append(f"- **[{ntype}] {name}**{attr_str}")

        if edges:
            lines.append("\n**Relational Causal Paths:**")
            for e in edges[:15]:
                src = e.get("source")
                tgt = e.get("target")
                rel = e.get("relation_type", "CONNECTED_TO")
                desc = e.get("description", "")
                r_val = e.get("r_value")
                extra = f" (r={r_val:.2f})" if r_val is not None else ""
                lines.append(f"- `{src}` --[{rel}{extra}]--> `{tgt}`: {desc}")

        return "\n".join(lines)

    def format_global_relations_markdown(self, relations: List[Dict[str, Any]]) -> str:
        """Format global relationships into thematic summary markdown."""
        if not relations:
            return ""

        lines = ["#### Knowledge Graph Global Thematic Relationships"]
        for r in relations[:12]:
            s_name = r.get("source_name", r.get("source"))
            t_name = r.get("target_name", r.get("target"))
            rel = r.get("relation_type", "RELATION")
            desc = r.get("description", "")
            lines.append(f"- **{s_name}** ↔ **{t_name}** [{rel}]: {desc}")

        return "\n".join(lines)

    def cluster(self, resolution: float = 1.0) -> Dict[str, Any]:
        """
        Executes Graphify Leiden clustering, hub naming, cohesion scoring,
        God Node identification, and surprising connection discovery.
        """
        if not HAS_NETWORKX or self.graph is None or self.graph.number_of_nodes() == 0:
            return {}

        from app.services.graphify.cluster_bridge import cluster_and_analyze_dataset_graph
        communities, community_labels, analysis = cluster_and_analyze_dataset_graph(
            self.graph, resolution=resolution
        )
        self.graph.graph["communities"] = communities
        self.graph.graph["community_labels"] = community_labels
        self.graph.graph["analysis_metadata"] = analysis
        return analysis

    def get_analysis(self) -> Dict[str, Any]:
        """Returns topological analysis metadata if available."""
        if not HAS_NETWORKX or self.graph is None:
            return {}
        return self.graph.graph.get("analysis_metadata", {})

    def get_communities(self) -> Dict[int, List[str]]:
        """Returns detected community clusters."""
        if not HAS_NETWORKX or self.graph is None:
            return {}
        return self.graph.graph.get("communities", {})

    def get_community_labels(self) -> Dict[int, str]:
        """Returns community hub labels."""
        if not HAS_NETWORKX or self.graph is None:
            return {}
        return self.graph.graph.get("community_labels", {})

    def find_path(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """Finds shortest and causal paths between two entities."""
        from app.services.graphify.query_bridge import find_causal_path
        return find_causal_path(self.graph, source_id, target_id)

    def get_blast_radius(self, node_id: str) -> Dict[str, Any]:
        """Calculates upstream dependencies and downstream blast radius for a node."""
        from app.services.graphify.query_bridge import compute_blast_radius
        return compute_blast_radius(self.graph, node_id)

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Sub-millisecond fuzzy entity search via trigrams and token scoring."""
        from app.services.graphify.query_bridge import search_graph_entities
        return search_graph_entities(self.graph, query, limit=limit)

    def generate_html(self, output_html_path: str) -> str:
        """Generates a standalone, self-contained interactive Vis.js HTML view."""
        from app.services.graphify.export_bridge import generate_interactive_html
        comms = self.get_communities()
        return generate_interactive_html(self.graph, comms, output_html_path)

    def generate_svg(self, output_svg_path: str) -> str:
        """Generates a vector SVG for Typst PDF embedding."""
        from app.services.graphify.export_bridge import generate_typst_vector_svg
        comms = self.get_communities()
        labels = self.get_community_labels()
        return generate_typst_vector_svg(self.graph, output_svg_path, comms, labels)

    def attach_sql(self, sql_text: str, dialect: str = "snowflake") -> Dict[str, Any]:
        """Parses a SQL query and bridges its table/view lineage into this graph."""
        from app.services.graphify.sql_lineage_bridge import attach_sql_lineage_to_dataset_graph
        return attach_sql_lineage_to_dataset_graph(self.graph, sql_text, dialect=dialect)

    def get_layers(self) -> List[Dict[str, Any]]:
        """Classifies the graph into Understand-Anything canonical architectural layers."""
        from app.services.understand_anything.layer_detector import detect_dataset_layers
        return detect_dataset_layers(self.graph)

    def get_audit_tour(self, analysis_results: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generates an ordered, pedagogical investigation tour of dataset bottlenecks."""
        from app.services.understand_anything.tour_generator import generate_dataset_audit_tour
        return generate_dataset_audit_tour(self.graph, analysis_results=analysis_results)

    def get_fingerprint(self) -> Dict[str, Any]:
        """Computes a deterministic structural SHA-256 fingerprint for drift detection."""
        from app.services.understand_anything.fingerprint import compute_dataset_fingerprint
        return compute_dataset_fingerprint(self.graph)

    def to_understand_anything_json(self, project_name: str = "Dataset Audit", description: str = "") -> Dict[str, Any]:
        """Exports the graph in full Understand-Anything KnowledgeGraph schema format."""
        from app.services.understand_anything.exporter import export_to_understand_anything_schema
        return export_to_understand_anything_schema(self.graph, project_name=project_name, description=description)

    def to_json(self) -> str:
        """Serialize the graph to a JSON string compatible with networkx json_graph."""
        if not HAS_NETWORKX or self.graph is None:
            return "{}"
        data = json_graph.node_link_data(self.graph)
        return json.dumps(data)

    @classmethod
    def from_json(cls, task_id: str, json_str: str) -> 'DatasetGraphStore':
        """Construct a DatasetGraphStore instance from a serialized JSON string."""
        store = cls(task_id)
        if not HAS_NETWORKX or not json_str:
            return store
        data = json.loads(json_str)
        store.graph = json_graph.node_link_graph(data, directed=True, multigraph=True)
        return store

    def save_to_file(self, file_path: str):
        """Save graph to disk with directory creation."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        logger.info(f"Saved dataset knowledge graph to {file_path} ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)")

    @classmethod
    def load_from_file(cls, task_id: str, file_path: str) -> Optional['DatasetGraphStore']:
        """Load graph from disk if it exists."""
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return cls.from_json(task_id, content)
        except Exception as e:
            logger.error(f"Failed to load dataset knowledge graph from {file_path}: {e}")
            return None
