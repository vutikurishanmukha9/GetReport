"""
SQL Lineage Bridge for GetReport.
Uses sqlglot to extract table, view, CTE, and join lineage from SQL queries
(Snowflake, DuckDB, Postgres dialects) and integrate them into Graphify knowledge graphs.
"""
from __future__ import annotations
import hashlib
import logging
from typing import Dict, List, Any, Set, Tuple, Optional
import networkx as nx
import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


def extract_sql_lineage(
    sql_text: str,
    dialect: str = "snowflake"
) -> Dict[str, Any]:
    """
    Parses a SQL statement using sqlglot and extracts tables, views, CTEs,
    and directional dependency edges.
    
    Returns:
        Dict with "nodes" and "edges" ready for Graphify network integration.
    """
    if not sql_text or not sql_text.strip():
        return {"nodes": [], "edges": []}

    read_tables: Set[str] = set()
    write_tables: Set[str] = set()
    joined_tables: Set[Tuple[str, str]] = set()
    ctes: Set[str] = set()

    try:
        # Parse expression tree with fallback dialect
        parsed_expressions = sqlglot.parse(sql_text, read=dialect)
    except Exception as parse_err:
        logger.debug("Failed parsing with dialect '%s', falling back to default: %s", dialect, parse_err)
        try:
            parsed_expressions = sqlglot.parse(sql_text)
        except Exception as e:
            logger.warning("SQLGlot parse failure: %s", e)
            return {"nodes": [], "edges": [], "error": str(e)}

    for expression in parsed_expressions:
        if not expression:
            continue

        # 1. Identify Target / Write Tables (CREATE TABLE, CREATE VIEW, INSERT INTO)
        if isinstance(expression, (exp.Create, exp.Insert)):
            target = expression.this
            if target:
                t_name = str(target.name if hasattr(target, "name") else target).lower()
                write_tables.add(t_name)

        # 2. Extract CTE definitions
        for cte in expression.find_all(exp.CTE):
            cte_alias = cte.alias
            if cte_alias:
                ctes.add(str(cte_alias).lower())

        # 3. Extract Read / Source Tables
        for table in expression.find_all(exp.Table):
            t_name = table.name.lower()
            # Exclude CTE aliases from external tables
            if t_name not in ctes and t_name not in write_tables:
                read_tables.add(t_name)

        # 4. Extract Explicit Join Pairs
        for join in expression.find_all(exp.Join):
            join_table = join.this
            if isinstance(join_table, exp.Table):
                jt_name = join_table.name.lower()
                for rt in read_tables:
                    if rt != jt_name:
                        joined_tables.add((rt, jt_name))

    # Format into Graphify node/edge structures
    nodes = []
    edges = []

    for t in read_tables:
        nodes.append({
            "id": f"table:{t}",
            "label": t,
            "entity_type": "SQLSourceTable",
            "source_file": "schema.sql"
        })

    for wt in write_tables:
        nodes.append({
            "id": f"view:{wt}",
            "label": wt,
            "entity_type": "SQLTargetView",
            "source_file": "schema.sql"
        })

    for cte in ctes:
        nodes.append({
            "id": f"cte:{cte}",
            "label": cte,
            "entity_type": "SQLCTE",
            "source_file": "schema.sql"
        })

    # Lineage edges: Source Tables -> Target View
    for src in read_tables:
        for tgt in write_tables:
            edges.append({
                "source": f"table:{src}",
                "target": f"view:{tgt}",
                "relation": "FEEDS_INTO",
                "relation_type": "FEEDS_INTO",
                "confidence": "EXTRACTED",
                "weight": 2.0
            })

    # Join edges: Table A <-> Table B
    for t1, t2 in joined_tables:
        edges.append({
            "source": f"table:{t1}",
            "target": f"table:{t2}",
            "relation": "JOINS_WITH",
            "relation_type": "JOINS_WITH",
            "confidence": "EXTRACTED",
            "weight": 1.5
        })

    return {
        "read_tables": list(read_tables),
        "write_tables": list(write_tables),
        "ctes": list(ctes),
        "nodes": nodes,
        "edges": edges
    }


def attach_sql_lineage_to_dataset_graph(
    G: nx.MultiDiGraph,
    sql_text: str,
    dialect: str = "snowflake"
) -> Dict[str, Any]:
    """
    Parses SQL and adds table lineage directly to an existing DatasetGraphStore MultiDiGraph.
    """
    lineage = extract_sql_lineage(sql_text, dialect=dialect)

    for n in lineage["nodes"]:
        nid = n["id"]
        attrs = {k: v for k, v in n.items() if k != "id"}
        G.add_node(nid, **attrs)

    for e in lineage["edges"]:
        u, v = e["source"], e["target"]
        attrs = {k: v for k, v in e.items() if k not in ("source", "target")}
        G.add_edge(u, v, **attrs)

    return lineage
