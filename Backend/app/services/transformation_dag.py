"""
Transformation DAG - Audit-grade transformation tracking.

This module implements a Directed Acyclic Graph (DAG) to track every data
transformation applied during the cleaning pipeline. Each node represents
a single operation with full input/output state tracking.

Tier 3: Audit Gold - Every action traceable and reversible.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
import polars as pl


# ─── Operation Types ─────────────────────────────────────────────────────────

OperationType = Literal[
    "rename_columns",
    "fill_null_mean",
    "fill_null_median", 
    "fill_null_mode",
    "fill_null_value",
    "drop_null_rows",
    "drop_empty_columns",
    "remove_duplicates",
    "replace_outliers",
    "type_conversion",
    "custom",
]

ReversibilityLevel = Literal[
    "full",      # Can be completely undone
    "partial",   # Can be undone with some data loss
    "none",      # Cannot be undone (destructive)
]


# ─── Helper Functions ────────────────────────────────────────────────────────

def _compute_data_hash(df: pl.DataFrame, sample_size: int = 100) -> str:
    """
    Compute a hash of the first N rows for state verification.
    Used to verify data hasn't changed between steps.
    """
    if df.height == 0:
        return "empty"
    
    sample = df.head(min(sample_size, df.height))
    # Convert to string representation for hashing
    data_str = sample.write_csv()
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]


def _get_reversibility(operation: str) -> tuple[ReversibilityLevel, str | None]:
    """
    Determine if an operation can be reversed and how.
    Returns (level, reverse_operation_hint).
    """
    reversibility_map = {
        "rename_columns": ("full", "Reverse column mapping available"),
        "fill_null_mean": ("none", "Original null positions lost"),
        "fill_null_median": ("none", "Original null positions lost"),
        "fill_null_mode": ("none", "Original null positions lost"),
        "fill_null_value": ("none", "Original null positions lost"),
        "drop_null_rows": ("partial", "Dropped row indices stored"),
        "drop_empty_columns": ("partial", "Column names stored"),
        "remove_duplicates": ("partial", "Duplicate indices stored"),
        "replace_outliers": ("none", "Original values lost"),
        "type_conversion": ("partial", "Original type stored"),
        "custom": ("none", "Custom operation - reversibility unknown"),
    }
    return reversibility_map.get(operation, ("none", None))


# ─── Transformation Node ─────────────────────────────────────────────────────

@dataclass
class TransformationNode:
    """
    A single transformation step in the DAG.
    
    Captures complete audit information for one cleaning operation.
    """
    # Identity
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    operation: OperationType = "custom"
    
    # Target
    target_column: str | None = None
    
    # Parameters (operation-specific)
    parameters: dict[str, Any] = field(default_factory=dict)
    
    # Timing
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: float = 0.0
    
    # State tracking
    input_rows: int = 0
    input_cols: int = 0
    output_rows: int = 0
    output_cols: int = 0
    input_hash: str = ""
    output_hash: str = ""
    
    # Impact metrics
    rows_affected: int = 0
    values_changed: int = 0
    
    # Reversibility
    reversibility: ReversibilityLevel = "none"
    reverse_hint: str | None = None
    stored_data: dict[str, Any] | None = None  # For partial reversibility
    
    # Graph links
    parent_id: str | None = None
    child_id: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "id": self.id,
            "operation": self.operation,
            "target_column": self.target_column,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 2),
            "input_state": {
                "rows": self.input_rows,
                "cols": self.input_cols,
                "hash": self.input_hash,
            },
            "output_state": {
                "rows": self.output_rows,
                "cols": self.output_cols,
                "hash": self.output_hash,
            },
            "impact": {
                "rows_affected": self.rows_affected,
                "values_changed": self.values_changed,
            },
            "reversibility": {
                "level": self.reversibility,
                "hint": self.reverse_hint,
            },
            "links": {
                "parent_id": self.parent_id,
                "child_id": self.child_id,
            },
        }
    
    def summary(self) -> str:
        """Human-readable summary of this transformation."""
        target = f"on '{self.target_column}'" if self.target_column else ""
        impact = f"{self.rows_affected} rows affected" if self.rows_affected else ""
        return f"{self.operation} {target} - {impact}".strip()


# ─── Transformation DAG ──────────────────────────────────────────────────────

@dataclass
class TransformationDAG:
    """
    Complete transformation history as a Directed Acyclic Graph.
    
    Provides full audit trail of all cleaning operations applied
    to a dataset, enabling compliance reporting and (partial) undo.
    """
    # Graph structure
    nodes: dict[str, TransformationNode] = field(default_factory=dict)
    root_node_id: str | None = None
    current_node_id: str | None = None
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_name: str = ""
    original_rows: int = 0
    original_cols: int = 0
    
    # Status
    locked: bool = False
    locked_at: str | None = None
    
    def add_node(
        self,
        operation: OperationType,
        df_before: pl.DataFrame,
        df_after: pl.DataFrame,
        target_column: str | None = None,
        parameters: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
        stored_data: dict[str, Any] | None = None,
        values_changed: int = 0,
    ) -> TransformationNode:
        """
        Add a new transformation node to the DAG.
        
        Args:
            operation: Type of operation performed
            df_before: DataFrame before transformation
            df_after: DataFrame after transformation
            target_column: Column affected (if applicable)
            parameters: Operation-specific parameters
            duration_ms: Time taken for this operation
            stored_data: Data for partial reversibility (e.g., dropped rows)
            values_changed: Number of values modified (in-place)
            
        Returns:
            The newly created TransformationNode
        """
        if self.locked:
            raise ValueError("DAG is locked - no more transformations allowed")
        
        # Get reversibility info
        reversibility, reverse_hint = _get_reversibility(operation)
        
        # Calculate impact
        rows_affected = abs(df_before.height - df_after.height)
        
        # Create node
        node = TransformationNode(
            operation=operation,
            target_column=target_column,
            parameters=parameters or {},
            duration_ms=duration_ms,
            input_rows=df_before.height,
            input_cols=df_before.width,
            output_rows=df_after.height,
            output_cols=df_after.width,
            input_hash=_compute_data_hash(df_before),
            output_hash=_compute_data_hash(df_after),
            rows_affected=rows_affected,
            values_changed=values_changed,  # Populated from arg
            reversibility=reversibility,
            reverse_hint=reverse_hint,
            stored_data=stored_data,
            parent_id=self.current_node_id,
        )
        
        # Link to previous node
        if self.current_node_id and self.current_node_id in self.nodes:
            self.nodes[self.current_node_id].child_id = node.id
        
        # Add to graph
        self.nodes[node.id] = node
        
        # Update pointers
        if self.root_node_id is None:
            self.root_node_id = node.id
        self.current_node_id = node.id
        
        return node
    
    def get_chain(self) -> list[TransformationNode]:
        """
        Get all nodes in order (from root to current).
        
        Returns:
            List of TransformationNode in execution order
        """
        if not self.root_node_id:
            return []
        
        chain = []
        current_id = self.root_node_id
        
        while current_id and current_id in self.nodes:
            node = self.nodes[current_id]
            chain.append(node)
            current_id = node.child_id
        
        return chain
    
    def get_summary(self) -> dict[str, Any]:
        """Get high-level summary of all transformations."""
        chain = self.get_chain()
        
        total_rows_affected = sum(n.rows_affected for n in chain)
        total_values_changed = sum(n.values_changed for n in chain)
        
        reversibility_counts = {"full": 0, "partial": 0, "none": 0}
        for node in chain:
            reversibility_counts[node.reversibility] += 1
        
        return {
            "total_steps": len(chain),
            "total_rows_affected": total_rows_affected,
            "total_values_changed": total_values_changed,
            "reversibility": reversibility_counts,
            "operations": [n.operation for n in chain],
        }
    
    def lock(self) -> None:
        """Lock the DAG to prevent further modifications."""
        self.locked = True
        self.locked_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "created_at": self.created_at,
            "dataset_name": self.dataset_name,
            "original_state": {
                "rows": self.original_rows,
                "cols": self.original_cols,
            },
            "locked": self.locked,
            "locked_at": self.locked_at,
            "summary": self.get_summary(),
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "root_node_id": self.root_node_id,
            "current_node_id": self.current_node_id,
        }
    
    def to_audit_log(self, include_hashes: bool = True) -> dict[str, Any]:
        """
        Export as ISO-compliant audit log.
        
        Returns:
            Dictionary suitable for compliance reporting
        """
        chain = self.get_chain()
        
        transformation_chain = []
        for i, node in enumerate(chain, 1):
            entry = {
                "step": i,
                "operation": node.operation,
                "target": node.target_column,
                "timestamp": node.timestamp,
                "duration_ms": node.duration_ms,
                "input_state": {
                    "rows": node.input_rows,
                    "columns": node.input_cols,
                },
                "output_state": {
                    "rows": node.output_rows,
                    "columns": node.output_cols,
                },
                "impact": {
                    "rows_affected": node.rows_affected,
                    "values_changed": node.values_changed,
                },
                "reversible": node.reversibility != "none",
                "parameters": node.parameters,
            }
            
            if include_hashes:
                entry["input_hash"] = node.input_hash
                entry["output_hash"] = node.output_hash
            
            transformation_chain.append(entry)
        
        return {
            "audit_version": "1.0",
            "dataset": self.dataset_name,
            "export_timestamp": datetime.now().isoformat(),
            "original_state": {
                "rows": self.original_rows,
                "columns": self.original_cols,
            },
            "transformation_chain": transformation_chain,
            "summary": self.get_summary(),
            "locked": self.locked,
            "locked_at": self.locked_at,
        }
    
    def to_audit_csv(self) -> str:
        """Export transformation chain as CSV for spreadsheet review."""
        import io
        
        chain = self.get_chain()
        rows = []
        
        for i, node in enumerate(chain, 1):
            rows.append({
                "Step": i,
                "Operation": node.operation,
                "Target Column": node.target_column or "N/A",
                "Timestamp": node.timestamp,
                "Duration (ms)": node.duration_ms,
                "Input Rows": node.input_rows,
                "Output Rows": node.output_rows,
                "Rows Affected": node.rows_affected,
                "Reversible": node.reversibility,
            })
        
        if not rows:
            return "No transformations recorded"
        
        # Create CSV
        df = pl.DataFrame(rows)
        return df.write_csv()


# ─── Factory Functions ───────────────────────────────────────────────────────

def create_dag(df: pl.DataFrame, dataset_name: str = "") -> TransformationDAG:
    """
    Create a new TransformationDAG initialized with original data state.
    
    Args:
        df: The original DataFrame before any transformations
        dataset_name: Name of the dataset for audit trail
        
    Returns:
        New TransformationDAG ready to track transformations
    """
    dag = TransformationDAG(
        dataset_name=dataset_name,
        original_rows=df.height,
        original_cols=df.width,
    )
    return dag


def from_dict(data: dict[str, Any]) -> TransformationDAG:
    """
    Reconstruct a TransformationDAG from its dictionary representation.
    """
    dag = TransformationDAG(
        created_at=data.get("created_at", ""),
        dataset_name=data.get("dataset_name", ""),
        original_rows=data.get("original_state", {}).get("rows", 0),
        original_cols=data.get("original_state", {}).get("cols", 0),
        locked=data.get("locked", False),
        locked_at=data.get("locked_at"),
        root_node_id=data.get("root_node_id"),
        current_node_id=data.get("current_node_id"),
    )
    
    # Reconstruct nodes
    for node_id, node_data in data.get("nodes", {}).items():
        node = TransformationNode(
            id=node_data.get("id", node_id),
            operation=node_data.get("operation", "custom"),
            target_column=node_data.get("target_column"),
            parameters=node_data.get("parameters", {}),
            timestamp=node_data.get("timestamp", ""),
            duration_ms=node_data.get("duration_ms", 0.0),
            input_rows=node_data.get("input_state", {}).get("rows", 0),
            input_cols=node_data.get("input_state", {}).get("cols", 0),
            output_rows=node_data.get("output_state", {}).get("rows", 0),
            output_cols=node_data.get("output_state", {}).get("cols", 0),
            input_hash=node_data.get("input_state", {}).get("hash", ""),
            output_hash=node_data.get("output_state", {}).get("hash", ""),
            rows_affected=node_data.get("impact", {}).get("rows_affected", 0),
            values_changed=node_data.get("impact", {}).get("values_changed", 0),
            reversibility=node_data.get("reversibility", {}).get("level", "none"),
            reverse_hint=node_data.get("reversibility", {}).get("hint"),
            parent_id=node_data.get("links", {}).get("parent_id"),
            child_id=node_data.get("links", {}).get("child_id"),
        )
        dag.nodes[node_id] = node
    
    return dag
