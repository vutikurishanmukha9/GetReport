import pytest
import polars as pl
from app.services.transformation_dag import TransformationNode

def test_transformation_node_serialization_roundtrip():
    """Verify TransformationNode serializes to dict and includes all audit fields."""
    node = TransformationNode(
        id="test_node_01",
        operation="fill_null_median",
        target_column="age",
        parameters={"median": 35.0},
        duration_ms=4.5,
        input_rows=100,
        input_cols=5,
        output_rows=100,
        output_cols=5,
        input_hash="hash_a",
        output_hash="hash_b",
        rows_affected=0,
        values_changed=5,
        reversibility="none"
    )
    
    d = node.to_dict()
    assert d["id"] == "test_node_01"
    assert d["operation"] == "fill_null_median"
    assert d["target_column"] == "age"
    assert d["duration_ms"] == 4.5
    assert d["input_state"]["rows"] == 100
    assert d["output_state"]["rows"] == 100
    assert d["impact"]["values_changed"] == 5
