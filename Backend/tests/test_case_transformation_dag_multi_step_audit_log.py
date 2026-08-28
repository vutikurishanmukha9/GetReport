import pytest
import polars as pl
from app.services.transformation_dag import TransformationDAG

def test_transformation_dag_multi_step_audit_logging():
    """Verify DAG maintains a multi-step transformation sequence with sequential linkage."""
    df0 = pl.DataFrame({"x": [1, 2, None, 4], "y": ["A", "B", "C", "D"]})
    df1 = df0.with_columns(pl.col("x").fill_null(0))
    df2 = df1.rename({"x": "x_clean", "y": "y_clean"})
    
    dag = TransformationDAG(dataset_name="multi_step_run")
    n1 = dag.add_node("fill_null_value", df0, df1, target_column="x", parameters={"val": 0})
    n2 = dag.add_node("rename_columns", df1, df2, parameters={"mapping": {"x": "x_clean"}})
    
    assert len(dag.nodes) == 2
    assert n2.parent_id == n1.id
    assert n1.child_id == n2.id
    
    summary = dag.get_summary()
    assert summary["total_steps"] == 2
    assert "fill_null_value" in summary["operations"]
    assert "rename_columns" in summary["operations"]
