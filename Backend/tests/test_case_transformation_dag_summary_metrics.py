import pytest
import polars as pl
from app.services.transformation_dag import TransformationDAG

def test_transformation_dag_summary_metrics_aggregation():
    """Verify total rows affected, values changed, and execution durations aggregate properly."""
    df0 = pl.DataFrame({"val": [10, 20, 30, None, 50]})
    df1 = df0.drop_nulls()
    
    dag = TransformationDAG(dataset_name="metrics_run")
    dag.add_node("drop_null_rows", df0, df1, target_column="val", duration_ms=15.0, values_changed=0)
    
    summary = dag.get_summary()
    assert summary["total_steps"] == 1
    assert summary["total_rows_affected"] == 1
    assert "drop_null_rows" in summary["operations"]
