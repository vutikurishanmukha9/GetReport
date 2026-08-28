import pytest
import polars as pl
from app.services.transformation_dag import TransformationDAG

def test_transformation_dag_locked_error_enforcement():
    """Verify that adding transformations to a locked DAG raises ValueError."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    dag = TransformationDAG(dataset_name="locked_dag")
    dag.lock()
    
    with pytest.raises(ValueError, match="DAG is locked"):
        dag.add_node("drop_null_rows", df, df)
