import pytest
import polars as pl
from app.services.transformation_dag import TransformationDAG

def test_transformation_dag_node_creation_and_lineage():
    """Verify that TransformationDAG registers nodes with proper state hashes and parent-child links."""
    df_initial = pl.DataFrame({
        "name": [" Alice ", "Bob", "Charlie ", None],
        "age": [25, None, 30, 45],
        "salary": [50000.0, 60000.0, 75000.0, 120000.0]
    })
    
    dag = TransformationDAG(dataset_name="test_dataset_001")
    assert dag.dataset_name == "test_dataset_001"
    assert len(dag.nodes) == 0

    # Step 1: Impute age
    median_age = float(df_initial["age"].drop_nulls().median())
    df_step1 = df_initial.with_columns(pl.col("age").fill_null(median_age))
    
    node1 = dag.add_node(
        operation="fill_null_median",
        df_before=df_initial,
        df_after=df_step1,
        target_column="age",
        parameters={"fill_value": median_age},
        duration_ms=12.5,
        values_changed=1
    )
    
    assert len(dag.nodes) == 1
    assert node1.operation == "fill_null_median"
    assert node1.target_column == "age"
    assert node1.input_rows == 4
    assert node1.output_rows == 4
    assert node1.parent_id is None

    # Step 2: Drop null rows
    df_step2 = df_step1.drop_nulls(subset=["name"])
    node2 = dag.add_node(
        operation="drop_null_rows",
        df_before=df_step1,
        df_after=df_step2,
        target_column="name",
        duration_ms=8.0,
    )
    
    assert len(dag.nodes) == 2
    assert node2.parent_id == node1.id
    assert node1.child_id == node2.id
    assert node2.input_rows == 4
    assert node2.output_rows == 3
