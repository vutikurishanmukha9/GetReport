import pytest
import polars as pl
from app.services.data_processing import join_datasets

def test_polars_left_join_preserves_unmatched_rows():
    """Verify join_datasets with left join preserves unmatched left rows and fills nulls."""
    df1 = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "Diana"]
    })
    
    df2 = pl.DataFrame({
        "id": [1, 2],
        "department": ["Engineering", "Sales"]
    })
    
    joined = join_datasets({"df1": df1, "df2": df2}, join_key="id", how="left")
    assert joined.height == 4
    assert "department" in joined.columns
    # Rows 3 and 4 must have null for department
    assert joined["department"].null_count() == 2
