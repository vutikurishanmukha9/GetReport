import pytest
import polars as pl
from app.services.data_processing import join_datasets

def test_polars_join_multi_key_and_column_name_collisions():
    """Verify join_datasets handles duplicate non-key column names with suffix disambiguation."""
    df1 = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [85.0, 90.0, 95.0]
    })
    
    df2 = pl.DataFrame({
        "id": [1, 2, 3],
        "city": ["New York", "London", "Tokyo"],
        "score": [100.0, 200.0, 300.0]
    })
    
    joined = join_datasets({"df1": df1, "df2": df2}, join_key="id", how="inner")
    
    assert joined.height == 3
    assert "id" in joined.columns
    assert "name" in joined.columns
    assert "city" in joined.columns
    assert "score_2" in joined.columns # Disambiguated
