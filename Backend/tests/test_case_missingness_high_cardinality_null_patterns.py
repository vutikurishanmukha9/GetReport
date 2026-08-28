import pytest
import polars as pl
from app.services.analysis.missing import analyze_missing_patterns

def test_missingness_high_cardinality_patterns():
    """Verify missingness pattern extraction across multi-column simultaneous nulls."""
    df = pl.DataFrame({
        "col_a": [1.0, None, 3.0, None, 5.0, 6.0],
        "col_b": [None, 2.0, None, 4.0, None, 6.0],
        "col_c": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    })
    
    res = analyze_missing_patterns(df)
    assert res["has_missing"] is True
    assert res["columns_affected"] == 2
    assert "col_a" in res["column_details"]
    assert "col_b" in res["column_details"]
