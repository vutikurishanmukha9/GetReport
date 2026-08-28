import pytest
import polars as pl
from app.services.analysis.classification import classify_numeric_columns

def test_classification_detects_sequential_auto_increment_ids():
    """Verify classify_numeric_columns identifies sequential integer sequences as ID-like and excludes them from analytical metrics."""
    df = pl.DataFrame({
        "unknown_seq": list(range(1, 101)), # Sequential 1..100
        "sales_amount": [150.0 + i * 2.5 for i in range(100)]
    })
    
    res = classify_numeric_columns(df, ["unknown_seq", "sales_amount"])
    assert "unknown_seq" in res["excluded"]
    assert "strictly_sequential_auto_increment" in res["exclusion_reasons"]["unknown_seq"]
    assert "sales_amount" in res["analytical"]
