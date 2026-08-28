import pytest
import polars as pl
from app.services.confidence_scoring import calculate_confidence_scores

def test_confidence_scoring_dirty_dataset_penalties():
    """Verify confidence scoring heavily penalizes mostly-null and constant columns."""
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "mostly_null": [10.0, None, None, None, None],
        "outlier_col": [10, 12, 11, 10, 50000],
        "constant_col": [1, 1, 1, 1, 1]
    })
    
    report = calculate_confidence_scores(df)
    assert report.dataset_confidence < 85.0
    
    null_col = next(c for c in report.columns if c.column == "mostly_null")
    assert null_col.completeness == 20.0
    
    const_col = next(c for c in report.columns if c.column == "constant_col")
    assert const_col.stability <= 50.0
