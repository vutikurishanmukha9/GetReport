import pytest
import polars as pl
from app.services.analysis.outliers import detect_outliers

def test_outlier_detection_empty_numeric_list():
    """Verify detect_outliers returns empty dict when no numeric columns are provided."""
    df = pl.DataFrame({"category": ["A", "B", "C"]})
    outliers = detect_outliers(df, [])
    assert outliers == {}
