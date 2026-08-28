import pytest
import polars as pl
from app.services.analysis.outliers import detect_time_series_stl_outliers

def test_time_series_stl_insufficient_periods_graceful_fallback():
    """Verify that STL decomposition requires >= 14 observations and returns clean fallback for short series."""
    df = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "metric": [10.0, 20.0, 15.0, 30.0]
    })
    
    res = detect_time_series_stl_outliers(df, "date", "metric")
    assert res["has_stl_outliers"] is False
    assert "Insufficient observations" in res.get("reason", "")
