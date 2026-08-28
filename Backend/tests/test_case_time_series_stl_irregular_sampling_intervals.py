import pytest
import polars as pl
from app.services.analysis.outliers import detect_time_series_stl_outliers

def test_time_series_stl_irregular_intervals():
    """Verify that irregular datetime sampling intervals with sufficient observations can be analyzed."""
    # 20 points spaced irregularly
    dates = [f"2024-01-{i:02d}" for i in range(1, 21)]
    values = [float(50 + (i % 7) * 5) for i in range(20)]
    
    df = pl.DataFrame({
        "timestamp": dates,
        "sales": values
    })
    
    res = detect_time_series_stl_outliers(df, "timestamp", "sales")
    assert res["total_observations"] == 20
    assert "has_stl_outliers" in res
