import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from app.services.analysis.outliers import detect_time_series_stl_outliers

def test_time_series_stl_residual_outlier_detection():
    """Verify STL decomposition isolates seasonality and flags anomalies on residuals."""
    base_date = datetime(2024, 1, 1)
    dates = [(base_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(60)]
    
    # Trend + Weekly Seasonality (7-day periodic wave) + 1 giant anomaly at day 35
    series = [float(100 + i*2 + (15 if i % 7 in (5, 6) else 0)) for i in range(60)]
    series[35] = 950.0  # massive sudden anomaly
    
    df = pl.DataFrame({
        "timestamp": dates,
        "traffic": series
    })
    
    stl_res = detect_time_series_stl_outliers(df, "timestamp", "traffic")
    assert stl_res["has_stl_outliers"] is True
    assert stl_res["stl_outlier_count"] >= 1
    assert stl_res["total_observations"] == 60
