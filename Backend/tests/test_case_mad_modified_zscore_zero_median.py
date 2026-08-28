import pytest
import polars as pl
import numpy as np
from app.services.analysis.outliers import detect_outliers

def test_mad_modified_zscore_zero_median_fallback():
    """Verify that MAD modified Z-score does not divide by zero when median is 0."""
    # Data with median = 0, but distinct outliers
    values = [0.0] * 50 + [0.0, 1.0, 0.0, 5000.0, -2500.0]
    df = pl.DataFrame({"metric_zero_median": values})
    
    outliers = detect_outliers(df, ["metric_zero_median"])
    assert "metric_zero_median" in outliers
    assert outliers["metric_zero_median"]["count"] >= 2
