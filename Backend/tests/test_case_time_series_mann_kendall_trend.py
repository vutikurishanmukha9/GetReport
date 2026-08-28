import pytest
import numpy as np
import polars as pl
from app.services.analysis.time_series import mann_kendall_trend_test, detect_trend

def test_time_series_mann_kendall_non_parametric_trend():
    """Verify Mann-Kendall non-parametric test detects monotonic upward trend."""
    # Monotonically strictly increasing series
    series = np.array([10.0, 12.0, 15.0, 19.0, 24.0, 30.0, 37.0, 45.0, 54.0, 64.0])
    
    res = mann_kendall_trend_test(series)
    assert res["statistically_significant"] is True
    assert res["direction"] == "upward"
    assert res["kendall_tau"] == 1.0
    assert res["p_value"] < 0.05
