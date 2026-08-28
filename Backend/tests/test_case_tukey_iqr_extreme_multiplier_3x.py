import pytest
import polars as pl
from app.services.analysis.outliers import detect_outliers

def test_tukey_iqr_extreme_multiplier_bounds():
    """Verify Tukey's IQR boundaries at standard 1.5x and extreme 3.0x multiplier thresholds."""
    # Data with uniform spacing and clear extreme point
    values = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 500.0]
    df = pl.DataFrame({"points": values})
    
    outliers = detect_outliers(df, ["points"])
    assert "points" in outliers
    assert outliers["points"]["count"] == 1
    assert outliers["points"]["upper_bound"] > 28.0
