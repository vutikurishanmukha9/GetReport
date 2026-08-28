import pytest
import polars as pl
from app.services.analysis.statistics import compute_correlation

def test_pearson_correlation_zero_variance_handling():
    """Verify that Pearson correlation gracefully handles constant zero-variance columns without crashing."""
    df = pl.DataFrame({
        "constant_col": [5.0, 5.0, 5.0, 5.0, 5.0],
        "varying_col": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    
    corr_matrix, strong_pairs = compute_correlation(df, ["constant_col", "varying_col"])
    assert isinstance(corr_matrix, dict)
    assert isinstance(strong_pairs, list)
    # Zero variance must not produce bogus 1.0 correlation pairs
    assert len(strong_pairs) == 0
