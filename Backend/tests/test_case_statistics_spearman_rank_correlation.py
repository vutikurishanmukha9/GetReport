import pytest
import polars as pl
from app.services.analysis.statistics import compute_correlation

def test_statistics_spearman_rank_and_zero_variance_safety():
    """Verify compute_correlation computes Spearman rho on non-linear monotonic data and handles zero variance safely."""
    # Exponential monotonic relationship: y = 2^x (Non-linear, but perfect monotonic Spearman rho = 1.0)
    x_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y_vals = [2.0**x for x in x_vals]
    const_vals = [42.0] * len(x_vals)
    
    df = pl.DataFrame({
        "x": x_vals,
        "y": y_vals,
        "constant": const_vals
    })
    
    corr_dict, strong_pairs = compute_correlation(df, ["x", "y", "constant"])
    
    # Constant column should be safely 0.0 with no divide-by-zero crash
    assert corr_dict["constant"]["x"] == 0.0
    assert corr_dict["constant"]["y"] == 0.0
    
    # Check monotonic pair
    xy_pair = next((p for p in strong_pairs if ("x" in (p["column_a"], p["column_b"]) and "y" in (p["column_a"], p["column_b"]))), None)
    assert xy_pair is not None
    assert "spearman_rho" in xy_pair
    assert xy_pair["spearman_rho"] == pytest.approx(1.0, 1e-3)
