import os
import polars as pl
import numpy as np
import pytest
from app.services.data_processing import scan_dataframe_lazy, compute_streaming_summary_stats

def test_polars_lazy_streaming_skewness_and_extreme_distributions(tmp_path):
    """
    Verify streaming summary statistics handle extreme numerical distributions:
    1. Zero-variance constant columns.
    2. Highly skewed log-normal distributions.
    3. Negative-only float ranges.
    """
    n = 10000
    np.random.seed(123)
    
    constant_col = np.full(n, 42.0)
    skewed_col = np.random.lognormal(mean=2.0, sigma=1.2, size=n)
    negative_col = -np.random.exponential(scale=50.0, size=n)
    
    df = pl.DataFrame({
        "constant_val": constant_col,
        "skewed_val": skewed_col,
        "negative_val": negative_col
    })
    
    parquet_path = tmp_path / "extreme_dist.parquet"
    df.write_parquet(str(parquet_path))
    
    lazy_df = scan_dataframe_lazy(str(parquet_path))
    stats = compute_streaming_summary_stats(lazy_df)
    
    # 1. Constant Column Checks
    assert stats["constant_val"]["min"] == 42.0
    assert stats["constant_val"]["max"] == 42.0
    assert stats["constant_val"]["mean"] == 42.0
    assert abs(stats["constant_val"]["std"]) < 1e-6
    
    # 2. Skewed Column Checks
    assert stats["skewed_val"]["min"] > 0
    assert stats["skewed_val"]["max"] > stats["skewed_val"]["mean"]
    assert abs(stats["skewed_val"]["mean"] - float(df["skewed_val"].mean())) < 1e-3
    
    # 3. Negative Column Checks
    assert stats["negative_val"]["max"] <= 0
    assert abs(stats["negative_val"]["mean"] - float(df["negative_val"].mean())) < 1e-3
