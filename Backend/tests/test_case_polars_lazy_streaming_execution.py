import os
import polars as pl
import numpy as np
import pytest
from app.services.data_processing import scan_dataframe_lazy, compute_streaming_summary_stats

def test_polars_lazy_streaming_execution(tmp_path):
    """
    Verify Phase 4 Polars Lazy Streaming Execution:
    1. scan_dataframe_lazy correctly reads CSV and Parquet lazily.
    2. compute_streaming_summary_stats matches eager aggregations exactly.
    3. Handles null values and large vector batches correctly.
    """
    # 1. Generate test dataset (5,000 rows x 4 numeric columns)
    np.random.seed(42)
    n = 5000
    df = pl.DataFrame({
        "sales": np.random.normal(1000, 150, n),
        "expenses": np.random.normal(600, 80, n),
        "quantity": np.random.randint(1, 50, n).astype(float),
        "discount": np.random.uniform(0.05, 0.30, n)
    })
    
    # Introduce some nulls
    df = df.with_columns([
        pl.when(pl.col("sales") > 1200).then(None).otherwise(pl.col("sales")).alias("sales")
    ])
    
    csv_path = tmp_path / "stream_test.csv"
    parquet_path = tmp_path / "stream_test.parquet"
    
    df.write_csv(str(csv_path))
    df.write_parquet(str(parquet_path))
    
    # 2. Test Lazy Scanning
    lazy_csv = scan_dataframe_lazy(str(csv_path))
    assert isinstance(lazy_csv, pl.LazyFrame)
    
    lazy_parquet = scan_dataframe_lazy(str(parquet_path))
    assert isinstance(lazy_parquet, pl.LazyFrame)
    
    # 3. Test Streaming Summary Stats Computation
    streaming_stats = compute_streaming_summary_stats(lazy_csv)
    
    assert "sales" in streaming_stats
    assert "expenses" in streaming_stats
    assert "quantity" in streaming_stats
    assert "discount" in streaming_stats
    
    # Verify exact match with eager Polars calculation
    expected_sales_mean = float(df["sales"].drop_nulls().mean())
    expected_sales_nulls = int(df["sales"].null_count())
    
    assert abs(streaming_stats["sales"]["mean"] - expected_sales_mean) < 1e-4
    assert streaming_stats["sales"]["null_count"] == expected_sales_nulls
    assert streaming_stats["sales"]["count"] == n - expected_sales_nulls
