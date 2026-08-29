import os
import gc
import tracemalloc
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import pytest
from app.services.analysis import analyze_dataset
from app.services.visualization import generate_charts
from app.services.data_processing import impute_multivariate_mice
from app.services.smart_schema import discover_symbolic_equations
from app.core.config import settings

def test_memory_allocation_bounds_and_figure_cleanup():
    """
    Verify memory budget controls:
    1. Matplotlib figures are fully closed with 0 lingering figures in memory.
    2. Symbolic equation solver and MICE imputation run within tight memory overhead.
    3. Intermediate buffers are garbage collected properly.
    """
    tracemalloc.start()
    gc.collect()
    
    # Generate a realistic numerical + categorical dataset (10,000 rows x 8 columns)
    np.random.seed(42)
    n_rows = 10000
    df = pl.DataFrame({
        "revenue": np.random.normal(5000, 1000, n_rows),
        "cost": np.random.normal(3000, 500, n_rows),
        "profit": np.random.normal(2000, 700, n_rows),
        "tax": np.random.uniform(50, 300, n_rows),
        "units": np.random.randint(1, 100, n_rows).astype(float),
        "category": np.random.choice(["Alpha", "Beta", "Gamma", "Delta", "Epsilon"], n_rows),
        "region": np.random.choice(["North", "South", "East", "West"], n_rows),
        "rating": np.random.uniform(1.0, 5.0, n_rows)
    })
    
    # 1. Test Analysis and Chart generation
    analysis_res = analyze_dataset(df)
    assert "summary" in analysis_res
    
    charts, _ = generate_charts(df)
    assert len(charts) > 0
    
    # Check that all Matplotlib figures have been closed and freed
    assert len(plt.get_fignums()) == 0
    
    # 2. Test Symbolic Equation discovery with memory optimization
    equations = discover_symbolic_equations(df)
    assert isinstance(equations, list)
    
    # 3. Test MICE imputation
    df_with_nulls = df.with_columns([
        pl.when(pl.col("revenue") > 6000).then(None).otherwise(pl.col("revenue")).alias("revenue")
    ])
    imputed_df = impute_multivariate_mice(df_with_nulls, numeric_cols=["revenue", "cost", "profit", "tax"])
    assert imputed_df["revenue"].null_count() == 0
    
    # Clean up and measure memory usage
    del df, df_with_nulls, imputed_df, charts, analysis_res, equations
    gc.collect()
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_mb = peak_mem / (1024 * 1024)
    # Peak memory during execution must remain strictly controlled (< 80 MB Python heap)
    assert peak_mb < 80.0
