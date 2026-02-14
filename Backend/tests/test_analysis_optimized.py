import sys
import os
import time
import polars as pl
import numpy as np
import pytest

# Add Backend to path
sys.path.append(os.path.join(os.getcwd(), "Backend"))

from app.services.analysis.statistics import compute_summary
from app.services.analysis.outliers import detect_outliers
from app.core.config import settings

def test_compute_summary_lazy():
    # Create synthetic data
    df = pl.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50],
        "c": [0.0, 0.0, 0.0, 0.0, 100.0] # Skewed
    })
    
    start = time.perf_counter()
    summary = compute_summary(df, ["a", "b", "c"])
    elapsed = (time.perf_counter() - start) * 1000
    print(f"\nSummary computation took: {elapsed:.2f}ms")
    
    assert "a" in summary
    assert summary["a"]["mean"] == 3.0
    assert summary["a"]["min"] == 1.0
    assert summary["a"]["max"] == 5.0
    
    assert "b" in summary
    assert summary["b"]["mean"] == 30.0
    
    assert "c" in summary
    assert summary["c"]["max"] == 100.0

def test_detect_outliers_lazy():
    # Create data with known outliers
    # IQR for 1-5 is 4-2=2. Bounds: 2 - 1.5*2 = -1, 4 + 1.5*2 = 7.
    # Outliers: 100
    data = [1, 2, 3, 4, 5, 100]
    df = pl.DataFrame({"val": data})
    
    start = time.perf_counter()
    outliers = detect_outliers(df, ["val"])
    elapsed = (time.perf_counter() - start) * 1000
    print(f"\nOutlier detection took: {elapsed:.2f}ms")
    
    assert "val" in outliers
    assert outliers["val"]["count"] == 1
    assert outliers["val"]["max_outlier"] == 100

if __name__ == "__main__":
    print("Running Lazy Analysis Tests...")
    test_compute_summary_lazy()
    test_detect_outliers_lazy()
    print("All tests passed!")
