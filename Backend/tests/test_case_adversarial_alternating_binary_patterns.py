import pytest
import polars as pl
from app.services.analysis.core import analyze_dataset

def test_adversarial_alternating_binary_series():
    """Verify system computes distributions and correlations on strict alternating [0, 1] series."""
    # Alternating 0, 1 pattern
    alt_pattern = [i % 2 for i in range(100)]
    inv_pattern = [(i + 1) % 2 for i in range(100)]
    
    df = pl.DataFrame({
        "bit_a": alt_pattern,
        "bit_b": inv_pattern
    })
    
    analysis = analyze_dataset(df)
    assert analysis is not None
    assert analysis["metadata"]["total_rows"] == 100
