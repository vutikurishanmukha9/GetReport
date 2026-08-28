import pytest
import polars as pl
from app.services.analysis.statistics import compute_summary

def test_adversarial_64bit_large_integer_values():
    """Verify statistical summarization handles maximum signed 64-bit integer values without integer overflow."""
    # Near max int64: 2^62
    large_vals = [2**60, 2**61, 2**62, 2**60, 2**61]
    df = pl.DataFrame({"large_ints": large_vals})
    
    summary = compute_summary(df, ["large_ints"])
    assert "large_ints" in summary
    assert summary["large_ints"]["max"] > 10**18
    assert summary["large_ints"]["mean"] > 0
