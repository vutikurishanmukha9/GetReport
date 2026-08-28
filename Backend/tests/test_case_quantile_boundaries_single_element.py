import pytest
import polars as pl
from app.services.analysis.statistics import compute_summary

def test_quantile_boundaries_small_sample():
    """Verify summary statistics calculation on small datasets (5 items)."""
    df = pl.DataFrame({"metric": [10.0, 20.0, 30.0, 40.0, 50.0]})
    summary = compute_summary(df, ["metric"])
    assert "metric" in summary
    assert summary["metric"]["min"] == 10.0
    assert summary["metric"]["max"] == 50.0
    assert summary["metric"]["mean"] == 30.0
    assert summary["metric"]["50%"] == 30.0
