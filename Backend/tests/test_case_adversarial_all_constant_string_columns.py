import pytest
import polars as pl
from app.services.analysis.core import analyze_dataset

def test_adversarial_constant_strings_only():
    """Verify system analyzes datasets containing only constant repetitive text columns."""
    df = pl.DataFrame({
        "status": ["active"] * 100,
        "region": ["EMEA"] * 100
    })
    
    analysis = analyze_dataset(df)
    assert analysis is not None
    assert analysis["metadata"]["total_rows"] == 100
    assert analysis["metadata"]["categorical_columns"] == 2
    assert "status" in analysis["categorical_distribution"]
