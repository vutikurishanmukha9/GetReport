import pytest
import polars as pl
from app.services.confidence_scoring import calculate_confidence_scores

def test_confidence_scoring_stability_zero_variance_penalty():
    """Verify stability pillar specifically penalizes zero-variance constant numeric columns."""
    df = pl.DataFrame({
        "constant_feature": [42.0, 42.0, 42.0, 42.0, 42.0],
        "normal_feature": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    
    report = calculate_confidence_scores(df)
    const_col = next(c for c in report.columns if c.column == "constant_feature")
    assert const_col.stability <= 20.0
    assert any("zero variance" in iss.lower() or "constant" in iss.lower() for iss in const_col.issues)
