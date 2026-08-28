import pytest
import polars as pl
from app.services.confidence_scoring import calculate_confidence_scores
from app.services.analysis.ml_readiness import calculate_ml_readiness

def test_ml_readiness_constant_columns_penalty():
    """Verify that having constant zero-variance features deducts points and flags warnings in ML readiness."""
    df = pl.DataFrame({
        "constant_1": [10.0] * 50,
        "constant_2": [99.0] * 50,
        "active_feature": [float(i) for i in range(50)]
    })
    
    conf = calculate_confidence_scores(df)
    ml_report = calculate_ml_readiness(conf, df)
    
    assert any("constant" in r.lower() or "zero variance" in r.lower() for r in ml_report["reasons"])
    assert ml_report["score"] <= 95.0
