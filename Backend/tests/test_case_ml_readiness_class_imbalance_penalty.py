import pytest
import polars as pl
from app.services.confidence_scoring import calculate_confidence_scores
from app.services.analysis.ml_readiness import calculate_ml_readiness

def test_ml_readiness_class_imbalance_penalty():
    """Verify that >95% class imbalance is flagged in ML readiness diagnostics."""
    targets = ["Class_A"] * 98 + ["Class_B"] * 2
    df = pl.DataFrame({
        "feature_1": [float(i) for i in range(100)],
        "target_col": targets
    })
    
    conf = calculate_confidence_scores(df)
    ml_report = calculate_ml_readiness(conf, df)
    
    assert ml_report["score"] <= 95.0
    assert any("heavily imbalanced" in r.lower() for r in ml_report["reasons"])
