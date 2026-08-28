import pytest
import polars as pl
from app.services.confidence_scoring import calculate_confidence_scores
from app.services.analysis.ml_readiness import calculate_ml_readiness

def test_ml_readiness_critical_null_gate_failure():
    """Verify that any column with >70% null values instantly triggers the Not Ready hard failure gate."""
    df = pl.DataFrame({
        "id": list(range(100)),
        "clean_feature": [float(i) for i in range(100)],
        "critical_missing_col": [float(i) if i < 25 else None for i in range(100)] # 75% missing
    })
    
    conf = calculate_confidence_scores(df)
    ml_report = calculate_ml_readiness(conf, df)
    
    assert ml_report["score"] == 0.0
    assert ml_report["status"] == "Not Ready"
    assert any("critical missingness" in r.lower() or "missing values" in r.lower() for r in ml_report["reasons"])
