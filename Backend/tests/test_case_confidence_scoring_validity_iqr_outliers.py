import pytest
import polars as pl
from app.services.confidence_scoring import calculate_confidence_scores

def test_confidence_scoring_validity_iqr_outlier_penalty():
    """Verify validity pillar penalizes columns with high ratios of extreme outliers."""
    # Column with normal values and extreme spikes
    values = [10.0, 11.0, 12.0, 10.5, 11.2, 9.8, 10.1, 10.9, 1000.0, 5000.0]
    df = pl.DataFrame({"spiky_data": values})
    
    report = calculate_confidence_scores(df)
    spiky_col = next(c for c in report.columns if c.column == "spiky_data")
    assert spiky_col.validity < 100.0
    assert any("outlier" in iss.lower() for iss in spiky_col.issues)
