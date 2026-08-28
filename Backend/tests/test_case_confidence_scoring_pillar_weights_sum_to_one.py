import pytest
import polars as pl
from app.services.confidence_scoring import calculate_confidence_scores

def test_confidence_scoring_pillar_weights_sum_to_100():
    """Verify mathematical formula weights: Completeness (35%), Consistency (25%), Validity (25%), Stability (15%)."""
    # 0.35 + 0.25 + 0.25 + 0.15 == 1.00
    weights = [0.35, 0.25, 0.25, 0.15]
    assert sum(weights) == pytest.approx(1.0, 1e-6)
    
    df = pl.DataFrame({"metric": [10.0, 20.0, 30.0, 40.0, 50.0]})
    report = calculate_confidence_scores(df)
    col = report.columns[0]
    
    expected_overall = (
        col.completeness * 0.35 +
        col.consistency * 0.25 +
        col.validity * 0.25 +
        col.stability * 0.15
    )
    assert col.overall == pytest.approx(expected_overall, 0.01)
