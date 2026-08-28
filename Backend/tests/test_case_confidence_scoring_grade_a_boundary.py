import pytest
import polars as pl
from app.services.confidence_scoring import calculate_confidence_scores

def test_confidence_scoring_grade_a_boundary_assignment():
    """Verify that clean columns scoring >=90% overall receive a Grade A."""
    df = pl.DataFrame({
        "perfect_col": [float(i) for i in range(100)]
    })
    
    report = calculate_confidence_scores(df)
    assert report.columns[0]._get_grade() == "A"
    assert report.columns[0].overall >= 90.0
    assert len(report.critical_issues) == 0
