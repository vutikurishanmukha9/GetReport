import pytest
import polars as pl
from app.services.confidence_scoring import ColumnConfidence

def test_confidence_scoring_grade_f_boundary_assignment():
    """Verify that columns scoring <40% overall receive a Grade F."""
    low_confidence_col = ColumnConfidence(
        column="failed_col",
        completeness=10.0,
        consistency=20.0,
        validity=20.0,
        stability=10.0,
        overall=15.0,
        issues=["Critical data degradation"]
    )
    
    assert low_confidence_col._get_grade() == "F"
    assert low_confidence_col.overall < 40.0
