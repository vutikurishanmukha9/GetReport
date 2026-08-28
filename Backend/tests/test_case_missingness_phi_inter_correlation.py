import pytest
import polars as pl
from app.services.analysis.missing import analyze_missing_patterns

def test_missingness_phi_inter_correlation_and_deletion_impact():
    """Verify analyze_missing_patterns computes Phi inter-correlations and complete-case analysis impact."""
    # Col A and Col B are missing in the exact same rows (Simultaneous missingness, Phi ~ 1.0)
    df = pl.DataFrame({
        "col_a": [1.0, None, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "col_b": [10.0, None, 30.0, None, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "col_c": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    })
    
    res = analyze_missing_patterns(df)
    assert res["has_missing"] is True
    assert "missing_inter_correlations" in res
    assert len(res["missing_inter_correlations"]) >= 1
    
    inter_corr = res["missing_inter_correlations"][0]
    assert inter_corr["phi_coefficient"] == pytest.approx(1.0, 1e-2)
    assert inter_corr["co_occurrence"] == "simultaneous"
    
    assert "complete_cases_percentage" in res["row_patterns"]
    assert res["row_patterns"]["complete_cases_percentage"] == 80.0
