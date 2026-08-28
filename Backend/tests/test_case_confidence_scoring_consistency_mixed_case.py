import pytest
import polars as pl
from app.services.confidence_scoring import calculate_confidence_scores

def test_confidence_scoring_consistency_mixed_case_penalty():
    """Verify consistency pillar penalizes text columns with erratic casing and whitespace pollution."""
    # 50% uppercase, lowercase, and weird casing
    cities = ["london", "PARIS", "berlin", "TOKYO", "mAdRiD", "rOmE"] * 10
    df = pl.DataFrame({"city": cities})
    
    report = calculate_confidence_scores(df)
    city_col = next(c for c in report.columns if c.column == "city")
    assert city_col.consistency < 100.0
    assert any("casing" in iss.lower() or "case" in iss.lower() for iss in city_col.issues)
