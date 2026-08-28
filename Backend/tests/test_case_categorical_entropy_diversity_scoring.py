import pytest
import polars as pl
from app.services.confidence_scoring import calculate_confidence_scores

def test_categorical_entropy_diversity_scoring():
    """Verify confidence scoring penalizes columns where single category dominates >95%."""
    # 96 rows of 'A' and 4 rows of 'B'
    dominance = ["A"] * 96 + ["B"] * 4
    df = pl.DataFrame({"dominant_cat": dominance})
    
    report = calculate_confidence_scores(df)
    dom_col = next(c for c in report.columns if c.column == "dominant_cat")
    # Low category diversity penalty applied to stability pillar
    assert dom_col.stability <= 50.0
    assert any("dominates" in iss.lower() or "diversity" in iss.lower() for iss in dom_col.issues)
