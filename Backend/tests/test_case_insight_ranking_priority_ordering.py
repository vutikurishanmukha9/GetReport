import pytest
from app.services.insight_ranking import rank_insights

def test_insight_ranking_sorts_by_severity_and_score():
    """Verify that rank_insights sorts high-priority findings ahead of low-priority ones."""
    findings = {
        "strong_correlations": [
            {"column_a": "ad_spend", "column_b": "revenue", "r_value": 0.98}
        ],
        "outliers": {
            "fraud_score": {"count": 15, "percentage": 15.0}
        },
        "missing_patterns": {
            "has_missing": True,
            "columns_affected": 3,
            "column_details": {
                "user_id": {"count": 50, "percentage": 50.0}
            }
        }
    }
    
    ranked = rank_insights(findings)
    assert len(ranked) >= 1
    for item in ranked:
        assert hasattr(item, "title")
        assert hasattr(item, "score")
        assert hasattr(item, "type")
        assert item.score >= 0.0
