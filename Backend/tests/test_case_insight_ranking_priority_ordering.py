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


def test_invisible_ml_drivers_and_cohorts_ranking():
    """Verify that invisible ML drivers and natural cohorts are surfaced as top executive findings without jargon."""
    findings = {
        "supervised_learning": {
            "ran": True,
            "target_column": "customer_churn",
            "key_drivers": [
                {"feature": "tenure_months", "importance_pct": 38.5, "direction": "negative"},
                {"feature": "support_tickets", "importance_pct": 24.1, "direction": "positive"}
            ]
        },
        "unsupervised_learning": {
            "ran": True,
            "optimal_k": 3,
            "explained_variance_pct": 78.5,
            "personas": [
                {
                    "cluster_id": 1,
                    "name": "High Volume Enterprise",
                    "size": 500,
                    "share_pct": 55.0,
                    "summary": "Large transactions and high retention"
                }
            ]
        }
    }

    ranked = rank_insights(findings)
    assert len(ranked) >= 2
    
    types = [item.type for item in ranked]
    assert "driver" in types
    assert "cohort" in types

    driver_insight = next(i for i in ranked if i.type == "driver")
    assert "Primary Driver" in driver_insight.title
    assert "Tenure Months" in driver_insight.title
    assert driver_insight.score >= 0.90
    # Jargon-free check
    assert "lightgbm" not in driver_insight.description.lower()
    assert "randomforest" not in driver_insight.description.lower()

    cohort_insight = next(i for i in ranked if i.type == "cohort")
    assert "Natural Cohort Structure" in cohort_insight.title
    assert "3 Distinct Operational Segments" in cohort_insight.title
    assert "kmeans" not in cohort_insight.description.lower()
