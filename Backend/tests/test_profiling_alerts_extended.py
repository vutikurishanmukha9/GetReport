"""
Tests for Extended Alert Taxonomy adapted from fg-data-profiling:
- zero_inflation
- extreme_skewness
- rare_categories
- uniform_distribution
"""
import polars as pl
import numpy as np
from app.services.issue_ledger import (
    detect_issues,
    _detect_zero_inflation_issues,
    _detect_extreme_skewness_issues,
    _detect_rare_category_issues,
    _detect_uniform_distribution_issues,
)


def test_zero_inflation_detection():
    # 50 zeros out of 100 rows = 50% zeros (>= 40% threshold)
    values = [0] * 50 + list(range(1, 51))
    df = pl.DataFrame({"feature_z": values})

    issues = _detect_zero_inflation_issues(df)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type == "zero_inflation"
    assert issue.column == "feature_z"
    assert issue.affected_rows == 50
    assert issue.affected_pct == 50.0
    assert "zero inflation" in issue.description


def test_extreme_skewness_detection():
    # Heavy right tail (exponential or power law distribution)
    np.random.seed(42)
    skewed_vals = list(np.random.exponential(scale=2.0, size=200) ** 3)
    df = pl.DataFrame({"skewed_col": skewed_vals})

    issues = _detect_extreme_skewness_issues(df)
    assert len(issues) >= 1
    issue = issues[0]
    assert issue.issue_type == "extreme_skewness"
    assert issue.column == "skewed_col"
    assert "Extreme distribution skewness" in issue.description
    assert "log1p" in issue.suggested_fix or "transformation" in issue.suggested_fix


def test_rare_categories_detection():
    # 200 total rows. Category 'A' has 100 rows, 'B' has 96 rows, 'Rare1' has 2 rows (1%), 'Rare2' has 2 rows (1%)
    cats = ["A"] * 100 + ["B"] * 96 + ["Rare1"] * 2 + ["Rare2"] * 2
    df = pl.DataFrame({"category_col": cats})

    issues = _detect_rare_category_issues(df)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type == "rare_categories"
    assert issue.column == "category_col"
    assert "rare category levels" in issue.description
    assert "Other" in issue.suggested_fix


def test_uniform_distribution_detection():
    # 10 categories, each repeating exactly 20 times (200 rows, perfectly uniform)
    categories = [f"Cat_{i}" for i in range(10)]
    cats = categories * 20
    df = pl.DataFrame({"uniform_col": cats})

    issues = _detect_uniform_distribution_issues(df)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type == "uniform_distribution"
    assert issue.column == "uniform_col"
    assert "Near-uniform distribution detected" in issue.description


def test_detect_issues_integration_with_extended_alerts():
    # Combined dataset exhibiting multiple alerts
    n = 200
    zeros = [0] * 120 + list(range(1, 81)) # 60% zeros
    skewed = list(np.random.exponential(scale=3.0, size=n) ** 3)
    categories = ["Main"] * 196 + ["Minor_A"] * 2 + ["Minor_B"] * 2 # rare categories (<1% frequency)

    df = pl.DataFrame({
        "zeros_col": zeros,
        "skew_col": skewed,
        "cat_col": categories,
    })

    ledger = detect_issues(df)
    issue_types = {i.issue_type for i in ledger.issues}

    assert "zero_inflation" in issue_types
    assert "extreme_skewness" in issue_types
    assert "rare_categories" in issue_types

    summary = ledger.get_summary()
    assert summary["total"] >= 3
