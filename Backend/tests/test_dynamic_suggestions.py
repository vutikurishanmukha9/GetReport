import polars as pl
import pytest
from app.services.data_processing import inspect_dataset, _compute_dynamic_missing_suggestion


def test_dynamic_suggestion_skewed_numeric():
    """Verify that skewed numeric distributions dynamically suggest median imputation."""
    # Highly skewed distribution (e.g. salary or prices)
    df = pl.DataFrame({
        "salary": [25000.0, 28000.0, 31000.0, 32000.0, 35000.0, 400000.0, None, 420000.0]
    })
    res = inspect_dataset(df)
    issues = [i for i in res["issues"] if i["column"] == "salary"]
    assert len(issues) == 1
    issue = issues[0]
    assert issue["action"] == "fill_median"
    assert issue["label"] == "Fill with Median"
    assert "skewed" in issue["rationale"].lower() or "tail" in issue["rationale"].lower()


def test_dynamic_suggestion_normal_numeric():
    """Verify that symmetric numeric distributions dynamically suggest mean imputation."""
    # Symmetrically distributed continuous values
    df = pl.DataFrame({
        "temperature": [20.0, 20.1, 20.2, 19.9, 19.8, 20.0, None, 20.1]
    })
    res = inspect_dataset(df)
    issues = [i for i in res["issues"] if i["column"] == "temperature"]
    assert len(issues) == 1
    issue = issues[0]
    assert issue["action"] == "fill_mean"
    assert issue["label"] == "Fill with Average"
    assert "normal" in issue["rationale"].lower() or "mean" in issue["rationale"].lower()


def test_dynamic_suggestion_discrete_integer():
    """Verify that discrete integer scales (ratings) dynamically suggest mode to prevent decimals."""
    df = pl.DataFrame({
        "rating": [5, 5, 4, 5, 3, 5, None, 4]
    })
    res = inspect_dataset(df)
    issues = [i for i in res["issues"] if i["column"] == "rating"]
    assert len(issues) == 1
    issue = issues[0]
    assert issue["action"] == "fill_mode"
    assert issue["label"] == "Fill with Most Frequent"
    assert "discrete" in issue["rationale"].lower()


def test_dynamic_suggestion_identifier_column():
    """Verify that ID columns dynamically suggest dropping rows to protect key uniqueness."""
    df = pl.DataFrame({
        "customer_id": ["C101", "C102", "C103", None, "C105"]
    })
    res = inspect_dataset(df)
    issues = [i for i in res["issues"] if i["column"] == "customer_id"]
    assert len(issues) == 1
    issue = issues[0]
    assert issue["action"] == "drop_rows"
    assert issue["label"] == "Drop Missing Rows"
    assert "identifier" in issue["rationale"].lower() or "key" in issue["rationale"].lower()


def test_dynamic_suggestion_low_cardinality_category():
    """Verify that low-cardinality categorical attributes dynamically suggest mode imputation."""
    df = pl.DataFrame({
        "status": ["Active", "Active", "Inactive", "Active", None, "Pending"]
    })
    res = inspect_dataset(df)
    issues = [i for i in res["issues"] if i["column"] == "status"]
    assert len(issues) == 1
    issue = issues[0]
    assert issue["action"] == "fill_mode"
    assert "Active" in issue["rationale"] or "Active" in issue["suggestion"]
