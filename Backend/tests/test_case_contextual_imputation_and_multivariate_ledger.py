import pytest
import polars as pl
import numpy as np
from app.services.data_processing import clean_data
from app.services.issue_ledger import detect_issues


def test_contextual_imputation_correlated_features():
    """Verify that contextual imputation reconstructs missing values via covariance with correlated features."""
    np.random.seed(42)
    n = 40
    # Two strongly correlated features: square_feet and price (price = 2 * sqft + 50)
    sqft = [500 + i * 50 for i in range(n)]
    price = [float(2 * s + 50) for s in sqft]

    # Introduce missing values at index 10 and 25
    price[10] = None
    price[25] = None

    df = pl.DataFrame({
        "square_feet": sqft,
        "price": price,
    })

    cleaned_df, report, dag = clean_data(df)

    assert report.numeric_nans_filled == 2
    # Verify values are reconstructed contextually, not flat median
    val_10 = cleaned_df["price"][10]
    val_25 = cleaned_df["price"][25]
    median_val = float(df["price"].drop_nulls().median())

    assert val_10 is not None
    assert val_25 is not None
    # val_10 (sqft=1000) should be close to 2050, far from median (~2450)
    assert abs(val_10 - 2050.0) < 50.0
    # val_25 (sqft=1750) should be close to 3550
    assert abs(val_25 - 3550.0) < 50.0

    # Verify DAG recorded contextual_imputation
    context_nodes = [node for node in dag.nodes.values() if node.operation == "contextual_imputation"]
    assert len(context_nodes) >= 1
    node = context_nodes[0]
    assert node.target_column == "price"
    assert node.parameters["method"] == "contextual_reconstruction"
    assert "square_feet" in node.parameters["predictors"]
    assert node.parameters["confidence_r2"] > 0.90


def test_contextual_imputation_fallback_to_median():
    """Verify graceful fallback to median when features are uncorrelated or lone."""
    df = pl.DataFrame({
        "row_id": list(range(25)),
        "status": ["active", "pending", "active", "active", "active"] * 5,
        "amount": [10.0, None, 30.0, 40.0, 50.0] * 5,
    })

    cleaned_df, report, dag = clean_data(df)
    assert report.numeric_nans_filled == 5
    median_nodes = [node for node in dag.nodes.values() if node.operation == "fill_null_median"]
    assert len(median_nodes) >= 1


def test_multivariate_anomaly_detected_in_issue_ledger():
    """Verify that multi-column anomalies are surfaced in the Issue Ledger with plain business narrative."""
    # 30 standard transactions: age proportional to income
    ages = [25 + (i % 30) for i in range(30)]
    incomes = [30000.0 + (i % 30) * 2000.0 for i in range(30)]
    # Inject a distinct multivariate anomaly: age 20 with extreme 5,000,000 income
    ages.append(20)
    incomes.append(5000000.0)

    df = pl.DataFrame({"age": ages, "income": incomes})

    ledger = detect_issues(df)

    # Check for multivariate_anomaly issue
    multi_issues = [issue for issue in ledger.issues if issue.issue_type == "multivariate_anomaly"]
    assert len(multi_issues) >= 1
    issue = multi_issues[0]

    assert issue.severity in ("high", "medium")
    assert "multi-column anomalies detected" in issue.description
    assert "Review flagged records for cross-column inconsistencies" in issue.suggested_fix

    # Strict constraint: ZERO ML jargon in user-facing texts
    banned_jargon = ["isolation forest", "local outlier factor", "hyperparameter", "automl", "scikit-learn"]
    for word in banned_jargon:
        assert word not in issue.description.lower()
        assert word not in issue.suggested_fix.lower()


def test_issue_ledger_zero_em_dashes():
    """Verify that no issue description, suggested fix, or fix code contains em dashes."""
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": [" Alice ", "Bob", None, "Alice", "Charlie"],
        "price": [10.0, 20.0, 5000.0, None, 15.0],
        "flag": ["true", "false", "yes", "no", "true"],
    })

    ledger = detect_issues(df)
    for issue in ledger.issues:
        assert "—" not in issue.description, f"Em dash found in description: {issue.description}"
        assert "—" not in issue.suggested_fix, f"Em dash found in suggested_fix: {issue.suggested_fix}"
        assert "—" not in issue.fix_code, f"Em dash found in fix_code: {issue.fix_code}"
