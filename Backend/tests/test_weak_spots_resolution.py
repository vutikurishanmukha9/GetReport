import pytest
import polars as pl
from app.services.analysis import classify_numeric_columns, analyze_dataset
from app.services.llm_insight import generate_insights_sync
from app.services.data_processing import clean_data

def test_retailer_id_classified_as_id_not_analytical():
    df = pl.DataFrame({
        "retailer_id": [101, 102, 103, 104, 101, 102, 103, 104, 101, 102],
        "store_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "sales_amount": [150.5, 200.0, 350.2, 90.0, 420.1, 180.3, 290.0, 310.0, 450.0, 120.0],
        "invoice_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10"],
    })

    res = classify_numeric_columns(df, ["retailer_id", "store_id", "sales_amount"])
    assert "retailer_id" in res["excluded"]
    assert "store_id" in res["excluded"]
    assert "sales_amount" in res["analytical"]

def test_invoice_date_excluded_from_categorical_distribution():
    df = pl.DataFrame({
        "sales": [100.0, 200.0, 300.0, 400.0, 500.0],
        "invoice_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
        "region": ["North", "South", "North", "East", "West"],
    })

    analysis = analyze_dataset(df)
    cat_dist = analysis.get("categorical_distribution", {})
    assert "region" in cat_dist
    assert "invoice_date" not in cat_dist

def test_ai_insights_rule_based_fallback_generation():
    analysis_data = {
        "metadata": {"total_rows": 500, "total_columns": 6, "missing_value_pct": 2.5},
        "ranked_insights": [
            {
                "title": "Strong Correlation",
                "description": "Sales and Profit have r=0.92 correlation.",
                "actionable_recommendation": "Cross-sell high profit items."
            }
        ],
        "strong_correlations": [{"column_a": "sales", "column_b": "profit", "r_value": 0.92, "direction": "positive"}],
        "outliers": {"sales": {"count": 12}},
    }

    res = generate_insights_sync(analysis_data)
    assert res.success is True
    assert "Sales and Profit" in res.insights_text
    assert len(res.insights_text) > 20

def test_automated_cleaning_strips_masked_nulls_and_imputes():
    df = pl.DataFrame({
        "status": ["active", "N/A", "null", "?", "pending"],
        "sales": [100.0, None, 300.0, None, 500.0],
    })

    cleaned_df, report, dag = clean_data(df)
    assert report.numeric_nans_filled == 2
    assert report.total_changes >= 2
    assert (cleaned_df["sales"] == 300.0).sum() >= 1
