import pytest
import polars as pl
import pandas as pd
from datetime import datetime, timedelta

from app.services.foreign_key_integrity import detect_foreign_key_violations
from app.services.analysis.outliers import detect_time_series_stl_outliers
from app.services.visualization import generate_stl_decomposition_chart, generate_er_relationship_diagram

def test_foreign_key_violations_detection():
    # Parent table: Customers
    customers_df = pl.DataFrame({
        "customer_id": [101, 102, 103, 104],
        "name": ["Alice", "Bob", "Charlie", "Diana"],
    })

    # Child table: Orders (109 and 110 are orphans)
    orders_df = pl.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "cust_id": [101, 102, 109, 110, 103],
        "amount": [250.0, 120.0, 450.0, 90.0, 310.0],
    })

    res = detect_foreign_key_violations(
        primary_df=customers_df,
        foreign_df=orders_df,
        pk_col="customer_id",
        fk_col="cust_id",
        primary_name="customers",
        foreign_name="orders",
    )

    assert res["has_issue"] is True
    assert res["orphan_count"] == 2
    assert res["referential_completeness"] == 60.0
    assert 109 in res["sample_orphan_keys"]
    assert "semi" in res["fix_code"]

def test_time_series_stl_outliers_decomposition():
    # Create 30 days of daily time-series data with weekly seasonality + 1 residual spike
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(30)]
    values = [float(100 + (i % 7) * 20) for i in range(30)]
    values[15] = 950.0  # Residual anomaly

    df = pl.DataFrame({
        "transaction_date": dates,
        "revenue": values,
    })

    res = detect_time_series_stl_outliers(df, date_col="transaction_date", numeric_col="revenue")

    assert res["has_stl_outliers"] is True
    assert res["stl_outlier_count"] >= 1
    assert res["total_observations"] == 30

def test_phase2_visualizations():
    # 1. STL Chart
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(20)]
    values = [50 + (i % 7) * 10 for i in range(20)]
    df = pl.DataFrame({"dt": dates, "val": values})

    stl_chart = generate_stl_decomposition_chart(df, date_col="dt", numeric_col="val")
    assert stl_chart is None or isinstance(stl_chart, str)

    # 2. ER Diagram
    relationships = [{
        "primary_table": "Customers",
        "foreign_table": "Orders",
        "pk_col": "customer_id",
        "fk_col": "cust_id",
        "orphan_count": 2,
    }]
    er_chart = generate_er_relationship_diagram(relationships)
    assert isinstance(er_chart, str)
    assert len(er_chart) > 100
