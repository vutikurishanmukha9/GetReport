import pytest
from datetime import datetime, timedelta
import polars as pl
from app.services.hiplot_service import HiPlotDataService, HiPlotDimensionType


def test_prepare_hiplot_payload():
    base_date = datetime(2026, 1, 1)
    df = pl.DataFrame({
        "user_id": list(range(1, 101)),
        "age": [20 + (i % 50) for i in range(100)],
        "income": [10.0 * (10 ** (i % 4)) for i in range(100)],  # Heavy scale spread -> numeric_log
        "tier": ["Bronze", "Silver", "Gold", "Platinum"] * 25,
        "joined_date": [base_date + timedelta(days=i) for i in range(100)]
    })

    payload = HiPlotDataService.prepare_hiplot_payload(df, max_rows=50)

    assert payload["total_rows"] == 100
    assert payload["sampled_rows"] == 50
    assert payload["is_sampled"] is True
    assert len(payload["dimensions"]) == 5

    dim_map = {d["name"]: d for d in payload["dimensions"]}
    assert dim_map["age"]["type"] == HiPlotDimensionType.NUMERIC
    assert dim_map["age"]["is_numeric"] is True
    assert "p5" in dim_map["age"]
    assert "p95" in dim_map["age"]

    assert dim_map["income"]["type"] == HiPlotDimensionType.NUMERIC_LOG
    assert dim_map["tier"]["type"] == HiPlotDimensionType.CATEGORICAL
    assert len(dim_map["tier"]["categories"]) == 4
    assert dim_map["joined_date"]["type"] == HiPlotDimensionType.TIMESTAMP

    assert len(payload["datapoints"]) == 50
    assert "age" in payload["datapoints"][0]
