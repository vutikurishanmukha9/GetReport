"""
Unit Tests for Apache Superset Native In-Memory Analytics Engine (superset_engine.py).
Tests type mapping, dataset introspection, ad-hoc filters, temporal bucketing,
and ECharts configuration generation for all enterprise chart types.
"""

from datetime import datetime, date
import math
import pytest
import polars as pl

from app.services.superset_engine import (
    SupersetEngine,
    SupersetExploreQuery,
    ExploreFilter,
    GenericDataType,
    map_polars_to_superset_dtype,
    _sanitize_for_json,
    SUPERSET_COLOR_PALETTES,
)


@pytest.fixture
def sample_df():
    """Generates a representative analytical dataset with temporal, numeric, and categorical columns."""
    return pl.DataFrame({
        "order_id": list(range(1, 101)),
        "order_date": [
            date(2026, 1, 1 + (i % 25)) if i < 50 else date(2026, 2, 1 + (i % 25))
            for i in range(100)
        ],
        "category": (["Electronics", "Apparel", "Home Goods", "Automotive"] * 25),
        "region": (["North", "South", "East", "West", "Central"] * 20),
        "revenue": [round(15.5 * (i + 1) + (i % 7) * 3.2, 2) for i in range(100)],
        "units": [(i % 12) + 1 for i in range(100)],
        "discount": [0.05 if i % 2 == 0 else 0.15 for i in range(100)],
        "notes": [f"Order note {i}" if i % 5 != 0 else None for i in range(100)],
    })


@pytest.fixture
def mock_engine(sample_df, monkeypatch):
    """Instantiates SupersetEngine with mocked dataset resolution."""
    engine = SupersetEngine()
    monkeypatch.setattr(engine, "resolve_dataset_df", lambda task_id: sample_df)
    return engine


# ─── 1. Type Mapping & Sanitization Tests ────────────────────────────────────

def test_type_mapping():
    assert map_polars_to_superset_dtype(pl.Int32) == GenericDataType.NUMERIC
    assert map_polars_to_superset_dtype(pl.Float64) == GenericDataType.NUMERIC
    assert map_polars_to_superset_dtype(pl.Date) == GenericDataType.TEMPORAL
    assert map_polars_to_superset_dtype(pl.Datetime) == GenericDataType.TEMPORAL
    assert map_polars_to_superset_dtype(pl.Boolean) == GenericDataType.BOOLEAN
    assert map_polars_to_superset_dtype(pl.String) == GenericDataType.STRING


def test_sanitize_for_json():
    assert _sanitize_for_json(None) is None
    assert _sanitize_for_json(float("nan")) is None
    assert _sanitize_for_json(float("inf")) is None
    assert _sanitize_for_json(12.3456) == 12.35
    assert _sanitize_for_json(100) == 100
    assert _sanitize_for_json("Electronics") == "Electronics"


# ─── 2. Schema Introspection Tests ───────────────────────────────────────────

def test_introspect_dataset(mock_engine):
    meta = mock_engine.introspect_dataset("test_task_123")
    assert meta["task_id"] == "test_task_123"
    assert meta["total_rows"] == 100
    assert meta["total_columns"] == 8
    assert "revenue" in meta["numeric_columns"]
    assert "units" in meta["numeric_columns"]
    assert "category" in meta["categorical_columns"]
    assert "order_date" in meta["temporal_columns"]
    assert "day" in meta["time_grains"]

    # Check sample values in column metadata
    cat_col = next(c for c in meta["columns"] if c["name"] == "category")
    assert len(cat_col["sample_values"]) <= 10
    assert "Electronics" in cat_col["sample_values"]


# ─── 3. Ad-Hoc Filter Engine Tests ───────────────────────────────────────────

def test_apply_filters_equality(mock_engine, sample_df):
    filters = [ExploreFilter(column="category", operator="eq", value="Electronics")]
    filtered = mock_engine._apply_filters(sample_df, filters)
    assert len(filtered) == 25
    assert (filtered["category"] == "Electronics").all()


def test_apply_filters_comparison(mock_engine, sample_df):
    filters = [ExploreFilter(column="units", operator="gt", value=10)]
    filtered = mock_engine._apply_filters(sample_df, filters)
    assert (filtered["units"] > 10).all()


def test_apply_filters_in_list(mock_engine, sample_df):
    filters = [ExploreFilter(column="region", operator="in", value=["North", "South"])]
    filtered = mock_engine._apply_filters(sample_df, filters)
    assert len(filtered) == 40
    assert set(filtered["region"].unique().to_list()) == {"North", "South"}


def test_apply_filters_nulls(mock_engine, sample_df):
    filters = [ExploreFilter(column="notes", operator="is_null", value=None)]
    filtered = mock_engine._apply_filters(sample_df, filters)
    assert len(filtered) == 20
    assert filtered["notes"].is_null().all()


# ─── 4. Temporal Bucketing Tests ─────────────────────────────────────────────

def test_temporal_bucketing_month(mock_engine, sample_df):
    df_bucketed = mock_engine._apply_temporal_bucketing(sample_df, "order_date", "month")
    unique_dates = df_bucketed["order_date"].unique().to_list()
    assert len(unique_dates) == 2
    assert "2026-01-01" in unique_dates
    assert "2026-02-01" in unique_dates


def test_temporal_bucketing_week(mock_engine, sample_df):
    df_bucketed = mock_engine._apply_temporal_bucketing(sample_df, "order_date", "week")
    unique_dates = df_bucketed["order_date"].unique().to_list()
    assert len(unique_dates) > 2


# ─── 5. Aggregations & ECharts Generation Tests ──────────────────────────────

def test_execute_explore_query_bar(mock_engine):
    query = SupersetExploreQuery(
        task_id="test_task_123",
        chart_type="echarts_bar",
        dimension="category",
        metric="revenue",
        aggregate="SUM",
        row_limit=10,
    )
    result = mock_engine.execute_explore_query(query)

    assert result["total_groups"] == 4
    assert len(result["records"]) == 4
    assert result["kpis"]["total_rows"] == 100
    assert result["kpis"]["sum"] is not None

    option = result["echarts_option"]
    assert option["xAxis"]["type"] == "category"
    assert len(option["series"]) == 1
    assert option["series"][0]["type"] == "bar"


def test_execute_explore_query_stacked_bar(mock_engine):
    query = SupersetExploreQuery(
        task_id="test_task_123",
        chart_type="echarts_bar",
        dimension="category",
        secondary_dimension="region",
        metric="revenue",
        aggregate="AVG",
    )
    result = mock_engine.execute_explore_query(query)
    option = result["echarts_option"]
    assert len(option["series"]) > 1  # Multi-series stacked
    assert option["series"][0]["stack"] == "total"


def test_execute_explore_query_pie(mock_engine):
    query = SupersetExploreQuery(
        task_id="test_task_123",
        chart_type="echarts_pie",
        dimension="category",
        metric="units",
        aggregate="SUM",
    )
    result = mock_engine.execute_explore_query(query)
    option = result["echarts_option"]
    assert option["series"][0]["type"] == "pie"
    assert len(option["series"][0]["data"]) == 4


def test_execute_explore_query_scatter(mock_engine):
    query = SupersetExploreQuery(
        task_id="test_task_123",
        chart_type="echarts_scatter",
        dimension="units",
        metric="revenue",
        aggregate="AVG",
    )
    result = mock_engine.execute_explore_query(query)
    option = result["echarts_option"]
    assert option["series"][0]["type"] == "scatter"


def test_execute_explore_query_heatmap(mock_engine):
    query = SupersetExploreQuery(
        task_id="test_task_123",
        chart_type="echarts_heatmap",
        dimension="category",
        secondary_dimension="region",
        metric="revenue",
        aggregate="AVG",
    )
    result = mock_engine.execute_explore_query(query)
    option = result["echarts_option"]
    assert option["series"][0]["type"] == "heatmap"
    assert "visualMap" in option


def test_execute_explore_query_boxplot(mock_engine):
    query = SupersetExploreQuery(
        task_id="test_task_123",
        chart_type="echarts_boxplot",
        dimension="category",
        metric="revenue",
    )
    result = mock_engine.execute_explore_query(query)
    option = result["echarts_option"]
    assert option["series"][0]["type"] == "boxplot"
    # Each data item should have [min, q1, med, q3, max]
    assert len(option["series"][0]["data"][0]) == 5


def test_execute_explore_query_datazoom_trigger(mock_engine):
    # Query with >15 categories or temporal bucketing triggers dataZoom
    query = SupersetExploreQuery(
        task_id="test_task_123",
        chart_type="echarts_line",
        dimension="order_date",
        time_grain="day",
        metric="revenue",
        aggregate="SUM",
        row_limit=50,
    )
    result = mock_engine.execute_explore_query(query)
    option = result["echarts_option"]
    assert "dataZoom" in option
    assert option["dataZoom"][0]["type"] == "slider"


def test_execute_explore_query_empty_filtered(mock_engine):
    # Filter that matches nothing
    query = SupersetExploreQuery(
        task_id="test_task_123",
        chart_type="echarts_bar",
        dimension="category",
        filters=[ExploreFilter(column="category", operator="eq", value="NonExistentCategory")],
    )
    result = mock_engine.execute_explore_query(query)
    assert result["records"] == []
    assert result["kpis"]["total_rows"] == 0
    assert "title" in result["echarts_option"]

