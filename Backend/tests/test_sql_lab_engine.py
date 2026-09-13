import pytest
import polars as pl
from app.services.duckdb_engine import DuckDBAnalyticalSession, DuckDBSecurityError
from app.services.chart_schema import DeclarativeChartBuilder


def test_duckdb_schema_metadata_introspection():
    session = DuckDBAnalyticalSession()
    try:
        # Register two tables
        users_df = pl.DataFrame({
            "user_id": [1, 2, 3],
            "username": ["alice", "bob", "carol"],
            "is_active": [True, True, False]
        })
        orders_df = pl.DataFrame({
            "order_id": [101, 102, 103],
            "user_id": [1, 2, 1],
            "amount": [99.5, 149.0, 25.0]
        })

        session.register_polars("users", users_df)
        session.register_polars("orders", orders_df)

        meta = session.get_schema_metadata()
        assert meta["total_tables"] == 2
        table_names = [t["table_name"] for t in meta["tables"]]
        assert "users" in table_names
        assert "orders" in table_names

        users_table = next(t for t in meta["tables"] if t["table_name"] == "users")
        assert users_table["row_count"] == 3
        assert users_table["column_count"] == 3
        col_names = [c["name"] for c in users_table["columns"]]
        assert "user_id" in col_names
        assert "username" in col_names
        assert len(users_table["columns"][1]["sample_values"]) > 0
    finally:
        session.close()


def test_duckdb_parameterized_query_execution():
    session = DuckDBAnalyticalSession()
    try:
        sales_df = pl.DataFrame({
            "region": ["North", "South", "East", "West"],
            "revenue": [50000.0, 12000.0, 45000.0, 8000.0]
        })
        session.register_polars("sales", sales_df)

        # Parameterized query with {{ min_rev }}
        query = "SELECT region, revenue FROM sales WHERE revenue > {{ min_rev }} ORDER BY revenue DESC"
        res = session.execute_parameterized_query(query, params={"min_rev": 20000.0})

        assert res["success"] is True
        assert res["row_count"] == 2
        assert res["data"][0]["region"] == "North"
        assert res["data"][1]["region"] == "East"
        assert "duration_ms" in res

        # Verify security blocking
        with pytest.raises(DuckDBSecurityError):
            session.execute_parameterized_query("DROP TABLE sales")
    finally:
        session.close()


def test_declarative_chart_builder_and_auto_infer():
    # 1. Manual Spec Builder
    data = [
        {"department": "Engineering", "headcount": 45},
        {"department": "Sales", "headcount": 30},
        {"department": "Marketing", "headcount": 15}
    ]
    spec = DeclarativeChartBuilder.build_spec(
        chart_type="bar",
        title="Department Headcount",
        data=data,
        x_axis="department",
        metrics=["headcount"],
        palette_name="cyberpunk"
    )

    assert spec["chart_type"] == "bar"
    assert spec["title"] == "Department Headcount"
    assert spec["encoding"]["x"]["field"] == "department"
    assert spec["encoding"]["y"]["fields"] == ["headcount"]
    assert len(spec["encoding"]["color"]["palette"]) > 0
    assert len(spec["data"]) == 3

    # 2. Auto-Infer Time Series
    ts_data = [
        {"date": "2026-01-01", "active_users": 100},
        {"date": "2026-01-02", "active_users": 120},
        {"date": "2026-01-03", "active_users": 115}
    ]
    auto_ts = DeclarativeChartBuilder.auto_infer_chart_spec(ts_data, title="User Growth")
    assert auto_ts is not None
    assert auto_ts["chart_type"] == "line"
    assert auto_ts["encoding"]["x"]["field"] == "date"
    assert auto_ts["encoding"]["y"]["fields"] == ["active_users"]
