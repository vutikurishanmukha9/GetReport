import tempfile
import os
import polars as pl
import pytest

from app.services.data_processing import join_datasets

def test_polars_join_datasets_inner_and_left():
    """
    Test joining 2 Polars DataFrames on a primary key column.
    """
    df1 = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [85.0, 92.5, 78.0]
    })
    
    df2 = pl.DataFrame({
        "id": [1, 2, 4],
        "city": ["New York", "London", "Tokyo"],
        "score": [100.0, 200.0, 300.0]
    })

    # Test Inner Join
    inner_joined = join_datasets({"df1": df1, "df2": df2}, join_key="id", how="inner")
    assert inner_joined.height == 2
    assert "id" in inner_joined.columns
    assert "name" in inner_joined.columns
    assert "city" in inner_joined.columns
    assert "score_2" in inner_joined.columns # Overlapping column renamed

    # Test Left Join
    left_joined = join_datasets({"df1": df1, "df2": df2}, join_key="id", how="left")
    assert left_joined.height == 3
    assert left_joined["name"].to_list() == ["Alice", "Bob", "Charlie"]


import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.anyio
async def test_healthz_endpoint():
    """
    Test the Render /healthz probe endpoint.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/healthz")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "process_ram_mb" in data
        assert data["uptime"] == "ok"


