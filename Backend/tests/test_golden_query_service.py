"""
Tests for Dataherald-inspired Golden Query Context Store and Service.
"""
import uuid
import asyncio
import pytest
from app.db import init_db
from app.services.golden_query_service import GoldenQueryService, GoldenQuery
from app.services.duckdb_engine import DuckDBSecurityError


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_save_and_retrieve_golden_query():
    task_id = str(uuid.uuid4())
    question = "What is the average sales by region?"
    sql = "SELECT region, AVG(sales) as avg_sales FROM dataset GROUP BY region"

    saved = GoldenQueryService.save_golden_query(
        task_id=task_id,
        question=question,
        sql_query=sql,
        description="Core Regional KPI",
    )

    assert saved.id is not None
    assert saved.task_id == task_id
    assert saved.question == question
    assert saved.sql_query == sql
    assert saved.description == "Core Regional KPI"

    queries = GoldenQueryService.get_golden_queries(task_id)
    assert len(queries) == 1
    assert queries[0].id == saved.id
    assert queries[0].question == question


def test_save_golden_query_security_validation():
    task_id = str(uuid.uuid4())

    # Mutation query should be blocked by DuckDBSecurityError
    with pytest.raises(DuckDBSecurityError):
        GoldenQueryService.save_golden_query(
            task_id=task_id,
            question="Drop table",
            sql_query="DROP TABLE dataset",
        )

    # Empty question/sql should raise ValueError
    with pytest.raises(ValueError):
        GoldenQueryService.save_golden_query(task_id, "", "SELECT 1")

    with pytest.raises(ValueError):
        GoldenQueryService.save_golden_query(task_id, "Question", "")


def test_find_matching_golden_query():
    task_id = str(uuid.uuid4())
    q1 = "What is the total revenue for 2024?"
    sql1 = "SELECT SUM(revenue) FROM dataset WHERE year = 2024"

    q2 = "Who are the top 5 customers by order count?"
    sql2 = "SELECT customer_name, COUNT(*) as orders FROM dataset GROUP BY 1 ORDER BY orders DESC LIMIT 5"

    GoldenQueryService.save_golden_query(task_id, q1, sql1)
    GoldenQueryService.save_golden_query(task_id, q2, sql2)

    # 1. Exact match (ignoring casing and whitespace)
    match_exact = GoldenQueryService.find_matching_query(task_id, "  WHAT IS THE TOTAL REVENUE FOR 2024?  ")
    assert match_exact is not None
    assert match_exact.sql_query == sql1

    # 2. Token overlap match
    match_overlap = GoldenQueryService.find_matching_query(task_id, "What is total revenue for 2024")
    assert match_overlap is not None
    assert match_overlap.sql_query == sql1

    # 3. Completely unrelated query should return None
    match_none = GoldenQueryService.find_matching_query(task_id, "What is the employee attrition rate in department?")
    assert match_none is None


def test_delete_golden_query():
    task_id = str(uuid.uuid4())
    saved = GoldenQueryService.save_golden_query(
        task_id=task_id,
        question="Count rows",
        sql_query="SELECT COUNT(*) FROM dataset",
    )

    queries_before = GoldenQueryService.get_golden_queries(task_id)
    assert len(queries_before) == 1

    deleted = GoldenQueryService.delete_golden_query(task_id, saved.id)
    assert deleted is True

    queries_after = GoldenQueryService.get_golden_queries(task_id)
    assert len(queries_after) == 0


def test_async_golden_query_crud():
    async def _run():
        task_id = str(uuid.uuid4())
        saved = await GoldenQueryService.save_golden_query_async(
            task_id=task_id,
            question="Async test query",
            sql_query="SELECT 42 as answer",
            description="Async test",
        )
        assert saved.id is not None

        fetched = await GoldenQueryService.get_golden_queries_async(task_id)
        assert len(fetched) == 1
        assert fetched[0].id == saved.id

        del_res = await GoldenQueryService.delete_golden_query_async(task_id, saved.id)
        assert del_res is True

        fetched_empty = await GoldenQueryService.get_golden_queries_async(task_id)
        assert len(fetched_empty) == 0

    asyncio.run(_run())
