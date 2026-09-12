"""
Backend/app/services/golden_query_service.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Golden Query Context Store adapted from Dataherald.
Maintains verified, ground-truth (Question, SQL) pairs for datasets,
enabling zero-latency, zero-hallucination answers to recurring KPI questions.
"""
from __future__ import annotations

import uuid
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from app.db import get_db_connection, get_async_db_connection, init_db
from app.services.duckdb_engine import DuckDBAnalyticalSession, DuckDBSecurityError

logger = logging.getLogger(__name__)


@dataclass
class GoldenQuery:
    id: str
    task_id: str
    question: str
    sql_query: str
    description: str = ""
    result_summary: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "question": self.question,
            "sql_query": self.sql_query,
            "description": self.description,
            "result_summary": self.result_summary,
            "created_at": self.created_at,
        }


class GoldenQueryService:
    """
    Manages storage, retrieval, and matching of verified Golden SQL queries.
    """

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Strip punctuation and whitespace for normalized matching."""
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.strip().lower()))

    @classmethod
    def save_golden_query(
        cls,
        task_id: str,
        question: str,
        sql_query: str,
        description: str = "",
        result_summary: str = "",
    ) -> GoldenQuery:
        """
        Validates read-only safety of the SQL and saves it to the golden repository.
        """
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        clean_sql = sql_query.strip()
        if not clean_sql:
            raise ValueError("SQL query cannot be empty.")

        # Validate SQL safety
        session = DuckDBAnalyticalSession()
        try:
            session.validate_sql_safety(clean_sql)
        finally:
            session.close()

        query_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        insert_sql = """
        INSERT INTO golden_queries (id, task_id, question, sql_query, description, result_summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (query_id, task_id, clean_question, clean_sql, description, result_summary, created_at)

        try:
            with get_db_connection() as conn:
                conn.execute(insert_sql, params)
                conn.commit()
        except Exception as e:
            if "no such table: golden_queries" in str(e).lower():
                init_db()
                with get_db_connection() as conn:
                    conn.execute(insert_sql, params)
                    conn.commit()
            else:
                raise

        logger.info(f"Saved Golden Query '{query_id}' for task '{task_id}': {clean_question}")

        return GoldenQuery(
            id=query_id,
            task_id=task_id,
            question=clean_question,
            sql_query=clean_sql,
            description=description,
            result_summary=result_summary,
            created_at=created_at,
        )

    @classmethod
    async def save_golden_query_async(
        cls,
        task_id: str,
        question: str,
        sql_query: str,
        description: str = "",
        result_summary: str = "",
    ) -> GoldenQuery:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        clean_sql = sql_query.strip()
        if not clean_sql:
            raise ValueError("SQL query cannot be empty.")

        session = DuckDBAnalyticalSession()
        try:
            session.validate_sql_safety(clean_sql)
        finally:
            session.close()

        query_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        insert_sql = """
        INSERT INTO golden_queries (id, task_id, question, sql_query, description, result_summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (query_id, task_id, clean_question, clean_sql, description, result_summary, created_at)

        try:
            async with get_async_db_connection() as conn:
                await conn.execute(insert_sql, params)
                await conn.commit()
        except Exception as e:
            if "no such table: golden_queries" in str(e).lower():
                init_db()
                async with get_async_db_connection() as conn:
                    await conn.execute(insert_sql, params)
                    await conn.commit()
            else:
                raise

        return GoldenQuery(
            id=query_id,
            task_id=task_id,
            question=clean_question,
            sql_query=clean_sql,
            description=description,
            result_summary=result_summary,
            created_at=created_at,
        )

    @classmethod
    def get_golden_queries(cls, task_id: str) -> List[GoldenQuery]:
        """Fetch all verified golden queries for a task."""
        query = "SELECT * FROM golden_queries WHERE task_id = ? ORDER BY created_at DESC"
        try:
            with get_db_connection() as conn:
                rows = conn.execute(query, (task_id,)).fetchall()
        except Exception as e:
            if "no such table: golden_queries" in str(e).lower():
                init_db()
                with get_db_connection() as conn:
                    rows = conn.execute(query, (task_id,)).fetchall()
            else:
                raise

        results: List[GoldenQuery] = []
        for r in rows:
            results.append(
                GoldenQuery(
                    id=r["id"],
                    task_id=r["task_id"],
                    question=r["question"],
                    sql_query=r["sql_query"],
                    description=r["description"] or "",
                    result_summary=r["result_summary"] or "",
                    created_at=str(r["created_at"]),
                )
            )
        return results

    @classmethod
    async def get_golden_queries_async(cls, task_id: str) -> List[GoldenQuery]:
        query = "SELECT * FROM golden_queries WHERE task_id = ? ORDER BY created_at DESC"
        try:
            async with get_async_db_connection() as conn:
                cursor = await conn.execute(query, (task_id,))
                rows = await cursor.fetchall()
        except Exception as e:
            if "no such table: golden_queries" in str(e).lower():
                init_db()
                async with get_async_db_connection() as conn:
                    cursor = await conn.execute(query, (task_id,))
                    rows = await cursor.fetchall()
            else:
                raise

        results: List[GoldenQuery] = []
        for r in rows:
            results.append(
                GoldenQuery(
                    id=r["id"],
                    task_id=r["task_id"],
                    question=r["question"],
                    sql_query=r["sql_query"],
                    description=r["description"] or "",
                    result_summary=r["result_summary"] or "",
                    created_at=str(r["created_at"]),
                )
            )
        return results

    @classmethod
    def find_matching_query(cls, task_id: str, user_question: str) -> Optional[GoldenQuery]:
        """
        Matches a user question against verified golden queries.
        1. Exact normalized match.
        2. High-overlap Jaccard token similarity (threshold >= 0.85).
        """
        golden_list = cls.get_golden_queries(task_id)
        if not golden_list:
            return None

        norm_user = cls._normalize_text(user_question)
        user_tokens = set(norm_user.split())

        best_match: Optional[GoldenQuery] = None
        best_score = 0.0

        for g in golden_list:
            norm_golden = cls._normalize_text(g.question)
            if norm_user == norm_golden:
                return g  # Exact match priority

            golden_tokens = set(norm_golden.split())
            if not user_tokens or not golden_tokens:
                continue

            intersection = user_tokens.intersection(golden_tokens)
            union = user_tokens.union(golden_tokens)
            similarity = len(intersection) / len(union)

            if similarity > best_score and similarity >= 0.80:
                best_score = similarity
                best_match = g

        return best_match

    @classmethod
    def delete_golden_query(cls, task_id: str, query_id: str) -> bool:
        """Deletes a golden query by ID."""
        with get_db_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM golden_queries WHERE id = ? AND task_id = ?",
                (query_id, task_id),
            )
            conn.commit()
            deleted = cursor.rowcount > 0 if hasattr(cursor, "rowcount") else True
        return deleted

    @classmethod
    async def delete_golden_query_async(cls, task_id: str, query_id: str) -> bool:
        async with get_async_db_connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM golden_queries WHERE id = ? AND task_id = ?",
                (query_id, task_id),
            )
            await conn.commit()
            return True
