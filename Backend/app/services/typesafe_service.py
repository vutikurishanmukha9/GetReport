"""
Backend/app/services/typesafe_service.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration service for TypeSafe AI's Jev "System One" Decision Model.
Provides ultra-fast (70-300ms), typed, probabilistic decisions for:
1. Fast Query Routing (Analytical SQL vs Graph Local vs Global vs Hybrid)
2. Zero-Shot Dataset Domain Classification
3. Contextual PII Sensitivity Gating
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class TypeSafeService:
    """
    Client and decision gateway for TypeSafe AI's Jev model.
    Calls the https://api.typesafe.ai/v1/systemone endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.TYPESAFE_API_KEY
        self.base_url = (base_url or settings.TYPESAFE_BASE_URL).rstrip("/")
        self.model = model or settings.TYPESAFE_MODEL or "jev-latest"

    @property
    def is_enabled(self) -> bool:
        """True if a valid TypeSafe API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    async def system_one_async(
        self,
        state: str,
        questions: Dict[str, Any],
        timeout: float = 3.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Asynchronously calls TypeSafe Jev System One endpoint.
        Returns answers dictionary or None on failure.
        """
        if not self.is_enabled:
            return None

        url = f"{self.base_url}/systemone"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "state": state[:4000],  # Bound input length
            "questions": questions,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("answers", data)
                else:
                    logger.warning(
                        "TypeSafe Jev API returned status %d: %s",
                        response.status_code,
                        response.text[:200],
                    )
                    return None
        except Exception as e:
            logger.debug("TypeSafe Jev async request failed or timed out: %s", e)
            return None

    def system_one_sync(
        self,
        state: str,
        questions: Dict[str, Any],
        timeout: float = 3.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Synchronous call to TypeSafe Jev System One endpoint for worker tasks.
        """
        if not self.is_enabled:
            return None

        url = f"{self.base_url}/systemone"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "state": state[:4000],
            "questions": questions,
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("answers", data)
                else:
                    logger.warning(
                        "TypeSafe Jev sync API returned status %d: %s",
                        response.status_code,
                        response.text[:200],
                    )
                    return None
        except Exception as e:
            logger.debug("TypeSafe Jev sync request failed or timed out: %s", e)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Domain-Specific Decision Helpers for GetReport
    # ─────────────────────────────────────────────────────────────────────────

    async def route_query_async(
        self,
        query: str,
        available_columns: List[str],
        timeout: float = 2.5,
    ) -> Optional[Dict[str, Any]]:
        """
        Routes user query using Jev's fast System 1 Choice and Noul primitives.
        Returns: {
            "query_type": "analytical_sql" | "graph_local" | "graph_global" | "hybrid",
            "needs_duckdb": bool,
            "confidence": float
        }
        """
        cols_summary = ", ".join(available_columns[:25]) if available_columns else "None"
        state = f"User natural language question: \"{query}\". Available dataset columns: [{cols_summary}]."

        questions = {
            "query_type": {
                "type": "choice",
                "instructions": "Determine the optimal processing strategy for this analytics question.",
                "criteria": {
                    "analytical_sql": "Exact mathematical calculations, aggregations, averages, totals, counts, minimums, maximums, or exact SQL metric queries.",
                    "graph_local": "Questions about specific column relationships, correlations, individual column data quality issues, or direct column definitions.",
                    "graph_global": "High-level dataset summary, overarching trends, general recommendations, or full dataset health overview.",
                    "hybrid": "Broad exploratory questions combining statistical metrics and general interpretations.",
                },
            },
            "needs_duckdb": {
                "type": "noul",
                "instructions": "Does answering this question require executing SQL or calculating mathematical aggregations over data rows?",
            },
        }

        answers = await self.system_one_async(state, questions, timeout=timeout)
        if not answers:
            return None

        qt_answer = answers.get("query_type", {})
        nd_answer = answers.get("needs_duckdb", {})

        # Parse Jev response format (handles both SDK object and raw REST dict)
        chosen_type = (
            qt_answer.get("choice")
            if isinstance(qt_answer, dict)
            else getattr(qt_answer, "choice", "hybrid")
        ) or "hybrid"

        confidence = (
            qt_answer.get("confidence", 0.9)
            if isinstance(qt_answer, dict)
            else getattr(qt_answer, "confidence", 0.9)
        )

        needs_duckdb_val = (
            nd_answer.get("noul")
            if isinstance(nd_answer, dict)
            else getattr(nd_answer, "noul", False)
        )

        return {
            "query_type": chosen_type,
            "needs_duckdb": bool(needs_duckdb_val),
            "confidence": float(confidence) if confidence is not None else 0.9,
            "provider": "typesafe_jev",
        }

    def classify_domain_sync(
        self,
        columns: List[str],
        sample_preview: str = "",
        timeout: float = 3.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Classifies dataset domain synchronously using Jev Choice primitive.
        """
        cols_str = ", ".join(columns[:30])
        state = f"Dataset column headers: {cols_str}. Sample data: {sample_preview[:200]}"

        questions = {
            "domain": {
                "type": "choice",
                "instructions": "Which industry or data domain best describes this dataset?",
                "criteria": {
                    "sales_ecommerce": "Orders, purchases, customers, revenue, products, transactions, prices, shipping",
                    "healthcare": "Patients, medical records, diagnosis, disease, clinical trials, prescriptions, hospitals",
                    "finance": "Accounts, balances, ledger, bank transactions, credit, debit, taxes, stocks",
                    "hr_employee": "Employees, staff, salaries, payroll, departments, hiring, appraisals",
                    "education": "Students, courses, grades, marks, exams, teachers, school enrollment",
                    "logistics": "Shipments, routes, tracking, warehouse inventory, freight, deliveries",
                    "supply_chain": "Procurement, vendors, lead times, inventory reorder, suppliers",
                    "iot_sensor": "Sensors, devices, telemetry, temperature, humidity, voltage readings",
                    "general": "General or mixed domain data not clearly fitting above categories",
                },
            }
        }

        answers = self.system_one_sync(state, questions, timeout=timeout)
        if not answers:
            return None

        domain_ans = answers.get("domain", {})
        chosen_domain = (
            domain_ans.get("choice")
            if isinstance(domain_ans, dict)
            else getattr(domain_ans, "choice", "general")
        ) or "general"

        confidence = (
            domain_ans.get("confidence", 0.85)
            if isinstance(domain_ans, dict)
            else getattr(domain_ans, "confidence", 0.85)
        )

        return {
            "domain": chosen_domain,
            "confidence": float(confidence) if confidence is not None else 0.85,
            "provider": "typesafe_jev",
        }


# Global singleton instance
typesafe_service = TypeSafeService()
