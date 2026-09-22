"""
Backend/tests/test_typesafe_jev.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for TypeSafe AI Jev System One model integration in GetReport.
Verifies client initialization, payload construction, QueryRouter fast-path,
and SemanticInference domain detection with mock responses and fallback handling.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import polars as pl

from app.services.typesafe_service import TypeSafeService, typesafe_service
from app.services.query_router import QueryRouter
from app.services.semantic_inference import detect_domain


def test_typesafe_service_disabled_by_default():
    """Service correctly identifies disabled state when API key is missing."""
    service = TypeSafeService(api_key=None)
    assert not service.is_enabled
    assert service.system_one_sync("test state", {}) is None


def test_typesafe_service_enabled_with_key():
    """Service is enabled when API key is provided."""
    service = TypeSafeService(api_key="ts_test_key_123", model="jev-latest")
    assert service.is_enabled
    assert service.model == "jev-latest"
    assert service.api_key == "ts_test_key_123"


@pytest.mark.asyncio
async def test_typesafe_route_query_async_mock():
    """Tests query routing with mocked TypeSafe Jev response."""
    service = TypeSafeService(api_key="ts_test_key_123")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "answers": {
            "query_type": {
                "choice": "analytical_sql",
                "confidence": 0.96,
                "probabilities": {"analytical_sql": 0.96, "hybrid": 0.04}
            },
            "needs_duckdb": {
                "noul": True,
                "confidence": 0.98
            }
        }
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
        result = await service.route_query_async(
            query="What is the average transaction total per store?",
            available_columns=["store_id", "transaction_total", "customer_id"]
        )

        assert result is not None
        assert result["query_type"] == "analytical_sql"
        assert result["needs_duckdb"] is True
        assert result["confidence"] == 0.96
        assert result["provider"] == "typesafe_jev"

        # Verify POST payload
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["model"] == "jev-latest"
        assert "query_type" in call_kwargs["json"]["questions"]
        assert "needs_duckdb" in call_kwargs["json"]["questions"]


def test_typesafe_classify_domain_sync_mock():
    """Tests domain classification with mocked TypeSafe Jev response."""
    service = TypeSafeService(api_key="ts_test_key_123")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "answers": {
            "domain": {
                "choice": "healthcare",
                "confidence": 0.92
            }
        }
    }

    with patch("httpx.Client.post", return_value=mock_response):
        result = service.classify_domain_sync(
            columns=["patient_mrn", "admission_timestamp", "systolic_bp", "diagnosis_icd10"],
            sample_preview="patient_mrn: 10482, diagnosis_icd10: I10"
        )

        assert result is not None
        assert result["domain"] == "healthcare"
        assert result["confidence"] == 0.92
        assert result["provider"] == "typesafe_jev"


@pytest.mark.asyncio
async def test_query_router_typesafe_fast_path():
    """Tests QueryRouter dispatches to TypeSafe Jev when enabled."""
    router = QueryRouter()
    cols = ["order_id", "total_amount", "customer_name"]

    mock_jev_result = {
        "query_type": "analytical_sql",
        "needs_duckdb": True,
        "confidence": 0.95,
        "provider": "typesafe_jev"
    }

    with patch.object(typesafe_service, "api_key", "ts_mock_key"), \
         patch.object(typesafe_service, "route_query_async", new_callable=AsyncMock, return_value=mock_jev_result):
        
        res = await router.route_query("Compute total sales volume", cols)
        assert res["query_type"] == "analytical_sql"
        assert res["duckdb_sql_needed"] is True


def test_semantic_inference_typesafe_fallback():
    """Tests SemanticInference detect_domain uses Jev when columns have no keyword matches."""
    # Obscure columns with zero dictionary keyword hits
    df = pl.DataFrame({
        "col_alpha": [1, 2, 3],
        "col_beta": [4, 5, 6],
        "col_gamma": ["a", "b", "c"]
    })

    mock_jev_result = {
        "domain": "finance",
        "confidence": 0.88,
        "provider": "typesafe_jev"
    }

    with patch.object(typesafe_service, "api_key", "ts_mock_key"), \
         patch.object(typesafe_service, "classify_domain_sync", return_value=mock_jev_result):
        
        domain_result = detect_domain(df)
        assert domain_result.primary_domain == "finance"
        assert domain_result.confidence == 0.88
        assert "typesafe_jev:finance" in domain_result.matched_keywords
