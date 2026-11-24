"""
test_analytics_security_remediation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for SEC-01 (Analytics IDOR/BOLA), SEC-02 (CORS Regex hardening),
and SEC-03 (Production Secret fail-closed validation).
"""
import re
import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException

from app.api.routes.analytics import _get_authorized_analytics_job
from app.core.config import settings
from app.main import validate_production_secrets, CORS_ORIGIN_REGEX


def test_cors_regex_strictness():
    """SEC-02: Ensure CORS origin regex does not permit arbitrary Render subdomains."""
    pattern = re.compile(CORS_ORIGIN_REGEX)

    # Valid origins
    assert pattern.match("https://getreport.onrender.com") is not None
    assert pattern.match("https://get-report.vercel.app") is not None
    assert pattern.match("https://get-report-git-main-user.vercel.app") is not None

    # Blocked attacker-controlled origins
    assert pattern.match("https://evil-tenant.onrender.com") is None
    assert pattern.match("https://attacker-getreport.onrender.com") is None
    assert pattern.match("https://evil.com") is None
    assert pattern.match("http://getreport.onrender.com") is None  # HTTP not allowed


def test_production_secret_fail_closed(monkeypatch):
    """SEC-03: Ensure startup checks halt if secrets are empty or default in production."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "REQUIRE_AUTH", True)
    monkeypatch.setattr(settings, "SECRET_KEY", "")

    with pytest.raises(RuntimeError, match="PRODUCTION BOOT REFUSED: A secure, cryptographically random SECRET_KEY"):
        validate_production_secrets()

    # Default Superset secret should also raise in production if enabled
    monkeypatch.setattr(settings, "SECRET_KEY", "prod-secret-1234567890123456")
    monkeypatch.setattr(settings, "SUPERSET_ENABLED", True)
    monkeypatch.setattr(settings, "SUPERSET_GUEST_TOKEN_SECRET", "getreport-superset-secret-key-change-in-prod-2026")
    with pytest.raises(RuntimeError, match="PRODUCTION BOOT REFUSED: SUPERSET_GUEST_TOKEN_SECRET"):
        validate_production_secrets()

    # Valid non-default secrets should succeed
    monkeypatch.setattr(settings, "SUPERSET_GUEST_TOKEN_SECRET", "strong-custom-production-secret-999")
    validate_production_secrets()  # Should not raise


from app.services.task_manager import title_task_manager
from unittest.mock import AsyncMock

@pytest.mark.anyio
async def test_analytics_idor_invalid_uuid():
    """SEC-01: Ensure invalid task IDs fail immediately with 400 Bad Request."""
    req = MagicMock()
    req.state.owner_id = "tenant-1"

    with pytest.raises(HTTPException) as exc:
        await _get_authorized_analytics_job(req, "malicious-id-or-path/../")
    assert exc.value.status_code == 400


@pytest.mark.anyio
async def test_analytics_idor_mismatched_tenant(monkeypatch):
    """SEC-01: Ensure cross-tenant access returns 403 Forbidden."""
    valid_uuid = "12345678-1234-5678-1234-567812345678"
    mock_job = MagicMock()
    mock_job.owner_id = "victim-tenant"

    monkeypatch.setattr(title_task_manager, "get_job_async", AsyncMock(return_value=mock_job))

    req = MagicMock()
    req.state.owner_id = "attacker-tenant"

    with pytest.raises(HTTPException) as exc:
        await _get_authorized_analytics_job(req, valid_uuid)
    assert exc.value.status_code == 403

