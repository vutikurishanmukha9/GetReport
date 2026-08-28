import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.anyio
async def test_security_headers_middleware_enforcement():
    """Verify security hardening headers: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        headers = response.headers
        assert headers.get("x-content-type-options") == "nosniff"
        assert headers.get("x-frame-options") == "DENY"
        assert "mode=block" in headers.get("x-xss-protection", "")
