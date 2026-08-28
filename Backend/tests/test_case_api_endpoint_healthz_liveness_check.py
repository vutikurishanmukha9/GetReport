import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.anyio
async def test_api_healthz_liveness_and_readiness_probe():
    """Verify /health and /api/healthz return status 200 and healthy status indicator."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"
        
        response_api = await client.get("/api/healthz")
        assert response_api.status_code == 200
