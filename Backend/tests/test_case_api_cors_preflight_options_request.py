import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.anyio
async def test_api_cors_preflight_headers():
    """Verify OPTIONS preflight request returns proper CORS allowed origins."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.options(
            "/api/analyze",
            headers={
                "Origin": "https://get-report.vercel.app",
                "Access-Control-Request-Method": "POST"
            }
        )
        # Should return 200 OK for preflight
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
