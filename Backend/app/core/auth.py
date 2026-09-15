"""
auth.py
~~~~~~~
API Key authentication and input validation utilities.
Security: VULN-01 (Authentication), VULN-02 (IDOR mitigation via UUID validation).
"""
import re
import logging
from fastapi import HTTPException, Security, Query, WebSocket
from fastapi.security import APIKeyHeader
from app.core.config import settings

logger = logging.getLogger(__name__)

# ─── API Key Authentication ──────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE
)

import secrets

if settings.DATABASE_URL and not settings.API_KEY and settings.REQUIRE_AUTH:
    logger.warning("REQUIRE_AUTH is enabled but API_KEY is unset! Endpoints will reject requests.")
elif settings.DATABASE_URL and not settings.API_KEY:
    logger.info("Database is configured (DATABASE_URL) and API_KEY is unset. Public access is enabled.")


async def verify_api_key(
    api_key_header: str = Security(_api_key_header),
    api_key_query: str | None = Query(None, alias="api_key"),
) -> None:
    """
    FastAPI dependency that enforces API key authentication.
    
    - If settings.API_KEY is empty/unset AND REQUIRE_AUTH is False → auth is DISABLED.
    - If settings.REQUIRE_AUTH is True but API_KEY is empty → REJECT (fail closed).
    - If settings.API_KEY is set, the request must include a valid X-API-Key header
      or ?api_key= query parameter (for browser EventSource / SSE compatibility).
    """
    if not settings.API_KEY:
        if settings.REQUIRE_AUTH:
            # Fail closed only when explicit REQUIRE_AUTH is configured
            logger.critical("SECURITY: REQUIRE_AUTH is enabled but API_KEY is empty. Rejecting all requests.")
            raise HTTPException(
                status_code=503,
                detail="Service misconfigured: authentication is required but not configured.",
            )
        # Auth disabled
        return

    api_key = api_key_header or api_key_query
    if not api_key or not secrets.compare_digest(api_key, settings.API_KEY):
        logger.warning("Unauthorized API request (invalid or missing API key)")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def verify_ws_api_key(api_key: str | None) -> bool:
    """
    Verify API key for WebSocket connections.
    
    Returns True if authorized, False otherwise.
    Fail-closed only if REQUIRE_AUTH is True but API_KEY is empty.
    """
    if not settings.API_KEY:
        if settings.REQUIRE_AUTH:
            logger.critical("SECURITY: REQUIRE_AUTH is enabled but API_KEY is empty. Rejecting WebSocket.")
            return False
        return True  # Auth disabled
    return bool(api_key and secrets.compare_digest(api_key, settings.API_KEY))


# ─── Input Validation ────────────────────────────────────────────────────────

def validate_task_id(task_id: str) -> str:
    """
    Validate that task_id is a proper UUID format.
    Prevents enumeration and injection via malformed IDs.
    """
    if not UUID_PATTERN.match(task_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid task ID format"
        )
    return task_id
