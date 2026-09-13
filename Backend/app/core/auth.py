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

if settings.DATABASE_URL and not settings.API_KEY:
    logger.warning("Production database is configured (DATABASE_URL) but API_KEY is unset! Endpoints are open to public access.")


async def verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    """
    FastAPI dependency that enforces API key authentication.
    
    - If settings.API_KEY is empty/unset AND no DATABASE_URL → auth is DISABLED (dev mode).
    - If settings.DATABASE_URL is set but API_KEY is empty → REJECT (fail closed in production).
    - Otherwise, the request must include a valid X-API-Key header.
    """
    if not settings.API_KEY:
        if settings.DATABASE_URL:
            # §2: Fail closed — production database is configured but no API_KEY set
            logger.critical("SECURITY: DATABASE_URL is set but API_KEY is empty. Rejecting all requests.")
            raise HTTPException(
                status_code=503,
                detail="Service misconfigured: authentication is required but not configured.",
            )
        # Auth disabled (development mode — no DATABASE_URL)
        return

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
    WebSockets can't use standard headers easily, so key is passed via protocol message.
    
    Returns True if authorized, False otherwise.
    Fail-closed in production if DATABASE_URL is set but API_KEY is empty.
    """
    if not settings.API_KEY:
        if settings.DATABASE_URL:
            logger.critical("SECURITY: DATABASE_URL is set but API_KEY is empty. Rejecting WebSocket.")
            return False
        return True  # Auth disabled (development mode)
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
