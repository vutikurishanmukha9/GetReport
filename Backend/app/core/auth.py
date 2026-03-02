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


async def verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    """
    FastAPI dependency that enforces API key authentication.
    
    - If settings.API_KEY is empty/unset → auth is DISABLED (dev mode).
    - Otherwise, the request must include a valid X-API-Key header.
    """
    if not settings.API_KEY:
        # Auth disabled (development mode)
        return

    if not api_key or api_key != settings.API_KEY:
        logger.warning("Unauthorized API request (invalid or missing API key)")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def verify_ws_api_key(api_key: str | None) -> bool:
    """
    Verify API key for WebSocket connections.
    WebSockets can't use standard headers easily, so key is passed via query param.
    
    Returns True if authorized, False otherwise.
    """
    if not settings.API_KEY:
        return True  # Auth disabled
    return api_key == settings.API_KEY


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
