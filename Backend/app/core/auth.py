"""
auth.py
~~~~~~~
API Key authentication and input validation utilities.
Security: VULN-01 (Authentication), VULN-02 (IDOR mitigation via UUID validation).
"""
import re
import logging
import secrets
import time
import threading
from fastapi import HTTPException, Security, Query, WebSocket
from fastapi.security import APIKeyHeader
from app.core.config import settings

logger = logging.getLogger(__name__)

# ─── API Key Authentication & Ephemeral Stream Tickets ─────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE
)

# In-memory single-use stream tickets: ticket -> expiry timestamp (VULN-02: prevent query param API key leakage)
_stream_tickets: dict[str, float] = {}
_ticket_lock = threading.Lock()


def create_stream_ticket(ttl_seconds: int = 60) -> str:
    """
    Generate a cryptographically secure, single-use ticket for streaming/SSE connections.
    Valid for ttl_seconds (default 60s). Avoids leaking sensitive API keys in GET URLs.
    """
    now = time.time()
    ticket = secrets.token_urlsafe(32)
    with _ticket_lock:
        # Clean expired tickets
        expired = [k for k, exp in _stream_tickets.items() if exp < now]
        for k in expired:
            _stream_tickets.pop(k, None)
        _stream_tickets[ticket] = now + ttl_seconds
    return ticket


def consume_stream_ticket(ticket: str | None) -> bool:
    """
    Validate and atomically consume a single-use stream ticket.
    Returns True if valid and not expired, False otherwise.
    """
    if not ticket:
        return False
    now = time.time()
    with _ticket_lock:
        exp = _stream_tickets.pop(ticket, None)
        if exp is not None and exp >= now:
            return True
    return False


if settings.DATABASE_URL and not settings.API_KEY and settings.REQUIRE_AUTH:
    logger.warning("REQUIRE_AUTH is enabled but API_KEY is unset! Endpoints will reject requests.")
elif settings.DATABASE_URL and not settings.API_KEY:
    logger.info("Database is configured (DATABASE_URL) and API_KEY is unset. Public access is enabled.")


async def verify_api_key(
    api_key_header: str = Security(_api_key_header),
    api_key_query: str | None = Query(None, alias="api_key"),
    ticket: str | None = Query(None, alias="ticket"),
) -> None:
    """
    FastAPI dependency that enforces API key or stream ticket authentication.
    
    - If settings.API_KEY is empty/unset AND REQUIRE_AUTH is False → auth is DISABLED.
    - If settings.REQUIRE_AUTH is True but API_KEY is empty → REJECT (fail closed).
    - Validates stream ticket first if provided (single-use, short-lived).
    - If settings.API_KEY is set, accepts valid X-API-Key header or stream ticket.
    - Deprecated: ?api_key= query parameter is accepted with a security warning log.
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

    # 1. Stream ticket validation (preferred for SSE / browser GET requests)
    if ticket and consume_stream_ticket(ticket):
        return

    # 2. Standard X-API-Key Header
    if api_key_header and secrets.compare_digest(api_key_header, settings.API_KEY):
        return

    # 3. Query parameter fallback (deprecated for CWE-598; logs warning)
    if api_key_query and secrets.compare_digest(api_key_query, settings.API_KEY):
        logger.warning(
            "DEPRECATION (VULN-02): API key passed via query parameter. "
            "Use short-lived stream tickets (?ticket=...) or X-API-Key header to avoid leaking keys in logs."
        )
        return

    logger.warning("Unauthorized API request (invalid or missing API key / stream ticket)")
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )


def verify_ws_api_key(api_key: str | None, ticket: str | None = None) -> bool:
    """
    Verify API key or stream ticket for WebSocket connections.
    
    Returns True if authorized, False otherwise.
    Fail-closed only if REQUIRE_AUTH is True but API_KEY is empty.
    """
    if not settings.API_KEY:
        if settings.REQUIRE_AUTH:
            logger.critical("SECURITY: REQUIRE_AUTH is enabled but API_KEY is empty. Rejecting WebSocket.")
            return False
        return True  # Auth disabled
    if ticket and consume_stream_ticket(ticket):
        return True
    return bool(api_key and secrets.compare_digest(api_key, settings.API_KEY))


# ─── Input Validation & Tenant Boundary Isolation ─────────────────────────────

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


def check_task_ownership(task_owner_id: str | None, caller_owner_id: str | None) -> None:
    """
    Enforces tenant boundary isolation (VULN-04 / CWE-285).
    If a task has an assigned owner_id and a caller_owner_id is provided,
    they must match or an HTTP 403 Forbidden is raised.
    """
    if task_owner_id and caller_owner_id and task_owner_id != caller_owner_id:
        logger.warning(
            f"Tenant boundary violation: Caller '{caller_owner_id}' attempted to access "
            f"task owned by '{task_owner_id}'."
        )
        raise HTTPException(
            status_code=403,
            detail="Access denied: You do not have permission to access this resource."
        )
