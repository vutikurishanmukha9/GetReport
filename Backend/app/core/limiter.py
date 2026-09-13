"""
Rate Limiting Module
Uses slowapi (Token Bucket) to protect API endpoints from abuse.
"""
import ipaddress
import logging
from starlette.requests import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.config import settings

logger = logging.getLogger(__name__)


def _get_real_client_ip(request: Request) -> str:
    """
    Extract client IP safely from trusted proxy headers (Cloudflare CF-Connecting-IP
    or validated X-Forwarded-For). Falls back to request.client.host if absent or invalid.
    """
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        candidate = cf_ip.strip()
        try:
            ipaddress.ip_address(candidate)
            return candidate
        except ValueError:
            pass

    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        candidate = xff.split(",")[0].strip()
        try:
            ipaddress.ip_address(candidate)
            return candidate
        except ValueError:
            pass
    return get_remote_address(request)


# Check if Redis is available
storage_uri = "memory://"
if settings.REDIS_URL:
    try:
        import redis
        client = redis.from_url(settings.REDIS_URL, socket_connect_timeout=1)
        client.ping()
        storage_uri = settings.REDIS_URL
        logger.info(f"Rate Limiter connected to Redis at {settings.REDIS_URL}")
    except Exception as e:
        logger.warning(f"Rate Limiter: Redis not available ({e}). Falling back to memory storage.")
        storage_uri = "memory://"

# Create limiter instance
# Key function: rate limit per real client IP address (proxy-aware)
limiter = Limiter(
    key_func=_get_real_client_ip,
    enabled=settings.RATE_LIMIT_ENABLED,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
    storage_uri=storage_uri,
)

# Endpoint-specific limits (importable constants)
UPLOAD_LIMIT = "10/minute"
ANALYZE_LIMIT = "30/minute"
CHAT_LIMIT = "60/minute"
STATUS_LIMIT = "120/minute"
REPORT_LIMIT = "20/minute"

logger.info(f"Rate limiting {'ENABLED' if settings.RATE_LIMIT_ENABLED else 'DISABLED'}")
