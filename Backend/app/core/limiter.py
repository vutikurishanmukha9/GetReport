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
    Extract client IP safely.
    Only trusts reverse-proxy headers (CF-Connecting-IP, X-Forwarded-For) if
    the immediate peer is a local, private, or proxy IP address, preventing direct clients
    from forging headers to evade rate limits.
    """
    client_host = request.client.host if request.client else None

    # Check if direct client is a local / private proxy (e.g. Render / Docker / Cloudflare container)
    is_trusted_proxy = False
    if client_host:
        try:
            ip_obj = ipaddress.ip_address(client_host)
            is_trusted_proxy = ip_obj.is_private or ip_obj.is_loopback
        except ValueError:
            is_trusted_proxy = False

    if is_trusted_proxy:
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
