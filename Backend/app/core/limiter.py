"""
Rate Limiting Module
Uses slowapi (Token Bucket) to protect API endpoints from abuse.
"""
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.config import settings

logger = logging.getLogger(__name__)

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
# Key function: rate limit per client IP address
limiter = Limiter(
    key_func=get_remote_address,
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
