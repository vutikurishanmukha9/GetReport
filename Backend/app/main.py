from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from app.core.config import settings
from app.core.limiter import limiter
from app.core.security_headers import SecurityHeadersMiddleware
from app.core.request_id import RequestIDMiddleware

from contextlib import asynccontextmanager

from app.api import endpoints
from app.db import init_db, close_db, init_async_db, close_async_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        init_db()  # Run Sync Schema Init
        close_db() # Immediately close the sync pool to prevent connection leaking in web workers!
        await init_async_db() # Init Async Pool
    except Exception as e:
        import logging
        logging.getLogger("uvicorn").error(f"DATABASE INIT FAILED: {e}")

    # Startup storage purge (VULN-01: clean up stale files from previous server runs)
    try:
        from app.services.cleanup import cleanup_old_files
        cleanup_old_files("temp_uploads", max_age_seconds=86400)
        cleanup_old_files("outputs", max_age_seconds=86400)
    except Exception as e:
        import logging
        logging.getLogger("uvicorn").warning(f"Startup storage cleanup failed: {e}")

    yield
    # Shutdown
    await close_async_db()
    close_db()

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

# Rate Limiting
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Security Headers (VULN-10)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)

# Set all CORS enabled origins (with production-aware guard)
_cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]

# Safety: warn if localhost origins are active alongside a real DATABASE_URL
import logging as _logging
_cors_logger = _logging.getLogger("cors")
if settings.DATABASE_URL and any("localhost" in o for o in _cors_origins):
    _cors_logger.warning(
        "CORS allows localhost origins while DATABASE_URL is set (production?). "
        "Set CORS_ORIGINS env var to restrict origins in production."
    )

# Security: warn if default static SECRET_KEY is active in production
if settings.DATABASE_URL or settings.API_KEY:
    if settings.SECRET_KEY == "getreport-vault-secret-key-32b":
        _logging.getLogger("security").critical(
            "CRITICAL SECURITY WARNING: SECRET_KEY is using the default hardcoded repository secret while API_KEY/DATABASE_URL is active. "
            "Set a cryptographically random SECRET_KEY environment variable for production deployments."
        )

# Ensure Vercel production and preview domains are permitted
if "https://get-report.vercel.app" not in _cors_origins and "*" not in _cors_origins:
    _cors_origins.append("https://get-report.vercel.app")
if "https://getreport.vercel.app" not in _cors_origins and "*" not in _cors_origins:
    _cors_origins.append("https://getreport.vercel.app")

_allow_all = "*" in _cors_origins

# CORS-aware Rate Limit Handler (ensures 429 carries CORS headers so browser shows 429 instead of CORS error)
def _cors_rate_limit_handler(request, exc):
    response = _rate_limit_exceeded_handler(request, exc)
    origin = request.headers.get("origin")
    if origin and (_allow_all or origin in _cors_origins or "vercel.app" in origin or "onrender.com" in origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
    return response

app.add_exception_handler(RateLimitExceeded, _cors_rate_limit_handler)

# CORS Middleware (Restricts Vercel preview domains strictly to GetReport prefixes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else _cors_origins,
    # §3: Scoped to project owner's Vercel preview deploys only (not arbitrary forks)
    allow_origin_regex=None if _allow_all else r"https://.*\.onrender\.com|https://get-report.*\.vercel\.app",
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(endpoints.router, prefix="/api", tags=["api"])

@app.get("/health")
def health_check():
    return {"status": "healthy", "project": settings.PROJECT_NAME}

@app.get("/")
def root():
    return {"message": "Welcome to GetReport API"}

