from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.core.config import settings
from app.core.limiter import limiter

from app.api import endpoints

app = FastAPI(title=settings.PROJECT_NAME)

# Rate Limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Set all CORS enabled origins (with production-aware guard)
_cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]

# Safety: warn if localhost origins are active alongside a real DATABASE_URL
import logging as _logging
_cors_logger = _logging.getLogger("cors")
if settings.DATABASE_URL and any("localhost" in o for o in _cors_origins):
    _cors_logger.warning(
        "⚠ CORS allows localhost origins while DATABASE_URL is set (production?). "
        "Set CORS_ORIGINS env var to restrict origins in production."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
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

