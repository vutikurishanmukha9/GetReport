from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.core.config import settings
from app.core.limiter import limiter

from contextlib import asynccontextmanager

from app.api import endpoints
from app.db import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Database
    try:
        init_db()
    except Exception as e:
        # Log critical error but don't crash app immediately (or do if DB is mandatory?)
        # For now, log it clearly.
        import logging
        logging.getLogger("uvicorn").error(f"DATABASE INIT FAILED: {e}")
    yield
    # Shutdown logic (if any)

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

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

@app.get("/debug-init-db")
def debug_init_db():
    try:
        init_db()
        return {"status": "success", "message": "Database initialized manually."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/debug-tables")
def debug_tables():
    from app.db import get_db_connection, settings
    try:
        if settings.DATABASE_URL:
            # Postgres: Query pg_stat_user_tables
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT schemaname, tablename 
                    FROM pg_catalog.pg_tables 
                    WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';
                """)
                tables = cursor.fetchall()
            return {"status": "success", "db_url": settings.DATABASE_URL.split("@")[1] if "@" in settings.DATABASE_URL else "MASKED", "tables": tables}
        else:
            # SQLite: Query sqlite_master
            return {"status": "info", "message": "SQLite mode (Local)"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

