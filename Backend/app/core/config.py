import os
import sys

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "GetReport API"
    GEMINI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    TYPESAFE_API_KEY: str | None = None
    TYPESAFE_BASE_URL: str = "https://api.typesafe.ai/v1"
    TYPESAFE_MODEL: str = "jev-latest"
    DATABASE_URL: str = "" # Logic: If set, use Postgres. Else, use SQLite.
    REDIS_URL: str = "" # Logic: If set and reachable, use ARQ Redis worker. Else, use native in-memory asyncio task runner.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: str = "60/minute"
    MAX_UPLOAD_SIZE_MB: int = 50
    
    # ─── Environment & Runtime Governance ────────────────────────────────
    ENVIRONMENT: str = "development" # "development", "staging", "production"
    REQUIRE_DURABLE_JOBS: bool = False # If True or in production, queue fails closed if Redis is offline
    ENABLE_CODE_EXECUTION: bool = False # Secure default: arbitrary python execution disabled
    
    # ─── Security ─────────────────────────────────────────────────────────
    # API Key: If set, all endpoints require X-API-Key header.
    # Leave empty to disable auth.
    API_KEY: str = ""
    # In production deployments (e.g. Render blueprint), REQUIRE_AUTH=true is set in env.
    REQUIRE_AUTH: bool = False
    # Secret used for Snowflake vault key derivation. Must be configured for persistent deployments.
    SECRET_KEY: str = ""
    
    # Max length for chat questions (prompt injection mitigation)
    MAX_CHAT_QUESTION_LENGTH: int = 500
    
    # SECURITY WARNING: These defaults are for local development and Vercel deployment.
    # In production, you can override CORS_ORIGINS via environment variable to restrict access.
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://localhost:8080,https://get-report.vercel.app,https://getreport.vercel.app"

    # ─── Analysis Thresholds ─────────────────────────────────────────────
    IQR_LOWER_MULTIPLIER: float = 1.5
    IQR_UPPER_MULTIPLIER: float = 1.5
    CORRELATION_STRONG_THRESHOLD: float = 0.7
    SKEWNESS_THRESHOLD: float = 1.0
    ID_UNIQUENESS_THRESHOLD: float = 0.98

    # ─── Storage ─────────────────────────────────────────────────────────
    STORAGE_TYPE: str = "local" # "local" or "s3"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    AWS_BUCKET_NAME: str = "getreport-uploads"

    # ─── Database ─────────────────────────────────────────────────────────
    DB_DIR: str = "data"  # Directory for SQLite files (relative to Backend/)
    DB_POOL_MIN_SIZE: int = 1
    DB_POOL_MAX_SIZE: int = 10
    MAX_EXCEL_DECOMPRESSED_SIZE_MB: int = 200

    # ─── Memory & Resource Limits ────────────────────────────────────────
    POLARS_MAX_THREADS: int = 2
    MAX_CHARTS_PER_REPORT: int = 8
    CHART_DPI: int = 96

    # ─── PDF Engine ──────────────────────────────────────────────────────
    PDF_ENGINE: str = "typst"  # "typst" (default, ultra-fast, <25MB RAM) or "reportlab" (fallback)
    
    # ─── RAG & Embeddings ────────────────────────────────────────────────
    EMBEDDING_DIMENSION: int = 768  # 768 for Google Gemini (default), 1536 for OpenAI

    # ─── Apache Superset Companion Integration ───────────────────────────
    SUPERSET_URL: str = "http://localhost:8088"
    SUPERSET_ADMIN_USERNAME: str = "admin"
    SUPERSET_ADMIN_PASSWORD: str = "admin"
    SUPERSET_GUEST_TOKEN_SECRET: str = "getreport-superset-secret-key-change-in-prod-2026"
    SUPERSET_DEFAULT_DASHBOARD_ID: str = "1"
    SUPERSET_DB_URL: str = "postgresql://superset:superset_password@localhost:5433/superset"
    SUPERSET_ENABLED: bool = False
    SUPERSET_EXTERNAL_COMPANION_ENABLED: bool = False
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()


# Ensure Polars does not exceed thread limit on multi-core VMs
if "POLARS_MAX_THREADS" not in os.environ:
    os.environ["POLARS_MAX_THREADS"] = str(settings.POLARS_MAX_THREADS)
