from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "GetReport API"
    OPENAI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    DATABASE_URL: str = "" # Logic: If set, use Postgres. Else, use SQLite.
    REDIS_URL: str = "redis://localhost:6379/0" # Default local Redis
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: str = "60/minute"
    MAX_UPLOAD_SIZE_MB: int = 50
    
    # ⚠ SECURITY WARNING: These defaults are for local development ONLY.
    # In production, you MUST override CORS_ORIGINS via environment variable to restrict access.
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://localhost:8080"

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

    # ─── PDF Engine ──────────────────────────────────────────────────────
    PDF_ENGINE: str = "reportlab"  # "reportlab" (local dev) or "weasyprint" (production)
    
    class Config:
        env_file = ".env"

settings = Settings()
