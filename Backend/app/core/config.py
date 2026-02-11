from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "GetReport API"
    OPENAI_API_KEY: str = ""
    DATABASE_URL: str = "" # Logic: If set, use Postgres. Else, use SQLite.
    REDIS_URL: str = "redis://localhost:6379/0" # Default local Redis
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: str = "60/minute"
    MAX_UPLOAD_SIZE_MB: int = 50
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://localhost:8080"
    
    class Config:
        env_file = ".env"

settings = Settings()
