from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "GetReport API"
    OPENAI_API_KEY: str = ""
    DATABASE_URL: str = "" # Logic: If set, use Postgres. Else, use SQLite.
    
    class Config:
        env_file = ".env"

settings = Settings()
