import os
from celery import Celery
from app.core.config import settings

def get_celery_app() -> Celery:
    # Use REDIS_URL from env, or default to localhost
    # If using Docker, this would be 'redis://redis:6379/0'
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # If we are in "Hybrid DB" mode (SQLite default), we might not have Redis.
    # In that case, we can force eager execution (sync) for dev, 
    # OR the user must install Redis.
    # Given the "Bitter Truth" critique, we assume Production intent = Redis exists.
    
    app = Celery(
        "getreport_tasks",
        broker=redis_url,
        backend=redis_url,
        include=["app.tasks"]  # We will create this module next
    )

    app.conf.update(
        result_expires=86400, # 24 hours
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        # Windows Support: 'solo' pool is likely needed for loose check
        # But we let the worker command decide that via -P flag
    )
    
    # Optional: If on Windows and no Redis, fallback to eager?
    # For now, let's keep it strict as per critique.
    
    return app

celery_app = get_celery_app()
