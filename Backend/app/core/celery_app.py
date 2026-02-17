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
        # Resilience: Don't ack tasks until AFTER they complete.
        # If a worker dies mid-task, Redis will re-queue the task to another worker.
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        # Give tasks 10 minutes before Redis considers them lost
        broker_transport_options={'visibility_timeout': 600},
        # Windows Support: 'solo' pool is likely needed for loose check
        # But we let the worker command decide that via -P flag
    )
    
    # Optional: If on Windows and no Redis, fallback to eager?
    # For now, let's keep it strict as per critique.
    
    # Safe Redis Check (Smart Fallback)
    # If Redis is not running, we switch to 'task_always_eager' (synchronous mode)
    # This prevents the API from crashing during development if infrastructure is missing.
    try:
        import redis
        client = redis.from_url(redis_url, socket_connect_timeout=1)
        client.ping()
        print(f"[Celery] Connected to Redis at {redis_url}")
    except Exception as e:
        print(f"[Celery] WARNING: Redis not available ({e}). Running in SYNC mode (task_always_eager=True).")
        app.conf.update(
            task_always_eager=True,
            task_eager_propagates=True
        )

    return app

celery_app = get_celery_app()
