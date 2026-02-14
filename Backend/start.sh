#!/bin/bash
set -e

# Start Celery Worker in the background
echo "Starting Celery Worker..."
celery -A app.core.celery_app worker --loglevel=info --concurrency=2 &

# Start FastAPI Application in the foreground
echo "Starting FastAPI Server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
