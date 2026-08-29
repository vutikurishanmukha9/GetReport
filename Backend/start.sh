#!/bin/bash
set -e

# Memory-constrained container tuning (limits Polars threads and trims glibc arenas)
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-2}
export MALLOC_TRIM_THRESHOLD_=65536

# Start Celery Worker with lightweight single-concurrency to prevent multi-fork RAM exhaustion
echo "Starting Celery Worker (Memory Optimized)..."
celery -A app.core.celery_app worker --loglevel=info --concurrency=1 -P solo --max-tasks-per-child=10 --max-memory-per-child=180000 &

# Start FastAPI Application in the foreground
echo "Starting FastAPI Server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
