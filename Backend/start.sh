#!/bin/bash
set -e

# Memory-constrained container tuning (limits Polars threads and trims glibc arenas)
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-2}
export MALLOC_TRIM_THRESHOLD_=65536

# Check Redis connectivity before attempting to start ARQ Worker
USE_ARQ=false
if [ -n "$REDIS_URL" ] && [ "$REDIS_URL" != "redis://localhost:6379/0" ] && [ "$REDIS_URL" != "redis://localhost:6379" ]; then
    echo "REDIS_URL detected. Verifying connectivity to Redis server..."
    if python -c "
import os, sys
try:
    import redis
    url = os.environ.get('REDIS_URL', '').strip()
    if not url:
        sys.exit(1)
    r = redis.from_url(url, socket_connect_timeout=2.0, socket_timeout=2.0)
    r.ping()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        USE_ARQ=true
    else
        echo "WARNING: REDIS_URL is configured but Redis server is unreachable or timed out."
        echo "Skipping ARQ worker pool. Background tasks will execute via native in-memory asyncio task runner."
    fi
fi

if [ "$USE_ARQ" = true ]; then
    echo "Redis connection verified. Starting Lean ARQ Worker Pool (~25MB RAM footprint)..."
    arq app.tasks_arq.WorkerSettings &
else
    echo "No reachable Redis service detected; background tasks will execute via native in-memory asyncio task runner."
fi

# Start FastAPI Application via Granian (Rust ASGI Server — 2.5x throughput vs Uvicorn)
echo "Starting FastAPI Server (Granian Rust ASGI)..."
exec granian --interface asgi --host 0.0.0.0 --port "${PORT:-8000}" --workers "${WEB_CONCURRENCY:-1}" app.main:app
