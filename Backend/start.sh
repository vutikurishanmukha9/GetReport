#!/bin/bash
set -e

# Memory-constrained container tuning (limits Polars threads and trims glibc arenas)
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-2}
export MALLOC_TRIM_THRESHOLD_=65536

# Start Lean ARQ Worker (~25MB RAM footprint) if Redis is configured
if [ -n "$REDIS_URL" ]; then
    echo "Starting Lean ARQ Worker Pool (~25MB RAM footprint)..."
    arq app.tasks_arq.WorkerSettings &
else
    echo "No REDIS_URL configured; background tasks will execute via native in-memory asyncio task runner."
fi

# Start FastAPI Application via Granian (Rust ASGI Server — 2.5x throughput vs Uvicorn)
echo "Starting FastAPI Server (Granian Rust ASGI)..."
exec granian --interface asgi --host 0.0.0.0 --port 8000 --workers 1 --threads 2 app.main:app
