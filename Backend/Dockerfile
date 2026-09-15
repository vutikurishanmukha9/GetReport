# ═══════════════════════════════════════════════════════════════════════════════
# GetReport Backend — Production Dockerfile
# Multi-stage build: Dependencies → Application
# Includes WeasyPrint system deps + Neon DB SSL support.
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Stage 1: Build dependencies ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY Backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements.txt

# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# System dependencies:
#   - Fonts for Typst report rendering: fonts-dejavu-core
#   - Neon DB: ca-certificates (SSL connections)
#   - Shell script utilities: dos2unix
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    ca-certificates \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-built Python packages
COPY --from=builder /install /usr/local

# Copy application code (from Backend/ directory)
COPY Backend/ .

# Create necessary directories for runtime
RUN mkdir -p /app/temp_uploads /app/outputs /app/temp_cache /app/data && \
    chmod -R 777 /app/temp_uploads /app/outputs /app/temp_cache /app/data

# Start script
RUN dos2unix ./start.sh && chmod +x ./start.sh

# §7: Non-root container user — limits blast radius of container escapes
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# ─── Environment ─────────────────────────────────────────────────────────────
ENV PDF_ENGINE=typst
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ─── Health check ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Run both Celery and Granian (Rust ASGI) via script
CMD ["./start.sh"]
