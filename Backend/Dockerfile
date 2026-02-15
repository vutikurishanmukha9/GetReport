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
    -r requirements.txt \
    weasyprint>=62.0

# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# System dependencies:
#   - WeasyPrint: libpango, libcairo, libgdk-pixbuf, fonts
#   - Neon DB:    ca-certificates (SSL connections)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    libcairo2 \
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
RUN mkdir -p /app/temp_uploads /app/outputs /app/temp_cache && \
    chmod -R 777 /app/temp_uploads /app/outputs /app/temp_cache

# ─── Environment ─────────────────────────────────────────────────────────────
ENV PDF_ENGINE=weasyprint
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Start script
RUN dos2unix ./start.sh && chmod +x ./start.sh

# ─── Health check ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Run both Celery and FastAPI via script
CMD ["./start.sh"]
