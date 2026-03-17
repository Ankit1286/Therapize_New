# Multi-stage build for lean production image
# Stage 1: Install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for asyncpg + httpx
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in isolated layer
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Startup: run migrations then start server
CMD ["sh", "-c", "python scripts/migrate.py && uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2"]
