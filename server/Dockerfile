# =============================================================================
# Mini SOC — Multi-stage Docker Build
# =============================================================================
# Stage 1: Base image with dependencies (cached layer)
# Stage 2: Production image with application code
#
# Build:  docker build -t mini-soc -f server/Dockerfile .
# Run:    docker run -p 8000:8000 mini-soc
# Test:   docker run mini-soc python -m pytest tests/ -v
# Agent:  docker compose up   (starts env + runs agent)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS deps

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Production image
# ---------------------------------------------------------------------------
FROM deps AS production

WORKDIR /app

# Copy OpenEnv root-level files
COPY models.py .
COPY client.py .
COPY __init__.py .
COPY openenv.yaml .

# Copy server package
COPY server/ ./server/

# Copy agent and test scripts
COPY inference.py .
COPY run_agent.py .
COPY tests/ ./tests/

# Create outputs directory
RUN mkdir -p outputs/logs outputs/evals

# Expose FastAPI server port
EXPOSE 8000

# Health check — verifies the environment API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: start the FastAPI SOC environment server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
