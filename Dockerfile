FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight Python dependencies (no torch/transformers — uses HF Inference API)
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.9.0" \
    "pydantic-settings>=2.5.0" \
    "langgraph>=0.2.0" \
    "langchain-core>=0.3.0" \
    "qdrant-client>=1.11.0" \
    "huggingface-hub>=0.25.0" \
    "structlog>=24.4.0" \
    "prometheus-client>=0.21.0" \
    "httpx>=0.27.0" \
    "python-dotenv>=1.0.0" \
    "tenacity>=9.0.0" \
    "python-multipart>=0.0.9" \
    "pdfplumber>=0.11.0"

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY data/eval/ data/eval/
COPY tests/ tests/

# Create data directories
RUN mkdir -p data/raw/edgar data/raw/uploads data/processed data/embeddings

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

EXPOSE ${PORT:-8000}

# Default: run the API server (PORT provided by cloud platforms)
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
