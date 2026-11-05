# ==============================================================================
# DeepCell Bioimaging & Spatial Biology Platform - Production GPU Container
# Base: NVIDIA CUDA 11.8.0 with cuDNN 8 runtime on Ubuntu 22.04
# ==============================================================================

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies & OpenJDK (for bio-formats / QuPath compatibility)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    git \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set up Python virtual environment
RUN python3.10 -m venv /opt/venv

# Install Python scientific & deep learning stack
WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy application source code
COPY . /app

# Non-root user for security best practices
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Healthcheck for container orchestration (Kubernetes / ECS)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Default entrypoint: launch production FastAPI microservice
CMD ["python", "cli.py", "serve", "--host", "0.0.0.0", "--port", "8000"]

