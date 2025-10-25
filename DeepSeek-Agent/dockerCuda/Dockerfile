# Multi-stage Dockerfile for DeepSeek-Agent
# Optimized for Apple Silicon (ARM64) and compatible with Linux

# Build stage - use slim image for smaller intermediate layers
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheels
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CPU support (works on both ARM64 and x86_64)
# Apple Silicon users can use this and utilize MPS acceleration
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

LABEL maintainer="DeepSeek-Agent Contributors"
LABEL description="DeepSeek-Agent: Vision-Based Screen Automation with Long-Term Memory"
LABEL version="1.0.0"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/app/cache/huggingface \
    HF_HOME=/app/cache/huggingface \
    TOKENIZERS_PARALLELISM=true

# Create necessary directories
RUN mkdir -p /app/cache/huggingface \
    && mkdir -p /app/logs/screenshots \
    && mkdir -p /app/logs/tokens \
    && mkdir -p /app/logs/metadata \
    && mkdir -p /app/logs/action_logs \
    && mkdir -p /app/checkpoints/projector_stage_a \
    && mkdir -p /app/checkpoints/agent_lora \
    && mkdir -p /app/data

# Copy application code
COPY . .

# Create a non-root user for security (optional, comment out if not needed)
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Default entrypoint - can be overridden
ENTRYPOINT ["python"]
CMD ["scripts/run_agent.py"]
