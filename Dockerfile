# Dockerfile
# VibeVoice OpenAI API Server
# Multi-architecture support: x86_64 (amd64) and ARM64 (aarch64/DGX Spark)

ARG TARGETARCH=amd64
ARG BASE_IMAGE_AMD64=nvcr.io/nvidia/pytorch:25.01-py3
ARG BASE_IMAGE_ARM64=nvcr.io/nvidia/pytorch:24.08-py3

# Architecture-specific base images
FROM ${BASE_IMAGE_AMD64} AS base-amd64
FROM ${BASE_IMAGE_ARM64} AS base-arm64

# Select base image based on target architecture
FROM base-${TARGETARCH} AS final

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy VibeVoice source and install
COPY ./VibeVoice /app/VibeVoice
RUN cd /app/VibeVoice && pip install --no-cache-dir -e .[tts,asr]

# Install Flash Attention (x86_64 only)
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed, will use SDPA"; \
    fi

# Copy and install application dependencies
COPY ./VibeVoiceDockerOpenaiApiServer/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY ./VibeVoiceDockerOpenaiApiServer/app /app/app

# Create non-root user (optional, for security)
# RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
# USER appuser

# Expose port
EXPOSE 8080

# Environment variable defaults
ENV DEVICE=cuda
ENV ATTN_IMPLEMENTATION=flash_attention_2
ENV ASR_MODEL_PATH=/models/VibeVoice-ASR
ENV TTS_MODEL_PATH=/models/VibeVoice-Realtime-0.5B
ENV VOICES_PATH=/models/voices
ENV ASR_ENABLED=true
ENV TTS_ENABLED=true
ENV API_HOST=0.0.0.0
ENV API_PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
