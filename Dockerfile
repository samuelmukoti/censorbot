# syntax=docker/dockerfile:1

# Multi-platform build with improved acceleration support
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Base stage with common dependencies
FROM ubuntu:22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    build-essential \
    pkg-config \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/

# Apple Silicon build stage (ARM64)
FROM base AS apple_silicon
RUN echo "Building for Apple Silicon (ARM64)"

# Install CPU-optimized PyTorch for Apple Silicon
RUN pip3 install --no-cache-dir \
    torch>=2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install MLX for Metal acceleration (Apple Silicon only)
RUN pip3 install --no-cache-dir \
    mlx>=0.4.0 \
    mlx-whisper>=0.4.0

# Install other requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# NVIDIA GPU build stage (AMD64 with CUDA)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS nvidia
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    build-essential \
    pkg-config \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt /app/

# Install CUDA-enabled PyTorch
RUN pip3 install --no-cache-dir \
    torch>=2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# CPU-only build stage (AMD64 without GPU)
FROM base AS cpu_only
RUN echo "Building for CPU-only (AMD64)"

# Install CPU-only PyTorch
RUN pip3 install --no-cache-dir \
    torch>=2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Final stage - lightweight with runtime detection
FROM base AS final

# Create cache directory for subliminal
RUN mkdir -p /root/.cache/subliminal

# Install base requirements (lightweight, platform-agnostic)
RUN pip3 install --no-cache-dir \
    faster-whisper>=1.0.0 \
    "numpy>=1.24.0,<2.0.0" \
    pysrt>=1.1.2 \
    subliminal==2.1.0 \
    babelfish>=0.6.0 \
    ffmpeg-python>=0.2.0 \
    chardet>=5.0.0 \
    tqdm>=4.65.0 \
    requests>=2.31.0 \
    guessit>=3.7.1 \
    click>=8.0.0 \
    python-opensubtitles==0.6.dev0

# Install PyTorch (will auto-detect platform at runtime)
RUN pip3 install --no-cache-dir torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu || \
    pip3 install --no-cache-dir torch>=2.2.0

# Conditionally install MLX for ARM64
RUN if [ "$(uname -m)" = "aarch64" ] || [ "$(uname -m)" = "arm64" ]; then \
        pip3 install --no-cache-dir mlx>=0.4.0 mlx-whisper>=0.4.0 || true; \
    fi

# Copy application code
COPY censor.py /app/
COPY badwords.txt /app/
RUN chmod +x /app/censor.py

# Create output directory
RUN mkdir -p /app/output

# Set Python path
ENV PYTHONPATH=/usr/local/lib/python3.10/dist-packages

WORKDIR /app

# Default entrypoint
ENTRYPOINT ["python3", "/app/censor.py"]
CMD ["--help"] 