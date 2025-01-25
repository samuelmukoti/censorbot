# syntax=docker/dockerfile:1

# Use multi-platform base image
ARG CUDA_IMAGE=nvidia/cuda:12.1.0-runtime-ubuntu22.04
FROM --platform=$BUILDPLATFORM ${CUDA_IMAGE} AS cuda_base

FROM --platform=$BUILDPLATFORM ubuntu:22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONPATH=/usr/local/lib/python3/site-packages

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
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install platform-specific dependencies
FROM base AS apple_silicon
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir coremltools

FROM base AS nvidia
COPY --from=cuda_base /usr/local/cuda/ /usr/local/cuda/
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt

# Final stage
FROM base AS final

# Create Python site-packages directory
RUN mkdir -p /usr/local/lib/python3/site-packages/

# Copy installed packages from the appropriate builder
COPY --from=apple_silicon /usr/local/lib/python3.*/site-packages/ /usr/local/lib/python3/site-packages/
COPY --from=nvidia /usr/local/lib/python3.*/site-packages/ /usr/local/lib/python3/site-packages/
COPY --from=nvidia /usr/local/cuda/ /usr/local/cuda/

# Copy application code
COPY censor.py .

# Make the script executable
RUN chmod +x censor.py

# Create cache directory for subliminal
RUN mkdir -p /root/.cache/subliminal

ENTRYPOINT ["python3", "censor.py"] 