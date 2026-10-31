# syntax=docker/dockerfile:1
# CPU image. For NVIDIA GPUs build with:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 -t censorbot:cuda .
# and run with --gpus all.
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchaudio --index-url ${TORCH_INDEX}

WORKDIR /opt/censorbot
COPY pyproject.toml README.md censor.py ./
COPY censorbot ./censorbot
RUN pip install ".[separation]"

WORKDIR /app
ENTRYPOINT ["censorbot"]
CMD ["--help"]
