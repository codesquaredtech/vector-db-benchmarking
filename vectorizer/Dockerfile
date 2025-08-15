# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.6.0-base-ubuntu22.04
WORKDIR /vectorizer

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    build-essential \
    g++ \
    curl \
    ca-certificates \
    git \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && mv /root/.local/bin/uv /usr/local/bin/uv
ENV UV_CACHE_DIR=/root/.cache/uv

COPY ./requirements.txt /vectorizer/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

COPY main.py /vectorizer/main.py
CMD ["python3", "main.py"]
