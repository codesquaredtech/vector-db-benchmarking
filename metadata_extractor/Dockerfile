FROM python:3.12

WORKDIR /metadata_extractor

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && mv /root/.local/bin/uv /usr/local/bin/uv
ENV UV_CACHE_DIR=/root/.cache/uv

COPY ./requirements.txt /metadata_extractor/requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

COPY main.py /metadata_extractor/main.py

CMD ["python", "main.py"]
