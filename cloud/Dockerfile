FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

LABEL org.opencontainers.image.source=https://github.com/aipachakutiqwan/le-wm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    python3 \
    python3-pip \
    libegl1 \
    libgl1 \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Venv lives at /opt/venv — outside /app so the dev repo mount never touches it
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Copy dependency files first — layer cache: only reruns if pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock ./

# Install all dependencies but not the project itself
RUN uv sync --frozen --no-dev --no-install-project

# Copy project source and install the project package
COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/opt/venv/bin:$PATH"

# Dataset volume is mounted here at runtime (Modal / S3 / GCS)
ENV STABLEWM_HOME=/stablewm-home

# No ENTRYPOINT — Modal overrides CMD but not ENTRYPOINT when launching containers
# from a registry image, so any ENTRYPOINT here gets prepended to Modal's own
# `python -m modal._container_entrypoint ...` bootstrap and breaks startup.
# Local docker callers pass `python <script>` explicitly (see devtools.py).
CMD ["python", "train.py"]
