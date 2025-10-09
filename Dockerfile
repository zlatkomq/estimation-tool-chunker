# Stage 1: Builder (Debian-based)
FROM python:3.11-slim-bookworm AS builder
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-cache --no-install-project --extra cu121
RUN uv pip install pikepdf

# Stage 2: Final image (CUDA-enabled)
# Changed base for CUDA support
FROM nvidia/cuda:12.3.1-base-ubuntu22.04
WORKDIR /app

# Install Python and essential dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    ocl-icd-opencl-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder
COPY --from=builder /app /app
COPY src /app/src

# Set up Python and environment
RUN ln -s /usr/bin/python3.11 /usr/bin/python
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH="/app/.venv/bin:$PATH"

CMD [ "python", "-m", "src.main" ]

