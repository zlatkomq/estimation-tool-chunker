# Stage 1: Builder (Debian-based)
FROM python:3.11-slim-bookworm AS builder
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-cache --no-install-project --extra cu121
RUN uv pip install pikepdf

# Stage 2: Final image (CUDA-enabled)
FROM nvidia/cuda:12.3.1-base-ubuntu22.04  # Changed base for CUDA support
WORKDIR /app

# Install essential GPU dependencies
RUN apt-get update && apt-get install -y \
    ocl-icd-opencl-dev \
    libgl1 \
    python3.11 \
    python3.11-distutils \
    nvidia-utils-535 \
    cuda-12-3 \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder
COPY --from=builder /app /app
COPY src /app/src

# Set up Python and environment
RUN ln -s /usr/bin/python3.11 /usr/bin/python
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH="/app/.venv/bin:$PATH"

CMD [ "python", "-m", "src.main" ]

