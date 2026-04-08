# ── Stage 1: C++26 builder ────────────────────────────────────────────────────
FROM ubuntu:24.04 AS cpp-builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git curl ca-certificates \
    python3.13 python3.13-dev python3-pip \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY CMakeLists.txt cmake/ ./
COPY src/cpp/ src/cpp/

RUN cmake -B cmake-build -S . \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -G Ninja \
    && cmake --build cmake-build --parallel "$(nproc)" \
    && cmake --install cmake-build

# ── Stage 2: Python runtime ──────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed C++ artifacts (nanobind .so already in Python site-packages)
COPY --from=cpp-builder /usr/local /usr/local

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY notebooks/ notebooks/
COPY data/ data/

# Install Python package (editable — no absolute paths)
RUN pip install --no-cache-dir -e ".[dev]"

# Expose JupyterLab port
EXPOSE 8888

ENV PYTHONPATH=/app/src/python
ENV LOG_LEVEL=INFO

# Default: launch JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", \
     "--notebook-dir=/app/notebooks", "--allow-root", \
     "--NotebookApp.token=''", "--NotebookApp.password=''"]
