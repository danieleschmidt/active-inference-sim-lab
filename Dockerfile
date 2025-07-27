# Multi-stage Dockerfile for Active Inference Simulation Lab

# Stage 1: Base development environment
FROM ubuntu:22.04 AS base

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    libeigen3-dev \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libboost-all-dev \
    googletest \
    libgtest-dev \
    doxygen \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    jupyter \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pybind11[global] \
    gymnasium[all] \
    tensorboard

# Set working directory
WORKDIR /workspace

# Stage 2: Build environment
FROM base AS builder

# Copy source code
COPY . /workspace/

# Build C++ library
RUN mkdir -p build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_DOCS=ON && \
    make -j$(nproc)

# Run tests
RUN cd build && ctest --output-on-failure

# Install Python package
RUN pip3 install -e .

# Stage 3: Runtime environment
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libeigen3-dev \
    libopenblas0 \
    liblapack3 \
    libhdf5-103 \
    libboost-system1.74.0 \
    && rm -rf /var/lib/apt/lists/*

# Install minimal Python dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    gymnasium

# Create app user
RUN useradd -m -s /bin/bash appuser

# Copy built library and Python package
COPY --from=builder /workspace/build/src/libactive_inference_core.so /usr/local/lib/
COPY --from=builder /workspace/build/python/*.so /usr/local/lib/python3.10/site-packages/
COPY --from=builder /workspace/src /app/src
COPY --from=builder /workspace/examples /app/examples

# Set ownership
RUN chown -R appuser:appuser /app
RUN ldconfig

# Switch to app user
USER appuser
WORKDIR /app

# Set Python path
ENV PYTHONPATH=/app/src:/usr/local/lib/python3.10/site-packages

# Default command
CMD ["python3", "-c", "import active_inference; print('Active Inference Simulation Lab ready!')"]

# Stage 4: Development environment (final)
FROM base AS development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    clang-format \
    clang-tidy \
    cppcheck \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python development tools
RUN pip3 install --no-cache-dir \
    pylint \
    isort \
    pre-commit \
    jupyter \
    plotly \
    seaborn \
    wandb \
    ray[tune]

# Set up git (for development)
RUN git config --global --add safe.directory /workspace

# Create development user
RUN useradd -m -s /bin/bash developer && \
    usermod -aG sudo developer && \
    echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to development user
USER developer
WORKDIR /workspace

# Set environment variables
ENV PYTHONPATH=/workspace/src
ENV CMAKE_BUILD_TYPE=Debug

# Default command for development
CMD ["/bin/bash"]

# Labels
LABEL org.opencontainers.image.title="Active Inference Simulation Lab"
LABEL org.opencontainers.image.description="Lightweight toolkit for building active inference agents"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.authors="Active Inference Research Team"
LABEL org.opencontainers.image.url="https://github.com/your-org/active-inference-sim-lab"
LABEL org.opencontainers.image.source="https://github.com/your-org/active-inference-sim-lab"
LABEL org.opencontainers.image.licenses="MIT"