# Multi-stage build for Active Inference Simulation Lab
FROM ubuntu:24.04 as builder

# Set build arguments
ARG PYTHON_VERSION=3.11
ARG CMAKE_BUILD_TYPE=Release

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    libeigen3-dev \
    libopenblas-dev \
    liblapack-dev \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy source code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Build C++ components
RUN mkdir -p build && cd build && \
    cmake .. \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DUSE_OPENMP=ON \
        -DUSE_EIGEN=ON && \
    ninja && \
    ninja install

# Production stage
FROM ubuntu:24.04 as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libopenblas0 \
    liblapack3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r aiuser && useradd -r -g aiuser -s /bin/bash aiuser

# Set up Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN python -m pip install --no-cache-dir --upgrade pip

# Copy built artifacts from builder stage
COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /usr/local/include/ /usr/local/include/
COPY --from=builder /workspace/dist/ /tmp/dist/

# Install the package
RUN pip install --no-cache-dir /tmp/dist/*.whl && rm -rf /tmp/dist/

# Set up working directory
WORKDIR /app
RUN chown -R aiuser:aiuser /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import active_inference; print('OK')" || exit 1

# Switch to non-root user
USER aiuser

# Set environment variables
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages:${PYTHONPATH}
ENV OMP_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-c", "import active_inference; print('Active Inference Sim Lab container ready!')"]

# Development stage (for development containers)
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-xdist \
    black \
    flake8 \
    mypy \
    jupyter \
    matplotlib \
    plotly \
    dash \
    tensorboard

# Install pre-commit hooks
RUN pip install --no-cache-dir pre-commit

# Create workspace
WORKDIR /workspace
RUN chown -R 1000:1000 /workspace

# Expose common ports
EXPOSE 8888 6006 8050

# Default development command
CMD ["bash"]