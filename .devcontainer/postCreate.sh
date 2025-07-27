#!/bin/bash

# Post-create script for development environment setup

echo "🚀 Setting up Active Inference Simulation Lab development environment..."

# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install C++ development tools
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    gcc-10 \
    g++-10 \
    clang-12 \
    clang-format-12 \
    clang-tidy-12 \
    gdb \
    valgrind

# Install scientific computing libraries
sudo apt-get install -y \
    libeigen3-dev \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libboost-all-dev

# Install Python development dependencies
pip install --upgrade pip setuptools wheel

# Install core Python packages
pip install \
    numpy \
    scipy \
    matplotlib \
    jupyter \
    ipykernel \
    pytest \
    pytest-cov \
    black \
    flake8 \
    pylint \
    mypy \
    isort \
    pre-commit \
    pybind11[global]

# Install machine learning packages
pip install \
    gymnasium[all] \
    mujoco \
    tensorboard \
    wandb \
    plotly

# Install development tools
pip install \
    sphinx \
    sphinx-rtd-theme \
    breathe \
    nbsphinx

# Setup git hooks
if [ -f .pre-commit-config.yaml ]; then
    pre-commit install
fi

# Create common directories
mkdir -p \
    src \
    include \
    tests \
    examples \
    docs/source \
    build \
    data \
    notebooks

# Install GitHub CLI
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh -y

# Configure git (if not already configured)
if [ -z "$(git config --global user.name)" ]; then
    echo "⚠️  Git user.name not configured. Please run:"
    echo "   git config --global user.name 'Your Name'"
    echo "   git config --global user.email 'your.email@example.com'"
fi

echo "✅ Development environment setup complete!"
echo "🔧 Available tools:"
echo "   - C++: gcc-10, clang-12, cmake, ninja"
echo "   - Python: python3.11, pip, pytest, black, pylint"
echo "   - ML: gymnasium, mujoco, tensorboard"
echo "   - Docs: sphinx, jupyter"
echo "   - Git: git, gh (GitHub CLI), pre-commit"

echo ""
echo "🚀 Ready to build active inference agents!"