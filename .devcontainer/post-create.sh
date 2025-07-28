#!/bin/bash
set -e

echo "🚀 Setting up Active Inference Sim Lab development environment..."

# Update system packages
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libboost-all-dev \
    libeigen3-dev \
    libgtest-dev \
    libgmock-dev \
    libbenchmark-dev \
    clang-format \
    clang-tidy \
    valgrind \
    gdb \
    htop \
    tree \
    jq \
    curl \
    wget \
    unzip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Setup Git configuration
echo "🔧 Configuring Git..."
git config --global core.autocrlf false
git config --global init.defaultBranch main
git config --global pull.rebase false

# Create useful aliases
echo "📝 Setting up shell aliases..."
cat >> ~/.bashrc << 'EOF'

# Active Inference Sim Lab aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Project specific aliases
alias build='make build'
alias test='make test'
alias clean='make clean'
alias format='make format'
alias lint='make lint'
alias docs='make docs'

# Python aliases
alias py='python3'
alias pip='python3 -m pip'
alias pytest='python3 -m pytest'
alias black='python3 -m black'
alias isort='python3 -m isort'
alias mypy='python3 -m mypy'
alias flake8='python3 -m flake8'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git pull'
alias gd='git diff'
alias gb='git branch'
alias gco='git checkout'
alias glog='git log --oneline --graph --decorate'

EOF

cat >> ~/.zshrc << 'EOF'

# Active Inference Sim Lab aliases (same as bash)
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'

# Project specific aliases
alias build='make build'
alias test='make test'
alias clean='make clean'
alias format='make format'
alias lint='make lint'
alias docs='make docs'

# Python aliases
alias py='python3'
alias pip='python3 -m pip'
alias pytest='python3 -m pytest'
alias black='python3 -m black'
alias isort='python3 -m isort'
alias mypy='python3 -m mypy'
alias flake8='python3 -m flake8'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git pull'
alias gd='git diff'
alias gb='git branch'
alias gco='git checkout'
alias glog='git log --oneline --graph --decorate'

EOF

# Setup development environment variables
echo "🌍 Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'

# Development environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CMAKE_BUILD_TYPE=Release
export NUM_CORES=$(nproc)
export CUDA_VISIBLE_DEVICES=""

EOF

cat >> ~/.zshrc << 'EOF'

# Development environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CMAKE_BUILD_TYPE=Release
export NUM_CORES=$(nproc)
export CUDA_VISIBLE_DEVICES=""

EOF

# Create useful directories
echo "📁 Creating development directories..."
mkdir -p ~/.config/matplotlib
mkdir -p ~/.cache/pip
mkdir -p ~/workspace/experiments
mkdir -p ~/workspace/models
mkdir -p ~/workspace/data

# Setup matplotlib backend for headless operation
echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc

# Install additional development tools
echo "🛠️ Installing additional development tools..."
pip install --user \
    ipython \
    jupyterlab \
    notebook \
    ipywidgets \
    matplotlib \
    seaborn \
    plotly \
    bokeh

# Setup Jupyter extensions
echo "📊 Setting up Jupyter..."
jupyter lab build

# Create a sample notebook
mkdir -p notebooks/getting-started
cat > notebooks/getting-started/welcome.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Welcome to Active Inference Sim Lab!\n",
    "\n",
    "This notebook will help you get started with the development environment.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "print(f\"Python version: {sys.version}\")\n",
    "print(f\"Python path: {sys.executable}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test basic imports\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import torch\n",
    "\n",
    "print(f\"NumPy version: {np.__version__}\")\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "  1. Run 'make build' to build the project"
echo "  2. Run 'make test' to run tests"
echo "  3. Open the welcome notebook: notebooks/getting-started/welcome.ipynb"
echo "  4. Check out the examples in the examples/ directory"
echo ""
echo "📚 Useful commands:"
echo "  make help    - Show all available make targets"
echo "  make format  - Format code with black and clang-format"
echo "  make lint    - Run linting checks"
echo "  make docs    - Build documentation"
echo ""