# Makefile for Active Inference Simulation Lab
# Provides convenient build targets for the C++/Python hybrid project

# Variables
BUILD_DIR := build
CMAKE_BUILD_TYPE := Release
PYTHON := python3
PIP := pip
NUM_CORES := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Python virtual environment
VENV_DIR := venv
VENV_ACTIVATE := $(VENV_DIR)/bin/activate

# Default target
.PHONY: all
all: build

# Help target
.PHONY: help
help:
	@echo "Active Inference Simulation Lab - Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  help              Show this help message"
	@echo "  build             Build C++ core and Python package"
	@echo "  build-cpp         Build only C++ core library"
	@echo "  build-python      Build only Python package"
	@echo "  build-debug       Build with debug symbols"
	@echo "  test              Run all tests"
	@echo "  test-cpp          Run C++ tests only"
	@echo "  test-python       Run Python tests only"
	@echo "  benchmark         Run benchmarks"
	@echo "  install           Install package in development mode"
	@echo "  install-deps      Install Python dependencies"
	@echo "  clean             Clean build artifacts"
	@echo "  clean-all         Clean everything including venv"
	@echo "  format            Format code with black and clang-format"
	@echo "  lint              Run linting checks"
	@echo "  docs              Build documentation"
	@echo "  docker            Build Docker image"
	@echo "  venv              Create Python virtual environment"
	@echo "  pre-commit        Install pre-commit hooks"

# Build targets
.PHONY: build
build: build-cpp build-python

.PHONY: build-cpp
build-cpp:
	@echo "Building C++ core library..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DBUILD_PYTHON_BINDINGS=ON \
		-DBUILD_TESTS=ON \
		-DUSE_OPENMP=ON \
		-DUSE_EIGEN=ON
	@cd $(BUILD_DIR) && make -j$(NUM_CORES)

.PHONY: build-python
build-python:
	@echo "Building Python package..."
	@$(PYTHON) -m pip install -e .

.PHONY: build-debug
build-debug:
	@echo "Building with debug symbols..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=Debug \
		-DBUILD_PYTHON_BINDINGS=ON \
		-DBUILD_TESTS=ON \
		-DUSE_OPENMP=ON \
		-DUSE_EIGEN=ON
	@cd $(BUILD_DIR) && make -j$(NUM_CORES)
	@$(PYTHON) -m pip install -e .

# Test targets
.PHONY: test
test: test-cpp test-python

.PHONY: test-cpp
test-cpp: build-cpp
	@echo "Running C++ tests..."
	@cd $(BUILD_DIR) && ctest --output-on-failure

.PHONY: test-python
test-python:
	@echo "Running Python tests..."
	@$(PYTHON) -m pytest tests/ -v

.PHONY: test-integration
test-integration:
	@echo "Running integration tests..."
	@$(PYTHON) -m pytest tests/integration/ -v

.PHONY: benchmark
benchmark: build-cpp
	@echo "Running benchmarks..."
	@cd $(BUILD_DIR) && ./run_benchmarks

# Installation targets
.PHONY: install
install: build
	@echo "Installing package in development mode..."
	@$(PYTHON) -m pip install -e .[dev]

.PHONY: install-deps
install-deps:
	@echo "Installing Python dependencies..."
	@$(PYTHON) -m pip install -r requirements-dev.txt

# Virtual environment
.PHONY: venv
venv:
	@echo "Creating Python virtual environment..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@. $(VENV_ACTIVATE) && pip install --upgrade pip setuptools wheel
	@echo "Virtual environment created. Activate with: source $(VENV_ACTIVATE)"

# Clean targets
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -rf dist/
	@rm -rf *.egg-info/
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.so" -delete

.PHONY: clean-all
clean-all: clean
	@echo "Cleaning everything..."
	@rm -rf $(VENV_DIR)
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf htmlcov/
	@rm -rf docs/build/

# Code quality targets
.PHONY: format
format:
	@echo "Formatting Python code..."
	@$(PYTHON) -m black src/ tests/ examples/ scripts/
	@$(PYTHON) -m isort src/ tests/ examples/ scripts/
	@echo "Formatting C++ code..."
	@find cpp/ include/ -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-format -i

.PHONY: lint
lint:
	@echo "Running Python linting..."
	@$(PYTHON) -m flake8 src/ tests/ examples/ scripts/
	@$(PYTHON) -m mypy src/
	@$(PYTHON) -m bandit -r src/
	@echo "Running C++ linting..."
	@find cpp/ include/ -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-tidy

# Security scanning targets
.PHONY: security-scan
security-scan:
	@echo "Running comprehensive security scans..."
	@$(PYTHON) -m bandit -r src/ -f json -o bandit-report.json
	@$(PYTHON) -m safety check --json --output safety-report.json
	@detect-secrets scan --baseline .secrets.baseline

.PHONY: vulnerability-check
vulnerability-check:
	@echo "Checking for known vulnerabilities..."
	@$(PYTHON) -m safety check
	@$(PYTHON) -m pip-audit

# Documentation
.PHONY: docs
docs:
	@echo "Building documentation..."
	@cd docs && make html
	@echo "Documentation built in docs/build/html/"

.PHONY: docs-clean
docs-clean:
	@echo "Cleaning documentation..."
	@cd docs && make clean

# Docker
.PHONY: docker
docker:
	@echo "Building Docker image..."
	@docker build -t active-inference-sim-lab:latest .

.PHONY: docker-dev
docker-dev:
	@echo "Building development Docker image..."
	@docker build -f Dockerfile.dev -t active-inference-sim-lab:dev .

# Pre-commit hooks
.PHONY: pre-commit
pre-commit:
	@echo "Installing pre-commit hooks..."
	@$(PYTHON) -m pre_commit install

# Package building and distribution
.PHONY: build-package
build-package: clean
	@echo "Building distribution packages..."
	@$(PYTHON) -m build

.PHONY: upload-test
upload-test: build-package
	@echo "Uploading to TestPyPI..."
	@$(PYTHON) -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: upload
upload: build-package
	@echo "Uploading to PyPI..."
	@$(PYTHON) -m twine upload dist/*

# Performance profiling
.PHONY: profile-python
profile-python:
	@echo "Running Python profiling..."
	@$(PYTHON) -m cProfile -o profile.stats examples/profile_example.py
	@$(PYTHON) -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

.PHONY: profile-cpp
profile-cpp: build-debug
	@echo "Running C++ profiling..."
	@cd $(BUILD_DIR) && valgrind --tool=callgrind ./run_tests
	@echo "Use kcachegrind to view callgrind output"

# Memory checking
.PHONY: memcheck
memcheck: build-debug
	@echo "Running memory check..."
	@cd $(BUILD_DIR) && valgrind --leak-check=full --show-leak-kinds=all ./run_tests

# Coverage
.PHONY: coverage
coverage:
	@echo "Running coverage analysis..."
	@$(PYTHON) -m pytest tests/ --cov=active_inference --cov-report=html --cov-report=term
	@echo "Coverage report in htmlcov/index.html"

# Continuous integration targets
.PHONY: ci-test
ci-test: install-deps build test lint

.PHONY: ci-benchmark
ci-benchmark: build benchmark

# Release targets
.PHONY: release-patch
release-patch:
	@echo "Creating patch release..."
	@scripts/release.sh patch

.PHONY: release-minor
release-minor:
	@echo "Creating minor release..."
	@scripts/release.sh minor

.PHONY: release-major
release-major:
	@echo "Creating major release..."
	@scripts/release.sh major

# Info targets
.PHONY: info
info:
	@echo "Project Information:"
	@echo "  Python: $(shell $(PYTHON) --version)"
	@echo "  Pip: $(shell $(PIP) --version)"
	@echo "  Build directory: $(BUILD_DIR)"
	@echo "  Number of cores: $(NUM_CORES)"
	@echo "  Virtual environment: $(VENV_DIR)"